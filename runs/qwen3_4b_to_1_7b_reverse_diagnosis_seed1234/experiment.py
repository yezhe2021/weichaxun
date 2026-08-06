from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from assets import rows_for
from data import cuda, load_json, load_model, normalize_answer, progress, representation_loss, save_json, seed_all, token_f1, tokenizer
from receiver import answer_logits, ce_loss, generate
from writers import copy_front_into_learnable, load_writer, make_writer, optimizer_groups


def root(cfg, mode, name): return Path(cfg["work_dir"]) / "artifacts" / mode / name
def scales_for(cfg, mode): return torch.load(root(cfg, mode, "scales.pt"), map_location="cpu", weights_only=False)


class Store:
    def __init__(self, cfg): self.cfg, self.cache = cfg, {}
    def _load(self, key, path):
        if key not in self.cache:
            if len(self.cache) > 3: self.cache.clear()
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.cache[key]
    def source4(self, split, sample_id): return self._load(("4", split, sample_id), Path(self.cfg["work_dir"]) / "cache" / "source4_full" / split / f"{sample_id}.pt")
    def source17(self, split, sample_id):
        base = Path(self.cfg["work_dir"]) / "cache" / "source1_7" / split / f"{sample_id}.pt"
        extra = Path(self.cfg["work_dir"]) / "cache" / "source1_7_eval128" / f"{sample_id}.pt"
        return self._load(("17", split, sample_id), base if base.exists() else extra)
    def positions(self, split, sample_id):
        path = Path(self.cfg["work_dir"]) / "cache" / "target4" / split / f"{sample_id}.pt"
        return self._load(("positions", split, sample_id), path)["positions"]


def save_writer(path, writer, extra=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"writer": {name: value.detach().cpu() for name, value in writer.state_dict().items()}}
    if extra: payload.update(extra)
    torch.save(payload, path)


def writer_for(cfg, mode, kind, checkpoint=None, front_checkpoint=None):
    writer = make_writer(kind, scales_for(cfg, mode), cfg).to(cuda())
    if checkpoint is not None: load_writer(writer, checkpoint)
    elif front_checkpoint is not None: copy_front_into_learnable(writer, front_checkpoint)
    return writer


@torch.no_grad()
def generate_text(model, tok, ids, cfg):
    tensor = torch.tensor([ids], dtype=torch.long, device=cuda()); positions = torch.arange(len(ids), device=cuda()).unsqueeze(0)
    output = model(input_ids=tensor, attention_mask=torch.ones_like(tensor), position_ids=positions, use_cache=True)
    past, token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True); generated = []
    for position in range(len(ids), len(ids) + cfg["max_new_tokens"]):
        value = int(token.item())
        if value == tok.eos_token_id: break
        generated.append(value)
        output = model(input_ids=token, attention_mask=torch.ones(1, past.get_seq_length() + 1, dtype=torch.long, device=cuda()), position_ids=torch.tensor([[position]], device=cuda()), past_key_values=past, use_cache=True)
        past, token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True)
    return tok.decode(generated, skip_special_tokens=True).strip()


def text_nll(model, sample, prompt_ids):
    target = sample["answer_token_ids"]; current = prompt_ids + target[:-1]
    ids = torch.tensor([current], dtype=torch.long, device=cuda()); positions = torch.arange(len(current), device=cuda()).unsqueeze(0)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions, use_cache=False).logits[0, len(prompt_ids)-1:len(prompt_ids)-1+len(target)].float()
    return F.cross_entropy(logits, torch.tensor(target, device=cuda())).item()


def metric(sample, condition, prediction, nll):
    return {"sample_id": sample["id"], "type": sample["type"], "condition": condition, "answer": sample["answer"], "prediction": prediction, "em": float(normalize_answer(prediction) == normalize_answer(sample["answer"])), "f1": token_f1(prediction, sample["answer"]), "nll": nll}


def aggregate(rows):
    output = {}
    for condition in sorted({x["condition"] for x in rows}):
        values = [x for x in rows if x["condition"] == condition]
        bridge = [x for x in values if x["type"] == "bridge"]; comparison = [x for x in values if x["type"] == "comparison"]
        output[condition] = {"count": len(values), "em": sum(x["em"] for x in values)/len(values), "f1": sum(x["f1"] for x in values)/len(values), "bridge_f1": sum(x["f1"] for x in bridge)/len(bridge) if bridge else None, "comparison_f1": sum(x["f1"] for x in comparison)/len(comparison) if comparison else None, "nll": sum(x["nll"] for x in values)/len(values)}
    return output


@torch.no_grad()
def oracle(cfg, mode):
    rows = rows_for(cfg, mode)["test"]; store = Store(cfg); model = load_model(cfg["model_4b"], cfg); tok = tokenizer(cfg["model_4b"]); output = []
    full36_matches = 0
    try:
        for index, sample in enumerate(rows, 1):
            qpred = generate_text(model, tok, sample["question_only_ids"], cfg); output.append(metric(sample, "a0_4b_question_only", qpred, text_nll(model, sample, sample["question_only_ids"])))
            tpred = generate_text(model, tok, sample["full_input_ids"], cfg); output.append(metric(sample, "a1_4b_full_context_text", tpred, text_nll(model, sample, sample["full_input_ids"])))
            native4 = store.source4("test", sample["id"]); k4, v4 = native4["pre_key"].to(cuda()), native4["value"].to(cuda())
            nll, _, _ = ce_loss(model, sample, k4, v4); pred, _ = generate(model, tok, sample, cfg, k4, v4); output.append(metric(sample, "a2_4b_native_full36", pred, nll.item())); full36_matches += int(pred == tpred)
            nll, _, _ = ce_loss(model, sample, k4[:28], v4[:28], skip=True); pred, _ = generate(model, tok, sample, cfg, k4[:28], v4[:28], skip=True); output.append(metric(sample, "a3_4b_native_partial28", pred, nll.item()))
            native17 = store.source17("test", sample["id"]); k17, v17 = native17["pre_key"].to(cuda()), native17["value"].to(cuda())
            nll, _, _ = ce_loss(model, sample, k17, v17, skip=True); pred, _ = generate(model, tok, sample, cfg, k17, v17, skip=True); output.append(metric(sample, "a4_1_7b_raw_native_partial28", pred, nll.item()))
            if index % 8 == 0: progress(f"{mode}: oracle {index}/{len(rows)}")
    finally:
        del model; torch.cuda.empty_cache()
    summary = aggregate(output); summary["a2_full36_generation_match_vs_text"] = full36_matches / len(rows)
    forward_rows = load_json(Path(cfg["forward_experiment_dir"]) / "artifacts" / "eval128" / "partial28_skip_ce.json")
    allowed = {x["id"] for x in rows}; forward_rows = [x for x in forward_rows if x["sample_id"] in allowed and x["condition"].endswith("_correct")]
    forward_f1 = sum(x["f1"] for x in forward_rows) / len(forward_rows)
    summary["a5_existing_1_7b_writer_partial28_skip_ce"] = {"count": len(forward_rows), "f1": forward_f1, "source": "frozen current eval128 result"}
    summary["protocol_loss"] = summary["a2_4b_native_full36"]["f1"] - summary["a3_4b_native_partial28"]["f1"]
    summary["forward_gap"] = summary["a3_4b_native_partial28"]["f1"] - forward_f1
    out = root(cfg, mode, "oracle_4b_native_partial28"); out.mkdir(parents=True, exist_ok=True); save_json(out / "per_sample.json", output); save_json(out / "summary.json", summary)


@torch.no_grad()
def receiver17_baseline(cfg, mode):
    rows = rows_for(cfg, mode)["test"]; store = Store(cfg); model = load_model(cfg["model_1_7b"], cfg); tok = tokenizer(cfg["model_1_7b"]); output = []; matches = 0
    try:
        for index, sample in enumerate(rows, 1):
            qpred = generate_text(model, tok, sample["question_only_ids"], cfg); output.append(metric(sample, "b1_q_1_7b_question_only", qpred, text_nll(model, sample, sample["question_only_ids"])))
            tpred = generate_text(model, tok, sample["full_input_ids"], cfg); output.append(metric(sample, "b1_f_1_7b_full_context_text", tpred, text_nll(model, sample, sample["full_input_ids"])))
            native = store.source17("test", sample["id"]); key, value = native["pre_key"].to(cuda()), native["value"].to(cuda())
            nll, _, _ = ce_loss(model, sample, key, value); pred, _ = generate(model, tok, sample, cfg, key, value); output.append(metric(sample, "b1_n_1_7b_native_cache", pred, nll.item())); matches += int(pred == tpred)
            if index % 8 == 0: progress(f"{mode}: 1.7B baseline {index}/{len(rows)}")
    finally:
        del model; torch.cuda.empty_cache()
    summary = aggregate(output); summary["native_generation_match_vs_text"] = matches / len(rows)
    out = root(cfg, mode, "receiver17_baseline"); out.mkdir(parents=True, exist_ok=True); save_json(out / "per_sample.json", output); save_json(out / "summary.json", summary)


def sample_rep(cfg, writer, store, split, sample):
    positions = store.positions(split, sample["id"]); source, target = store.source4(split, sample["id"]), store.source17(split, sample["id"])
    pred_k, pred_v = writer(source["pre_key"][:, positions].to(cuda()), source["value"][:, positions].to(cuda()))
    target_k, target_v = target["pre_key"][:, positions].to(cuda()), target["value"][:, positions].to(cuda())
    loss, metrics = representation_loss(pred_k, pred_v, target_k, target_v, cfg["stage_a_cosine_weight"])
    return loss, metrics, pred_k, pred_v


@torch.no_grad()
def rep_validation(cfg, writer, store, split, samples):
    writer.eval(); losses, all_metrics = [], []
    for sample in samples:
        loss, metrics, _, _ = sample_rep(cfg, writer, store, split, sample); losses.append(loss.item()); all_metrics.append(metrics)
    layers = [{"layer": layer, **{key: sum(row[layer][key] for row in all_metrics)/len(all_metrics) for key in ("k_nmse", "v_nmse", "k_cosine", "v_cosine")}} for layer in range(cfg["target_layers"])]
    writer.train(); return sum(losses)/len(losses), layers


def train_stage_a(cfg, mode, kind, name, overfit=False, initial=None):
    rows, store = rows_for(cfg, mode), Store(cfg); front_checkpoint = initial if kind == "learnable" else None
    writer = writer_for(cfg, mode, kind, checkpoint=initial if kind != "learnable" else None, front_checkpoint=front_checkpoint).train()
    out = root(cfg, mode, name); out.mkdir(parents=True, exist_ok=True)
    if mode == "smoke": maximum, grad_acc = cfg["smoke_updates"], cfg["smoke_gradient_accumulation"]
    elif overfit: maximum, grad_acc = cfg["overfit_updates"], cfg["gradient_accumulation"]
    elif name.startswith("r1_continued") or name.startswith("r2_"): maximum, grad_acc = cfg["e2_stage_a_updates"], cfg["gradient_accumulation"]
    else: maximum, grad_acc = cfg["stage_a_updates"], cfg["gradient_accumulation"]
    samples = list(rows["train"][:min(cfg["overfit_samples"], len(rows["train"]))] if overfit else rows["train"]); validation, split = (samples, "train") if overfit else (rows["validation"], "validation")
    e2 = name.startswith("r1_continued") or name.startswith("r2_"); linear_lr = cfg["e2_stage_a_linear_lr"] if e2 else cfg["stage_a_linear_lr"]
    optimizer = AdamW(optimizer_groups(writer, linear_lr, cfg["e2_stage_a_depth_lr"]), weight_decay=cfg["weight_decay"])
    initial_loss, _ = rep_validation(cfg, writer, store, split, validation); best = float("inf"); history, evaluations = [], []; cursor = epoch = 0
    ever_grad = {n: False for n, p in writer.named_parameters() if p.requires_grad}; eye = torch.eye(cfg["feature_dim"], device=cuda())[None]; initial_delta = max((writer.weight_k-eye).abs().max().item(), (writer.weight_v-eye).abs().max().item())
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for update in range(1, maximum + 1):
        optimizer.zero_grad(set_to_none=True); batch = []
        for _ in range(grad_acc):
            if cursor == 0: random.Random(cfg["seed"] + epoch).shuffle(samples); epoch += 1
            sample = samples[cursor]; cursor = (cursor + 1) % len(samples); loss, *_ = sample_rep(cfg, writer, store, "train", sample)
            if not torch.isfinite(loss): raise RuntimeError(f"{name}: non-finite loss")
            (loss/grad_acc).backward(); batch.append(loss.item())
        for parameter_name, parameter in writer.named_parameters():
            if parameter.requires_grad and parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().max().item() > 0: ever_grad[parameter_name] = True
        norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"])
        if not torch.isfinite(norm): raise RuntimeError(f"{name}: non-finite gradient")
        optimizer.step(); history.append({"update": update, "loss": sum(batch)/len(batch), "gradient_norm": norm.item()})
        interval = 1 if mode == "smoke" else cfg["stage_a_eval_every"]
        if update % interval == 0 or update == maximum:
            score, layers = rep_validation(cfg, writer, store, split, validation); selected = score < best; evaluations.append({"update": update, "validation_loss": score, "selected": selected})
            if selected: best = score; save_writer(out / "best.pt", writer, {"update": update, "validation_loss": score})
            progress(f"{mode}: {name} {update}/{maximum} validation={score:.6f}")
    final_loss, layers = rep_validation(cfg, writer, store, split, validation); missing = [n for n, ok in ever_grad.items() if not ok]; final_delta = max((writer.weight_k-eye).abs().max().item(), (writer.weight_v-eye).abs().max().item())
    summary = {"kind": kind, "overfit_code_validation_only": overfit, "initial_checkpoint": str(initial) if initial else None, "initial_loss": initial_loss, "final_loss": final_loss, "loss_decreased": final_loss < initial_loss, "all_parameters_received_gradient": not missing, "missing_gradients": missing, "parameters_changed": final_delta > initial_delta, "no_nan_inf": True, "best_validation_loss": best, "training_seconds": time.perf_counter()-started, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    if overfit:
        first = writer_for(cfg, mode, kind, checkpoint=out/"best.pt").eval(); second = writer_for(cfg, mode, kind, checkpoint=out/"best.pt").eval(); sample = validation[0]
        _, _, a, b = sample_rep(cfg, first, store, split, sample); _, _, c, d = sample_rep(cfg, second, store, split, sample); difference = max((a-c).float().abs().max().item(), (b-d).float().abs().max().item()); summary["checkpoint_reload_max_abs"] = difference; summary["checkpoint_save_load_passed"] = difference == 0
        if not summary["loss_decreased"] or missing or not summary["parameters_changed"] or difference != 0: raise RuntimeError(f"{name} implementation validation failed: {summary}")
        del first, second
    save_json(out/"history.json", history); save_json(out/"evaluations.json", evaluations); save_json(out/"final_layer_metrics.json", layers); save_json(out/"summary.json", summary)
    del writer; torch.cuda.empty_cache()


def full_source(writer, store, split, sample, source_id=None):
    record = store.source4(split, source_id or sample["id"]); return writer(record["pre_key"].to(cuda()), record["value"].to(cuda()))


@torch.no_grad()
def validation_scores(cfg, model, tok, writer, store, samples, generation=False):
    writer.eval(); values = []
    for sample in samples:
        key, value = full_source(writer, store, "validation", sample)
        if generation: values.append(token_f1(generate(model, tok, sample, cfg, key, value)[0], sample["answer"]))
        else: values.append(ce_loss(model, sample, key, value)[0].item())
    writer.train(); return sum(values)/len(values)


def train_ce(cfg, mode, kind, name, initial):
    rows, store = rows_for(cfg, mode), Store(cfg); writer = writer_for(cfg, mode, kind, checkpoint=initial).train(); model = load_model(cfg["model_1_7b"], cfg); tok = tokenizer(cfg["model_1_7b"])
    out = root(cfg, mode, name); out.mkdir(parents=True, exist_ok=True); maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["stage_b_updates"]; grad_acc = cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]
    optimizer = AdamW(optimizer_groups(writer, cfg["stage_b_linear_lr"], cfg["e2_stage_b_depth_lr"]), weight_decay=cfg["weight_decay"]); samples = list(rows["train"]); cursor = epoch = 0; history, evaluations, best_nll, best_f1 = [], [], float("inf"), -1.0
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for update in range(1, maximum+1):
        optimizer.zero_grad(set_to_none=True); batch = []
        for _ in range(grad_acc):
            if cursor == 0: random.Random(cfg["seed"]+epoch).shuffle(samples); epoch += 1
            sample = samples[cursor]; cursor = (cursor+1)%len(samples); key, value = full_source(writer, store, "train", sample); loss, _, _ = ce_loss(model, sample, key, value)
            if not torch.isfinite(loss): raise RuntimeError(f"{name}: non-finite CE")
            (loss/grad_acc).backward(); batch.append(loss.item())
        norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]); optimizer.step(); history.append({"update": update, "ce": sum(batch)/len(batch), "gradient_norm": norm.item()}); row = {"update": update}
        interval = 1 if mode == "smoke" else cfg["nll_eval_every"]
        if update % interval == 0 or update == maximum:
            nll = validation_scores(cfg, model, tok, writer, store, rows["validation"]); row["validation_nll"] = nll
            if nll < best_nll: best_nll = nll; save_writer(out/"best_nll.pt", writer, {"update": update, "validation_nll": nll})
        gen_interval = 1 if mode == "smoke" else cfg["generation_eval_every"]
        if update % gen_interval == 0 or update == maximum:
            count = cfg["smoke_generation_eval_samples"] if mode == "smoke" else cfg["generation_eval_samples"]; f1 = validation_scores(cfg, model, tok, writer, store, rows["validation"][:count], True); row["validation_generation_f1"] = f1
            if f1 > best_f1: best_f1 = f1; save_writer(out/"selected.pt", writer, {"update": update, "validation_generation_f1": f1, "selection_rule": "highest validation generation F1"})
        if len(row)>1: evaluations.append(row); progress(f"{mode}: {name} {update}/{maximum}")
    save_json(out/"history.json", history); save_json(out/"evaluations.json", evaluations); save_json(out/"summary.json", {"loss": "answer CE only", "initial_checkpoint": str(initial), "best_validation_nll": best_nll, "best_validation_generation_f1": best_f1, "selection_uses_test": False, "kd_used": False, "representation_loss_used": False, "tether_used": False, "training_seconds": time.perf_counter()-started, "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
    del writer, model; torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_writer(cfg, mode, condition, kind, checkpoint):
    rows, store = rows_for(cfg, mode)["test"], Store(cfg); model = load_model(cfg["model_1_7b"], cfg); tok = tokenizer(cfg["model_1_7b"]); writer = writer_for(cfg, mode, kind, checkpoint=checkpoint).eval(); output = []
    try:
        for shuffled in (False, True):
            label = condition + ("_shuffled" if shuffled else "_correct")
            for index, sample in enumerate(rows, 1):
                key, value = full_source(writer, store, "test", sample, sample["shuffle_id"] if shuffled else sample["id"]); nll, _, _ = ce_loss(model, sample, key, value); prediction, _ = generate(model, tok, sample, cfg, key, value); output.append(metric(sample, label, prediction, nll.item()))
                if index % 16 == 0: progress(f"{mode}: {label} {index}/{len(rows)}")
    finally:
        del writer, model; torch.cuda.empty_cache()
    out = root(cfg, mode, condition); out.mkdir(parents=True, exist_ok=True); save_json(out/"per_sample.json", output); save_json(out/"summary.json", aggregate(output))


@torch.no_grad()
def update0(cfg, mode):
    rows, store = rows_for(cfg, mode), Store(cfg); checkpoint = root(cfg, mode, "r1_stage_a")/"best.pt"; fixed = writer_for(cfg, mode, "front", checkpoint=checkpoint).eval(); learned = writer_for(cfg, mode, "learnable", front_checkpoint=checkpoint).eval(); model = load_model(cfg["model_1_7b"], cfg); tok = tokenizer(cfg["model_1_7b"])
    max_kv = max_logits = max_kl = 0.0; top1 = generations = 0; details = []
    for sample in rows["validation"]:
        source = store.source4("validation", sample["id"]); sk, sv = source["pre_key"].to(cuda()), source["value"].to(cuda()); fk, fv = fixed(sk, sv); lk, lv = learned(sk, sv)
        kv = max((fk-lk).float().abs().max().item(), (fv-lv).float().abs().max().item()); fl, _ = answer_logits(model, sample, fk, fv); ll, _ = answer_logits(model, sample, lk, lv); logits = (fl-ll).abs().max().item(); fp = fl.softmax(-1); kl = (fp*(fl.log_softmax(-1)-ll.log_softmax(-1))).sum(-1).mean().item(); fg = generate(model,tok,sample,cfg,fk,fv)[0]; lg = generate(model,tok,sample,cfg,lk,lv)[0]
        max_kv=max(max_kv,kv); max_logits=max(max_logits,logits); max_kl=max(max_kl,kl); top1+=int(torch.equal(fl.argmax(-1),ll.argmax(-1))); generations+=int(fg==lg); details.append({"sample_id":sample["id"],"kv_max_abs":kv,"logits_max_abs":logits,"kl":kl,"top1_match":torch.equal(fl.argmax(-1),ll.argmax(-1)),"generation_match":fg==lg})
    count=len(rows["validation"]); passed=max_kv<cfg["update0_kv_atol"] and max_logits<cfg["update0_logits_atol"] and max_kl<cfg["update0_kl_atol"] and top1==count and generations==count; report={"passed":passed,"count":count,"max_kv_abs":max_kv,"max_logits_abs":max_logits,"max_kl":max_kl,"top1_matches":top1,"generation_matches":generations,"rows":details}; save_json(root(cfg,mode,"update0.json"),report)
    del fixed,learned,model; torch.cuda.empty_cache()
    if not passed: raise RuntimeError(f"reverse update0 failed: {report}")


def matrix_outputs(cfg, mode):
    writer = writer_for(cfg, mode, "learnable", checkpoint=root(cfg,mode,"r2_ce")/"selected.pt").eval(); matrix=writer.depth_matrix().detach().cpu().float(); delta=writer.depth_delta.detach().cpu().float(); normalized=matrix.abs()/matrix.abs().sum(-1,keepdim=True).clamp_min(1e-12); out=root(cfg,mode,"final_evaluation"); out.mkdir(parents=True,exist_ok=True); torch.save(matrix,out/"depth_matrix.pt"); np.save(out/"depth_matrix.npy",matrix.numpy()); rows=[]
    for layer in range(matrix.shape[0]):
        values,indices=normalized[layer].topk(3); rows.append({"target_layer":layer,"row_l1":matrix[layer].abs().sum().item(),"row_l2":matrix[layer].square().sum().sqrt().item(),"top3":[{"source_layer":int(i),"raw_weight":matrix[layer,i].item(),"absolute_normalized_weight":float(v),"delta_from_initialization":delta[layer,i].item()} for v,i in zip(values,indices)]})
    save_json(out/"depth_matrix_top3.json",rows)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,8)); image=ax.imshow(matrix.numpy(),aspect="auto",cmap="coolwarm"); ax.set_xlabel("4B source layer"); ax.set_ylabel("1.7B target layer"); fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(out/"depth_matrix_heatmap.png",dpi=180); plt.close(fig); del writer


def finalize(cfg, mode):
    names=["r1_front28_f0","r1_front28_ce","r1_continued_f0","r2_learnable_f0","r1_continued_ce","r2_learnable_ce"]; all_rows=[]
    for name in names: all_rows += load_json(root(cfg,mode,name)/"per_sample.json")
    summary=aggregate(all_rows); baseline=load_json(root(cfg,mode,"receiver17_baseline")/"summary.json"); q=baseline["b1_q_1_7b_question_only"]["f1"]; full=baseline["b1_f_1_7b_full_context_text"]["f1"]
    comparisons={}
    for name in names:
        correct,shuffled=summary[name+"_correct"]["f1"],summary[name+"_shuffled"]["f1"]; denominator=full-q; comparisons[name]={"correct_minus_shuffled":correct-shuffled,"correct_minus_question_only":correct-q,"correct_minus_1_7b_full_text":correct-full,"reverse_gain":correct-q,"sender_advantage":correct-full,"recovery_reverse":(correct-q)/denominator if abs(denominator)>1e-12 else None,"recovery_denominator":denominator}
    oracle_summary=load_json(root(cfg,mode,"oracle_4b_native_partial28")/"summary.json"); out=root(cfg,mode,"final_evaluation"); out.mkdir(parents=True,exist_ok=True); save_json(out/"per_sample.json",all_rows); save_json(out/"summary.json",{"conditions":summary,"receiver17_baselines":baseline,"oracle":oracle_summary,"comparisons":comparisons}); matrix_outputs(cfg,mode); save_json(out/"completion.json",{"completed":True,"test_samples":len(rows_for(cfg,mode)["test"]),"kd_used":False,"reader_training":False,"receiver_frozen":True})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--mode",choices=("smoke","development"),required=True); parser.add_argument("action"); args=parser.parse_args(); cfg=load_json(args.config); seed_all(cfg["seed"]); r1=root(cfg,args.mode,"r1_stage_a")/"best.pt"
    actions={
        "oracle":lambda:oracle(cfg,args.mode),"baseline17":lambda:receiver17_baseline(cfg,args.mode),
        "overfit":lambda:train_stage_a(cfg,args.mode,"front","r1_overfit16",True),"stagea_r1":lambda:train_stage_a(cfg,args.mode,"front","r1_stage_a"),
        "f0_r1":lambda:evaluate_writer(cfg,args.mode,"r1_front28_f0","front",r1),"ce_r1":lambda:train_ce(cfg,args.mode,"front","r1_ce",r1),"eval_r1":lambda:evaluate_writer(cfg,args.mode,"r1_front28_ce","front",root(cfg,args.mode,"r1_ce")/"selected.pt"),
        "update0":lambda:update0(cfg,args.mode),"stagea_continued":lambda:train_stage_a(cfg,args.mode,"front","r1_continued_stage_a",initial=r1),"stagea_r2":lambda:train_stage_a(cfg,args.mode,"learnable","r2_stage_a",initial=r1),
        "f0_continued":lambda:evaluate_writer(cfg,args.mode,"r1_continued_f0","front",root(cfg,args.mode,"r1_continued_stage_a")/"best.pt"),"f0_r2":lambda:evaluate_writer(cfg,args.mode,"r2_learnable_f0","learnable",root(cfg,args.mode,"r2_stage_a")/"best.pt"),
        "ce_continued":lambda:train_ce(cfg,args.mode,"front","r1_continued_ce",root(cfg,args.mode,"r1_continued_stage_a")/"best.pt"),"ce_r2":lambda:train_ce(cfg,args.mode,"learnable","r2_ce",root(cfg,args.mode,"r2_stage_a")/"best.pt"),
        "eval_continued":lambda:evaluate_writer(cfg,args.mode,"r1_continued_ce","front",root(cfg,args.mode,"r1_continued_ce")/"selected.pt"),"eval_r2":lambda:evaluate_writer(cfg,args.mode,"r2_learnable_ce","learnable",root(cfg,args.mode,"r2_ce")/"selected.pt"),"finalize":lambda:finalize(cfg,args.mode)}
    actions[args.action]()


if __name__=="__main__":main()
