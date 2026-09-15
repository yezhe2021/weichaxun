from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.models.qwen3_5 import modeling_qwen3_5

from common import (
    Stores, answer_loss, build_alignment_and_cache, compute_scales, cuda, em,
    generate, load_json, load_reader, lora_parameters, native_memory, progress,
    rows_for, save_json, save_lora, seed_all, set_external, set_lora, summarize,
    token_f1, translated_memory, write_results,
)
from q35_anchor_injection import Q35AnchorInjection, post_rope
from q3_to_q35_writer import make_writer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(cfg, mode):
    required = {
        "q35_manifest": Path(cfg["q35_self_dir"]) / "artifacts" / mode / "manifest.json",
        "q35_cache": Path(cfg["q35_self_dir"]) / "cache" / mode,
        "q35_reader": Path(cfg["q35_self_dir"]) / "artifacts" / mode / "reader" / "best.pt",
        "q3_manifest": Path(cfg["q3_r1_dir"]) / "artifacts" / "formal" / "manifest.json",
        "q3_cache": Path(cfg["q3_r1_dir"]) / "cache" / "formal",
    }
    missing = [f"{key}: {path}" for key, path in required.items() if not path.exists()]
    if missing: raise RuntimeError("missing assets: " + "; ".join(missing))
    if cfg["q3_source_layers"] != [3,8,12,17,21,26,30,35]: raise RuntimeError("Qwen3 source map changed")
    if cfg["q35_target_layers"] != [3,7,11,15,19,23,27,31]: raise RuntimeError("Qwen3.5 target map changed")
    report = {
        "passed": True, "direction": "Qwen3-4B -> Qwen3.5-4B",
        "source_layers": cfg["q3_source_layers"], "target_layers": cfg["q35_target_layers"],
        "channel": "Qwen3.5 Full-Attention Anchor8 only",
        "excluded": ["24 DeltaNet Context states", "layer mixer", "depth residual", "calibration", "MLP", "gate", "low-rank residual"],
        "question_enters_sender": False, "reader_sender_independent": True,
        "q35_reader_sha256": sha256(required["q35_reader"]),
        "hard_result_gate": None,
    }
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "protocol_audit.json", report)
    progress(f"{mode}: protocol audit passed")


def memory_for_condition(cfg, store, split, sample, condition, writer=None):
    if condition.get("memory") == "native": return native_memory(store, split, sample, condition.get("kind", "correct"))
    if condition.get("memory") == "translated": return translated_memory(cfg, writer, store, split, sample, condition.get("kind", "correct"))
    return None


@torch.no_grad()
def evaluate(cfg, mode, model, tok, controller, rows, store, conditions, split="test"):
    records = []
    for condition in conditions:
        progress(f"{mode}: evaluate {condition['key']}")
        set_lora(model, condition.get("lora", False))
        writer = condition.get("writer")
        if writer is not None: writer.eval()
        for sample in rows[split]:
            memory = memory_for_condition(cfg, store, split, sample, condition, writer)
            run_mode = condition.get("mode", "anchor")
            nll = answer_loss(cfg, model, tok, controller, sample, memory, run_mode).item()
            prediction = generate(cfg, model, tok, controller, sample, memory, run_mode)
            records.append({
                "id": sample["id"], "type": sample.get("type", ""), "condition": condition["key"],
                "answer": sample["answer"], "prediction": prediction,
                "em": em(prediction, sample["answer"]), "token_f1": token_f1(prediction, sample["answer"]),
                "nll": nll, "reader": condition.get("reader", "base"),
            })
    return records


def p0(cfg, mode):
    seed_all(cfg["seed"]); rows = rows_for(cfg, mode); store = Stores(cfg, mode, rows)
    model, tok, _ = load_reader(cfg, mode); controller = Q35AnchorInjection(model, cfg["q35_target_layers"])
    conditions = [
        {"key":"q35_full_text", "mode":"full_text", "lora":False},
        {"key":"q35_official_hybrid_cache", "mode":"official", "lora":False},
        {"key":"q35_native_anchor8_lora_off", "memory":"native", "lora":False},
        {"key":"q35_native_anchor8_shuffled_lora_off", "memory":"native", "kind":"shuffled", "lora":False},
        {"key":"q35_anchor8_no_memory_lora_off", "lora":False},
    ]
    records = evaluate(cfg, mode, model, tok, controller, rows, store, conditions)
    summary = summarize(records)
    full = summary["q35_full_text"]["token_f1"]
    anchor = summary["q35_native_anchor8_lora_off"]["token_f1"]
    no_memory = summary["q35_anchor8_no_memory_lora_off"]["token_f1"]
    need_reader = anchor < 0.8 * max(full, 1e-8) or anchor <= no_memory + 0.05
    if mode == "smoke": need_reader = True
    summary["decision"] = {"train_reader": need_reader, "smoke_forces_training_path": mode == "smoke"}
    root = Path(cfg["work_dir"]) / "artifacts" / mode / "p0"
    write_results(root, records, summary)
    progress(f"{mode}: P0 completed; train_reader={need_reader}")


@torch.no_grad()
def reader_nll(cfg, model, tok, controller, rows, store):
    model.eval(); values=[]
    for sample in rows["validation"]:
        values.append(answer_loss(cfg, model, tok, controller, sample, native_memory(store,"validation",sample)).item())
    return sum(values)/len(values)


@torch.no_grad()
def reader_generation(cfg, model, tok, controller, rows, store, update):
    model.eval(); records=[]; limit=min(cfg["generation_eval_samples"],len(rows["validation"]))
    for sample in rows["validation"][:limit]:
        for kind in ("correct","shuffled","none"):
            memory = None if kind == "none" else native_memory(store,"validation",sample,kind)
            prediction=generate(cfg,model,tok,controller,sample,memory)
            records.append({"update":update,"id":sample["id"],"kind":kind,"answer":sample["answer"],"prediction":prediction,"em":em(prediction,sample["answer"]),"token_f1":token_f1(prediction,sample["answer"])})
    correct=[x for x in records if x["kind"]=="correct"]
    return records, sum(x["token_f1"] for x in correct)/len(correct)


def p1_reader(cfg, mode):
    p0_summary=load_json(Path(cfg["work_dir"])/"artifacts"/mode/"p0"/"summary.json")
    out=Path(cfg["work_dir"])/"artifacts"/mode/"anchor_reader"; out.mkdir(parents=True,exist_ok=True)
    if not p0_summary["decision"]["train_reader"]:
        save_json(out/"selection.json",{"trained":False,"lora_enabled":False,"proceed_writer":True,"checkpoint":None})
        progress(f"{mode}: P1 skipped; base Reader retained")
        return
    seed_all(cfg["seed"]); rows=rows_for(cfg,mode); store=Stores(cfg,mode,rows)
    model,tok,initial=load_reader(cfg,mode,trainable=True); controller=Q35AnchorInjection(model,cfg["q35_target_layers"])
    params=lora_parameters(model); optimizer=AdamW(params,lr=cfg["reader_lr"],weight_decay=cfg["weight_decay"])
    maximum=cfg["smoke_updates"] if mode=="smoke" else cfg["reader_updates"]
    grad_acc=cfg["smoke_gradient_accumulation"] if mode=="smoke" else cfg["gradient_accumulation"]
    nll_interval=1 if mode=="smoke" else cfg["reader_nll_interval"]
    gen_interval=maximum if mode=="smoke" else cfg["reader_generation_interval"]
    samples=list(rows["train"]); cursor=epoch=0; history=[]; evaluations=[]; generations=[]
    best_nll=float("inf"); best_f1=-1.0; best_gen_update=0; optimizer.zero_grad(set_to_none=True)
    initial_nll=reader_nll(cfg,model,tok,controller,rows,store)
    initial_gen,initial_f1=reader_generation(cfg,model,tok,controller,rows,store,0)
    save_lora(model,out/"best_nll.pt",update=0,validation_nll=initial_nll)
    save_lora(model,out/"best_generation_f1.pt",update=0,validation_generation_f1=initial_f1)
    best_nll,best_f1=initial_nll,initial_f1; generations.extend(initial_gen)
    for update in range(1,maximum+1):
        model.train(); losses=[]
        for _ in range(grad_acc):
            if cursor==0: random.Random(cfg["seed"]+epoch).shuffle(samples); epoch+=1
            sample=samples[cursor]; cursor=(cursor+1)%len(samples)
            loss=answer_loss(cfg,model,tok,controller,sample,native_memory(store,"train",sample))
            (loss/grad_acc).backward(); losses.append(loss.detach().item())
        norm=torch.nn.utils.clip_grad_norm_(params,cfg["gradient_clip"]).item(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        history.append({"update":update,"answer_nll":sum(losses)/len(losses),"gradient_norm":norm})
        if update%nll_interval==0 or update==maximum:
            value=reader_nll(cfg,model,tok,controller,rows,store); selected=value<best_nll
            if selected: best_nll=value; save_lora(model,out/"best_nll.pt",update=update,validation_nll=value)
            evaluations.append({"update":update,"validation_nll":value,"selected_nll":selected})
            progress(f"{mode}: P1 Reader {update}/{maximum}")
        if update%gen_interval==0 or update==maximum:
            generated,value=reader_generation(cfg,model,tok,controller,rows,store,update); generations.extend(generated)
            selected=value>best_f1
            if selected: best_f1=value; best_gen_update=update; save_lora(model,out/"best_generation_f1.pt",update=update,validation_generation_f1=value)
            evaluations[-1].update({"validation_generation_f1":value,"selected_generation":selected})
        save_json(out/"history.json",history); save_json(out/"evaluations.json",evaluations); save_json(out/"generation_snapshots.json",generations)
        if update%gen_interval==0 and update-best_gen_update>=cfg["reader_generation_patience_updates"]: break
    del model; torch.cuda.empty_cache()
    load_checkpoint=out/"best_generation_f1.pt"; model,tok,_=load_reader(cfg,mode,load_checkpoint); controller=Q35AnchorInjection(model,cfg["q35_target_layers"])
    final_records=evaluate(cfg,mode,model,tok,controller,rows,store,[
        {"key":"q35_native_anchor8_reader_correct","memory":"native","lora":True,"reader":"anchor8"},
        {"key":"q35_native_anchor8_reader_shuffled","memory":"native","kind":"shuffled","lora":True,"reader":"anchor8"},
        {"key":"q35_native_anchor8_reader_no_memory","lora":True,"reader":"anchor8"},
    ])
    final_summary=summarize(final_records); correct=final_summary["q35_native_anchor8_reader_correct"]["token_f1"]
    proceed=correct>max(final_summary["q35_native_anchor8_reader_shuffled"]["token_f1"],final_summary["q35_native_anchor8_reader_no_memory"]["token_f1"])
    if mode=="smoke": proceed=True
    write_results(out/"test_signal",final_records,final_summary)
    save_json(out/"selection.json",{"trained":True,"lora_enabled":True,"checkpoint":str(load_checkpoint),"best_nll":best_nll,"best_generation_f1":best_f1,"proceed_writer":proceed,"smoke_forces_proceed":mode=="smoke","initialized_from":str(initial)})


def assets(cfg, mode):
    build_alignment_and_cache(cfg,mode); compute_scales(cfg,mode)
    writer=make_writer(cfg,mode)
    if writer.zero_check()!=0.0: raise RuntimeError("Writer(0) != 0")
    if any(getattr(module,"bias",None) is not None for module in writer.modules()): raise RuntimeError("Writer bias exists")
    progress(f"{mode}: reverse alignment/cache/scales completed")


def rep_components(cfg,writer,store,split,sample):
    sk,sv,mask=store.source_memory(split,sample); target=store.q35(split,sample["id"])
    pred_k,pred_v=writer.standardized(sk.to(cuda()),sv.to(cuda()))
    gold_k=target["pre_key"].to(cuda()).float().reshape(8,-1,1024)/writer.scale_target_k[:,None]
    gold_v=target["value"].to(cuda()).float().reshape(8,-1,1024)/writer.scale_target_v[:,None]
    valid=mask[0].bool().to(cuda()); rows=[]
    for layer in range(8):
        pk,pv,gk,gv=pred_k[layer,valid],pred_v[layer,valid],gold_k[layer,valid],gold_v[layer,valid]
        kn=(pk-gk).square().mean()/gk.square().mean().clamp_min(1e-8); vn=(pv-gv).square().mean()/gv.square().mean().clamp_min(1e-8)
        kc=F.cosine_similarity(pk.flatten(),gk.flatten(),0); vc=F.cosine_similarity(pv.flatten(),gv.flatten(),0)
        rows.append({"k_nmse":kn,"v_nmse":vn,"k_cosine":kc,"v_cosine":vc})
    loss=torch.stack([x["k_nmse"]+x["v_nmse"]+cfg["cosine_weight"]*(2-x["k_cosine"]-x["v_cosine"]) for x in rows]).mean()
    return loss,rows


@torch.no_grad()
def rep_validation(cfg,writer,store,samples,split="validation"):
    writer.eval(); collected=[]; values=[]
    for sample in samples:
        loss,rows=rep_components(cfg,writer,store,split,sample); values.append(loss.item()); collected.append(rows)
    result=[]
    for layer in range(8):
        row={"source_layer":cfg["q3_source_layers"][layer],"target_layer":cfg["q35_target_layers"][layer]}
        for key in ("k_nmse","v_nmse","k_cosine","v_cosine"): row[key]=sum(float(x[layer][key]) for x in collected)/len(collected)
        result.append(row)
    writer.train(); return sum(values)/len(values),result


class QueryCapture:
    def __init__(self, model, layers):
        self.values = {}; self.handles = []
        for index in layers:
            self.handles.append(model.model.layers[index].self_attn.register_forward_pre_hook(
                self._hook(index), with_kwargs=True
            ))

    def _hook(self, index):
        def capture(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = hidden.shape[:-1]
            query, _ = torch.chunk(
                module.q_proj(hidden).view(*shape, -1, module.head_dim * 2), 2, dim=-1
            )
            self.values[index] = module.q_norm(query).detach()
        return capture

    def close(self):
        for handle in self.handles: handle.remove()


def rotate_query(model, query, positions):
    length = len(positions)
    position_ids = torch.tensor([positions], device=query.device)
    position_ids = position_ids[None].expand(4, 1, length)[1:]
    dummy = torch.empty(1, length, model.config.hidden_size, dtype=query.dtype, device=query.device)
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    q = query.transpose(1, 2)
    key = torch.zeros(1, 4, length, 256, dtype=query.dtype, device=query.device)
    rotated, _ = modeling_qwen3_5.apply_rotary_pos_emb(q, key, cos, sin)
    return rotated


@torch.no_grad()
def attention_observations(cfg, writer, store, samples, model, controller, split="validation"):
    writer.eval(); model.eval(); capture = QueryCapture(model, cfg["q35_target_layers"]); all_rows=[]
    try:
        for sample in samples:
            source_k, source_v, source_mask = store.source_memory(split, sample)
            pred_k, pred_v = writer(source_k.to(cuda()), source_v.to(cuda()))
            native = store.q35(split, sample["id"])
            gold_k, gold_v = native["pre_key"].to(cuda()), native["value"].to(cuda())
            mask = source_mask[0].bool().to(cuda())
            native_mem = store.native_memory(split, sample)
            set_external(cfg, controller, model, *native_mem, sample["selected_position_ids"])
            ids = torch.tensor([sample["question_token_ids"]], device=cuda())
            model(ids, attention_mask=torch.ones_like(ids), position_ids=torch.tensor([sample["question_position_ids"]], device=cuda()), use_cache=False)
            controller.assert_usage(cfg["q35_target_layers"]); controller.clear()
            positions = [p for p, keep in zip(sample["selected_position_ids"], mask.tolist()) if keep]
            pred_rotated = post_rope(model, pred_k[:, mask], positions)
            gold_rotated = post_rope(model, gold_k[:, mask], positions)
            rows=[]
            for slot, layer in enumerate(cfg["q35_target_layers"]):
                q = rotate_query(model, capture.values[layer], sample["question_position_ids"])[0]
                pk = pred_rotated[slot][0].repeat_interleave(4, 0)
                gk = gold_rotated[slot][0].repeat_interleave(4, 0)
                pv = pred_v[slot, mask].float().permute(1,0,2).repeat_interleave(4,0)
                gv = gold_v[slot, mask].float().permute(1,0,2).repeat_interleave(4,0)
                logits_p = torch.einsum("hqd,htd->hqt", q.float(), pk.float()) / math.sqrt(256)
                logits_g = torch.einsum("hqd,htd->hqt", q.float(), gk.float()) / math.sqrt(256)
                attn_p, attn_g = logits_p.softmax(-1), logits_g.softmax(-1)
                out_p = torch.einsum("hqt,htd->hqd", attn_p, pv)
                out_g = torch.einsum("hqt,htd->hqd", attn_g, gv)
                rows.append({
                    "route_kl": (attn_g * (attn_g.clamp_min(1e-12).log() - attn_p.clamp_min(1e-12).log())).sum(-1).mean().item(),
                    "attention_output_cosine": F.cosine_similarity(out_p.flatten(), out_g.flatten(), 0).item(),
                })
            all_rows.append(rows)
    finally:
        capture.close(); controller.clear()
    result=[]
    for slot, layer in enumerate(cfg["q35_target_layers"]):
        result.append({
            "target_layer":layer,
            "route_kl":sum(x[slot]["route_kl"] for x in all_rows)/len(all_rows),
            "attention_output_cosine":sum(x[slot]["attention_output_cosine"] for x in all_rows)/len(all_rows),
        })
    writer.train(); return result


def train_stage_a(cfg,mode,overfit=False):
    selection=load_json(Path(cfg["work_dir"])/"artifacts"/mode/"anchor_reader"/"selection.json")
    out=Path(cfg["work_dir"])/"artifacts"/mode/("stage_a_overfit16" if overfit else "stage_a"); out.mkdir(parents=True,exist_ok=True)
    if not selection["proceed_writer"]:
        save_json(out/"summary.json",{"completed":True,"skipped":True,"reason":"Qwen3.5 Anchor8 signal absent"}); return
    seed_all(cfg["seed"]+(10 if overfit else 20)); rows=rows_for(cfg,mode); store=Stores(cfg,mode,rows); writer=make_writer(cfg,mode).train()
    reader_checkpoint,lora_enabled,_=selected_reader(cfg,mode)
    diagnostic_model,_,_=load_reader(cfg,mode,reader_checkpoint); set_lora(diagnostic_model,lora_enabled)
    diagnostic_controller=Q35AnchorInjection(diagnostic_model,cfg["q35_target_layers"])
    optimizer=AdamW(writer.parameters(),lr=cfg["stage_a_lr"],weight_decay=cfg["weight_decay"])
    maximum=cfg["smoke_updates"] if mode=="smoke" else (cfg["stage_a_overfit_updates"] if overfit else cfg["stage_a_updates"])
    grad_acc=cfg["smoke_gradient_accumulation"] if mode=="smoke" else cfg["gradient_accumulation"]
    interval=1 if mode=="smoke" else cfg["stage_a_eval_interval"]
    samples=list(rows["train"][:cfg["stage_a_overfit_samples"]] if overfit else rows["train"]); cursor=epoch=0; history=[]; evaluations=[]; best=float("inf")
    initial,_=rep_validation(
        cfg, writer, store,
        samples if overfit else rows["validation"],
        "train" if overfit else "validation",
    )
    optimizer.zero_grad(set_to_none=True)
    for update in range(1,maximum+1):
        losses=[]
        for _ in range(grad_acc):
            if cursor==0: random.Random(cfg["seed"]+epoch).shuffle(samples); epoch+=1
            sample=samples[cursor]; cursor=(cursor+1)%len(samples); loss,_=rep_components(cfg,writer,store,"train",sample)
            (loss/grad_acc).backward(); losses.append(loss.detach().item())
        norm=torch.nn.utils.clip_grad_norm_(writer.parameters(),cfg["gradient_clip"]).item(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        history.append({"update":update,"representation_loss":sum(losses)/len(losses),"gradient_norm":norm})
        if update%interval==0 or update==maximum:
            validation_samples=samples if overfit else rows["validation"]
            value,per_layer=rep_validation(cfg,writer,store,validation_samples,"train" if overfit else "validation"); selected=value<best
            observed=attention_observations(
                cfg,writer,store,
                validation_samples[:min(cfg["diagnostic_samples"],len(validation_samples))],
                diagnostic_model,diagnostic_controller,"train" if overfit else "validation",
            )
            for row,extra in zip(per_layer,observed): row.update({"route_kl":extra["route_kl"],"attention_output_cosine":extra["attention_output_cosine"]})
            if selected:
                best=value; torch.save({"writer":writer.state_dict(),"update":update,"validation_representation_loss":value},out/"best.pt")
            evaluations.append({"update":update,"representation_loss":value,"selected":selected,"per_layer":per_layer})
            save_json(out/"history.json",history); save_json(out/"evaluations.json",evaluations); progress(f"{mode}: {'Stage A overfit' if overfit else 'Stage A'} {update}/{maximum}")
    passed=math.isfinite(best) and best<initial
    if mode=="smoke": passed=True
    save_json(out/"summary.json",{"completed":True,"overfit":overfit,"initial_loss":initial,"best_loss":best,"passed":passed,"only_loss":"standardized K/V NMSE + cosine"})
    del diagnostic_model; torch.cuda.empty_cache()


def selected_reader(cfg,mode):
    selection=load_json(Path(cfg["work_dir"])/"artifacts"/mode/"anchor_reader"/"selection.json")
    return selection.get("checkpoint"),selection.get("lora_enabled",False),selection


@torch.no_grad()
def functional_nll(cfg,writer,model,tok,controller,rows,store):
    writer.eval(); model.eval(); values=[]
    for sample in rows["validation"]: values.append(answer_loss(cfg,model,tok,controller,sample,translated_memory(cfg,writer,store,"validation",sample)).item())
    writer.train(); return sum(values)/len(values)


@torch.no_grad()
def functional_generation(cfg,writer,model,tok,controller,rows,store,update):
    writer.eval(); model.eval(); records=[]; limit=min(cfg["generation_eval_samples"],len(rows["validation"]))
    for sample in rows["validation"][:limit]:
        prediction=generate(cfg,model,tok,controller,sample,translated_memory(cfg,writer,store,"validation",sample))
        records.append({"update":update,"id":sample["id"],"answer":sample["answer"],"prediction":prediction,"em":em(prediction,sample["answer"]),"token_f1":token_f1(prediction,sample["answer"])})
    return records,sum(x["token_f1"] for x in records)/len(records)


def parameter_distance(writer,stage_a_state):
    values=[]
    for name in ("feature_k","feature_v"):
        current=getattr(writer,name).float(); base=stage_a_state[name].to(current.device).float()
        values.append(((current-base).square().sum()/base.square().sum().clamp_min(1e-8)).sqrt())
    return torch.stack(values).mean().item()


def train_stage_b(cfg,mode,variant):
    checkpoint,lora_enabled,selection=selected_reader(cfg,mode); out=Path(cfg["work_dir"])/"artifacts"/mode/f"stage_b_{variant}"; out.mkdir(parents=True,exist_ok=True)
    if not selection["proceed_writer"]:
        save_json(out/"summary.json",{"completed":True,"skipped":True}); return
    seed_all(cfg["seed"]+(30 if variant=="f1" else 40)); rows=rows_for(cfg,mode); store=Stores(cfg,mode,rows)
    model,tok,_=load_reader(cfg,mode,checkpoint); set_lora(model,lora_enabled); controller=Q35AnchorInjection(model,cfg["q35_target_layers"])
    stage_a_path=Path(cfg["work_dir"])/"artifacts"/mode/"stage_a"/"best.pt"; writer=make_writer(cfg,mode,stage_a_path if variant=="f2" else None).train()
    stage_a_state=torch.load(stage_a_path,map_location="cpu",weights_only=False)["writer"]
    optimizer=AdamW(writer.parameters(),lr=cfg["stage_b_lr"],weight_decay=cfg["weight_decay"])
    maximum=cfg["smoke_updates"] if mode=="smoke" else cfg["stage_b_updates"]
    grad_acc=cfg["smoke_gradient_accumulation"] if mode=="smoke" else cfg["gradient_accumulation"]
    nll_interval=1 if mode=="smoke" else cfg["stage_b_nll_interval"]; gen_interval=maximum if mode=="smoke" else cfg["stage_b_generation_interval"]
    samples=list(rows["train"]); cursor=epoch=0; history=[]; evaluations=[]; generations=[]; best_nll=float("inf"); best_f1=-1.; best_gen_update=0
    nll=functional_nll(cfg,writer,model,tok,controller,rows,store); gen,f1=functional_generation(cfg,writer,model,tok,controller,rows,store,0)
    torch.save({"writer":writer.state_dict(),"update":0,"validation_nll":nll},out/"best_nll.pt"); torch.save({"writer":writer.state_dict(),"update":0,"validation_generation_f1":f1},out/"best_generation_f1.pt")
    best_nll,best_f1=nll,f1; generations.extend(gen); evaluations.append({"update":0,"validation_nll":nll,"validation_generation_f1":f1,"parameter_distance_from_stage_a":parameter_distance(writer,stage_a_state)})
    optimizer.zero_grad(set_to_none=True)
    for update in range(1,maximum+1):
        losses=[]
        for _ in range(grad_acc):
            if cursor==0: random.Random(cfg["seed"]+epoch).shuffle(samples); epoch+=1
            sample=samples[cursor]; cursor=(cursor+1)%len(samples); memory=translated_memory(cfg,writer,store,"train",sample)
            loss=answer_loss(cfg,model,tok,controller,sample,memory); (loss/grad_acc).backward(); losses.append(loss.detach().item())
        norm=torch.nn.utils.clip_grad_norm_(writer.parameters(),cfg["gradient_clip"]).item(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        history.append({"update":update,"answer_nll":sum(losses)/len(losses),"gradient_norm":norm})
        event={"update":update}
        if update%nll_interval==0 or update==maximum:
            value=functional_nll(cfg,writer,model,tok,controller,rows,store); event["validation_nll"]=value; event["parameter_distance_from_stage_a"]=parameter_distance(writer,stage_a_state)
            rep,per_layer=rep_validation(cfg,writer,store,rows["validation"][:min(cfg["diagnostic_samples"],len(rows["validation"]))]); event["representation_loss_observation"]=rep; event["per_layer_representation"]=per_layer
            observed=attention_observations(cfg,writer,store,rows["validation"][:min(cfg["diagnostic_samples"],len(rows["validation"]))],model,controller)
            for row,extra in zip(event["per_layer_representation"],observed): row.update({"route_kl":extra["route_kl"],"attention_output_cosine":extra["attention_output_cosine"]})
            if value<best_nll: best_nll=value; torch.save({"writer":writer.state_dict(),"update":update,"validation_nll":value},out/"best_nll.pt"); event["selected_nll"]=True
        if update%gen_interval==0 or update==maximum:
            generated,value=functional_generation(cfg,writer,model,tok,controller,rows,store,update); generations.extend(generated); event["validation_generation_f1"]=value
            if value>best_f1: best_f1=value; best_gen_update=update; torch.save({"writer":writer.state_dict(),"update":update,"validation_generation_f1":value},out/"best_generation_f1.pt"); event["selected_generation"]=True
        if len(event)>1: evaluations.append(event); save_json(out/"evaluations.json",evaluations); save_json(out/"generation_snapshots.json",generations); save_json(out/"history.json",history); progress(f"{mode}: Stage B {variant.upper()} {update}/{maximum}")
        if update-best_gen_update>=cfg["stage_b_generation_patience_updates"]: break
    save_json(out/"summary.json",{"completed":True,"variant":variant,"only_loss":"gold answer CE","best_nll":best_nll,"best_generation_f1":best_f1,"updates_completed":history[-1]["update"]})


def load_writer(cfg,mode,path): return make_writer(cfg,mode,path).eval()


def final_evaluate(cfg,mode):
    checkpoint,lora_enabled,selection=selected_reader(cfg,mode); root=Path(cfg["work_dir"])/"artifacts"/mode/"evaluation"
    if not selection["proceed_writer"]:
        save_json(root/"completion.json",{"completed":True,"writer_skipped":True,"reason":"Qwen3.5 Anchor8 channel signal absent"}); return
    rows=rows_for(cfg,mode); store=Stores(cfg,mode,rows); model,tok,_=load_reader(cfg,mode,checkpoint); controller=Q35AnchorInjection(model,cfg["q35_target_layers"])
    stage=Path(cfg["work_dir"])/"artifacts"/mode
    f0=load_writer(cfg,mode,stage/"stage_a"/"best.pt"); f1=load_writer(cfg,mode,stage/"stage_b_f1"/"best_generation_f1.pt")
    f2n=load_writer(cfg,mode,stage/"stage_b_f2"/"best_nll.pt"); f2g=load_writer(cfg,mode,stage/"stage_b_f2"/"best_generation_f1.pt")
    conditions=[
        {"key":"q35_native_anchor8_correct","memory":"native","lora":lora_enabled,"reader":"selected"},
        {"key":"q35_native_anchor8_shuffled","memory":"native","kind":"shuffled","lora":lora_enabled,"reader":"selected"},
        {"key":"q35_native_anchor8_no_memory","lora":lora_enabled,"reader":"selected"},
        {"key":"q3_to_q35_f0","memory":"translated","writer":f0,"lora":lora_enabled,"reader":"selected"},
        {"key":"q3_to_q35_f1","memory":"translated","writer":f1,"lora":lora_enabled,"reader":"selected"},
        {"key":"q3_to_q35_f2_best_nll","memory":"translated","writer":f2n,"lora":lora_enabled,"reader":"selected"},
        {"key":"q3_to_q35_f2_best_generation","memory":"translated","writer":f2g,"lora":lora_enabled,"reader":"selected"},
        {"key":"q3_to_q35_f2_shuffled","memory":"translated","writer":f2g,"kind":"shuffled","lora":lora_enabled,"reader":"selected"},
        {"key":"q3_to_q35_f2_no_memory","lora":lora_enabled,"reader":"selected"},
    ]
    records=evaluate(cfg,mode,model,tok,controller,rows,store,conditions); p0_records=load_json(stage/"p0"/"per_sample.json"); records=p0_records+records; summary=summarize(records)
    correct=summary["q3_to_q35_f2_best_generation"]
    summary["f2_dependence"]={"correct_minus_shuffled_em":correct["em"]-summary["q3_to_q35_f2_shuffled"]["em"],"correct_minus_no_memory_em":correct["em"]-summary["q3_to_q35_f2_no_memory"]["em"],"correct_minus_shuffled_f1":correct["token_f1"]-summary["q3_to_q35_f2_shuffled"]["token_f1"],"correct_minus_no_memory_f1":correct["token_f1"]-summary["q3_to_q35_f2_no_memory"]["token_f1"]}
    if mode!="smoke":
        summary["old_direction_references"]={
            "q35_to_q3_anchor8":load_json(Path(cfg["reverse_anchor_dir"])/"artifacts"/"development"/"evaluation"/"summary.json"),
            "q35_to_q3_full36":load_json(Path(cfg["reverse_full36_dir"])/"artifacts"/"development"/"evaluation"/"summary.json"),
        }
    write_results(root,records,summary); save_json(root/"completion.json",{"completed":True,"hard_gate":None,"reader_checkpoint":checkpoint,"reader_lora_enabled":lora_enabled})
    progress(f"{mode}: final evaluation completed")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--mode",choices=("smoke","development"),required=True); parser.add_argument("action",choices=("audit","p0","reader","assets","overfit","stage_a","f1","f2","evaluate")); args=parser.parse_args(); cfg=load_json(args.config)
    actions={"audit":audit,"p0":p0,"reader":p1_reader,"assets":assets,"overfit":lambda c,m:train_stage_a(c,m,True),"stage_a":lambda c,m:train_stage_a(c,m,False),"f1":lambda c,m:train_stage_b(c,m,"f1"),"f2":lambda c,m:train_stage_b(c,m,"f2"),"evaluate":final_evaluate}; actions[args.action](cfg,args.mode)


if __name__=="__main__": main()
