from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from data import (
    Store, cuda, load_json, load_model, normalize_answer, progress,
    representation_loss, rows_for, save_json, seed_all, token_f1, tokenizer,
)
from receiver import answer_logits, ce_loss, generate, kd_loss
from writers import (
    linear_parameters, load_linear_core, make_writer, parameter_report,
    residual_parameters,
)


KINDS = ("linear_continued", "v2_h", "v2_hl")


def root(cfg, mode, name):
    return Path(cfg["work_dir"]) / "artifacts" / mode / name


def scales_for(cfg, mode):
    return torch.load(root(cfg, mode, "scales.pt"), map_location="cpu", weights_only=False)


def old_linear_checkpoint(cfg):
    return Path(cfg["work_dir"]) / "reused" / "linear_stage_a_best.pt"


def writer_for(cfg, mode, kind, checkpoint=None):
    writer = make_writer(kind, scales_for(cfg, mode), cfg).to(cuda())
    if checkpoint is None:
        load_linear_core(writer, old_linear_checkpoint(cfg))
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        writer.load_state_dict(state["writer"] if "writer" in state else state, strict=True)
    return writer


def save_writer(path, writer, extra=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"writer": {name: value.detach().cpu() for name, value in writer.state_dict().items()}}
    if extra: payload.update(extra)
    torch.save(payload, path)


def optimizer_for(cfg, writer, stage):
    linear_lr = cfg[f"{stage}_linear_lr"]
    groups = [{"params": linear_parameters(writer), "lr": linear_lr}]
    residual = residual_parameters(writer)
    if residual: groups.append({"params": residual, "lr": cfg[f"{stage}_residual_lr"]})
    return AdamW(groups, weight_decay=cfg["weight_decay"])


def sample_rep(cfg, writer, store, split, sample):
    source, target = store.source(split, sample["id"]), store.target(split, sample["id"])
    positions = target["positions"]
    sk, sv = source["pre_key"][:, positions].to(cuda()), source["value"][:, positions].to(cuda())
    tk, tv = target["pre_key"].to(cuda()), target["value"].to(cuda())
    pk, pv = writer(sk, sv)
    return (*representation_loss(pk, pv, tk, tv, cfg["stage_a_cosine_weight"]), pk, pv)


@torch.no_grad()
def rep_validation(cfg, writer, store, split, samples):
    writer.eval(); losses, all_layers = [], []
    for sample in samples:
        loss, layers, _, _ = sample_rep(cfg, writer, store, split, sample)
        losses.append(loss.item()); all_layers.append(layers)
    aggregate = []
    for layer in range(cfg["num_layers"]):
        aggregate.append({"layer": layer, **{
            key: sum(row[layer][key] for row in all_layers) / len(all_layers)
            for key in ("k_nmse", "v_nmse", "k_cosine", "v_cosine")
        }})
    writer.train(); return sum(losses) / len(losses), aggregate


def train_stage_a(cfg, mode, kind, overfit=False):
    rows = rows_for(cfg, mode); store = Store(cfg, mode, rows); writer = writer_for(cfg, mode, kind).train()
    stage = f"{kind}_{'overfit16' if overfit else 'stage_a'}"; out = root(cfg, mode, stage); out.mkdir(parents=True, exist_ok=True)
    maximum = cfg["smoke_updates"] if mode == "smoke" else (cfg["overfit_updates"] if overfit else cfg["stage_a_updates"])
    grad_acc = cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]
    samples = list(rows["train"][:min(cfg["overfit_samples"], len(rows["train"]))] if overfit else rows["train"])
    validation, validation_split = (samples, "train") if overfit else (rows["validation"], "validation")
    optimizer = optimizer_for(cfg, writer, "stage_a")
    initial, _ = rep_validation(cfg, writer, store, validation_split, validation)
    history, evaluations, best = [], [], float("inf")
    cursor = epoch = 0; ever_grad = {name: False for name, p in writer.named_parameters() if p.requires_grad}
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for update in range(1, maximum + 1):
        optimizer.zero_grad(set_to_none=True); batch = []
        for _ in range(grad_acc):
            if cursor == 0: random.Random(cfg["seed"] + epoch).shuffle(samples); epoch += 1
            sample = samples[cursor]; cursor = (cursor + 1) % len(samples)
            loss, *_ = sample_rep(cfg, writer, store, "train", sample)
            (loss / grad_acc).backward(); batch.append(loss.item())
        for name, parameter in writer.named_parameters():
            if parameter.requires_grad and parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().max().item() > 0:
                ever_grad[name] = True
        norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]).item(); optimizer.step()
        if not all(torch.isfinite(torch.tensor(value)) for value in batch): raise RuntimeError("non-finite Stage A loss")
        history.append({"update": update, "loss": sum(batch) / len(batch), "gradient_norm": norm})
        interval = 1 if mode == "smoke" else cfg["stage_a_eval_every"]
        if update % interval == 0 or update == maximum:
            value, layers = rep_validation(cfg, writer, store, validation_split, validation)
            selected = value < best; evaluations.append({"update": update, "validation_loss": value, "selected": selected, "layers": layers})
            if selected: best = value; save_writer(out / "best.pt", writer, {"update": update, "validation_loss": value})
            progress(f"{mode}: {stage} {update}/{maximum} validation={value:.6f}")
    final, layers = rep_validation(cfg, writer, store, validation_split, validation)
    missing = [name for name, received in ever_grad.items() if not received]
    summary = {
        "kind": kind, "overfit": overfit, "initialized_from": str(old_linear_checkpoint(cfg)),
        "initial_loss": initial, "final_loss": final, "loss_ratio": final / max(initial, 1e-12),
        "all_parameters_received_gradient": not missing, "missing_gradient_parameters": missing,
        "best_validation_loss": best, "training_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "parameters": parameter_report(writer),
    }
    if hasattr(writer, "diagnostics"): summary["writer_diagnostics"] = writer.diagnostics()
    save_json(out / "history.json", history); save_json(out / "evaluations.json", evaluations)
    save_json(out / "final_layer_metrics.json", layers); save_json(out / "summary.json", summary)
    if missing: raise RuntimeError(f"{stage}: parameters without finite nonzero gradient: {missing}")


def overfit_gate(cfg, mode):
    summaries = {kind: load_json(root(cfg, mode, f"{kind}_overfit16") / "summary.json") for kind in KINDS}
    linear = summaries["linear_continued"]["final_loss"]; limit = cfg["overfit_v2_vs_linear_max_ratio"]
    checks = {
        kind: summaries[kind]["all_parameters_received_gradient"] and (mode == "smoke" or summaries[kind]["final_loss"] <= linear * limit)
        for kind in ("v2_h", "v2_hl")
    }
    report = {"passed": all(checks.values()), "linear_final_loss": linear, "maximum_v2_to_linear_ratio": limit, "checks": checks, "summaries": summaries}
    save_json(root(cfg, mode, "overfit_gate.json"), report)
    if not report["passed"]: raise RuntimeError(f"Writer v2 overfit gate failed: {checks}")


def full_source(writer, store, split, sample, source_id=None):
    record = store.source(split, source_id or sample["id"])
    return writer(record["pre_key"].to(cuda()), record["value"].to(cuda()))


@torch.no_grad()
def update0_equivalence(cfg, mode):
    rows = rows_for(cfg, mode); samples = rows["validation"]; store = Store(cfg, mode, rows)
    model = load_model(cfg["model_4b"], cfg); tok = tokenizer(cfg["model_4b"])
    writers = {kind: writer_for(cfg, mode, kind).eval() for kind in KINDS}
    maximum_difference = {"v2_h": 0.0, "v2_hl": 0.0}; maximum_kl = {"v2_h": 0.0, "v2_hl": 0.0}
    top1_matches = {"v2_h": 0, "v2_hl": 0}; generation_matches = {"v2_h": 0, "v2_hl": 0}; rows_out = []
    for sample in samples:
        outputs = {kind: full_source(writer, store, "validation", sample) for kind, writer in writers.items()}
        base_k, base_v = outputs["linear_continued"]
        base_logits, _ = answer_logits(model, sample, base_k, base_v); base_prediction, _ = generate(model, tok, sample, cfg, base_k, base_v)
        row = {"sample_id": sample["id"], "linear_prediction": base_prediction}
        for kind in ("v2_h", "v2_hl"):
            key, value = outputs[kind]
            difference = max((key.float() - base_k.float()).abs().max().item(), (value.float() - base_v.float()).abs().max().item())
            logits, _ = answer_logits(model, sample, key, value)
            base_prob = base_logits.softmax(-1); kl = (base_prob * (base_logits.log_softmax(-1) - logits.log_softmax(-1))).sum(-1).mean().item()
            prediction, _ = generate(model, tok, sample, cfg, key, value)
            top1 = bool(torch.equal(base_logits.argmax(-1), logits.argmax(-1))); generation_match = prediction == base_prediction
            maximum_difference[kind] = max(maximum_difference[kind], difference); maximum_kl[kind] = max(maximum_kl[kind], kl)
            top1_matches[kind] += int(top1); generation_matches[kind] += int(generation_match)
            row[kind] = {"kv_max_abs_difference": difference, "answer_token_kl": kl, "top1_match": top1, "prediction": prediction, "generation_match": generation_match}
        rows_out.append(row)
    count = len(samples); passed = all(
        maximum_difference[kind] < cfg["update0_kv_atol"] and maximum_kl[kind] < cfg["update0_kl_atol"]
        and top1_matches[kind] == count and generation_matches[kind] == count
        for kind in ("v2_h", "v2_hl")
    )
    report = {"passed": passed, "count": count, "maximum_kv_abs_difference": maximum_difference, "maximum_answer_token_kl": maximum_kl, "top1_matches": top1_matches, "generation_matches": generation_matches, "rows": rows_out}
    save_json(root(cfg, mode, "update0_equivalence.json"), report)
    del model, writers; torch.cuda.empty_cache()
    if not passed: raise RuntimeError("update-0 equivalence failed")


@torch.no_grad()
def validation_nll(cfg, model, writer, store, samples):
    writer.eval(); values = []
    for sample in samples:
        key, value = full_source(writer, store, "validation", sample)
        loss, _, _ = ce_loss(model, sample, key, value); values.append(loss.item())
    writer.train(); return sum(values) / len(values)


@torch.no_grad()
def validation_generation(cfg, model, tok, writer, store, samples):
    writer.eval(); scores = []
    for sample in samples:
        key, value = full_source(writer, store, "validation", sample)
        prediction, _ = generate(model, tok, sample, cfg, key, value); scores.append(token_f1(prediction, sample["answer"]))
    writer.train(); return sum(scores) / len(scores)


def train_stage_b(cfg, mode, kind, branch):
    rows = rows_for(cfg, mode); store = Store(cfg, mode, rows)
    initial = root(cfg, mode, f"{kind}_stage_a") / "best.pt"
    writer = writer_for(cfg, mode, kind, initial).train(); model = load_model(cfg["model_4b"], cfg); tok = tokenizer(cfg["model_4b"])
    out = root(cfg, mode, f"{kind}_{branch}"); out.mkdir(parents=True, exist_ok=True)
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["stage_b_updates"]
    grad_acc = cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]
    optimizer = optimizer_for(cfg, writer, "stage_b"); samples = list(rows["train"]); cursor = epoch = 0
    history, evaluations, best_nll, best_f1 = [], [], float("inf"), -1.0
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for update in range(1, maximum + 1):
        optimizer.zero_grad(set_to_none=True); values = []
        for _ in range(grad_acc):
            if cursor == 0: random.Random(cfg["seed"] + epoch).shuffle(samples); epoch += 1
            sample = samples[cursor]; cursor = (cursor + 1) % len(samples)
            key, value = full_source(writer, store, "train", sample)
            ce, logits, _ = ce_loss(model, sample, key, value); loss = ce; kd_value = 0.0
            if branch == "kd":
                teacher = store.teacher("train", sample["id"])["logits"]
                kd = kd_loss(logits, teacher, cfg["kd_temperature"]); loss = ce + cfg["kd_lambda"] * kd; kd_value = kd.item()
            (loss / grad_acc).backward(); values.append((loss.item(), ce.item(), kd_value))
        norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]).item(); optimizer.step()
        history.append({"update": update, "loss": sum(x[0] for x in values) / len(values), "ce": sum(x[1] for x in values) / len(values), "kd": sum(x[2] for x in values) / len(values), "gradient_norm": norm})
        nll_interval = 1 if mode == "smoke" else cfg["nll_eval_every"]; gen_interval = 1 if mode == "smoke" else cfg["generation_eval_every"]
        row = {"update": update}
        if update % nll_interval == 0 or update == maximum:
            score = validation_nll(cfg, model, writer, store, rows["validation"]); row["validation_nll"] = score
            if score < best_nll: best_nll = score; save_writer(out / "best_nll.pt", writer, {"update": update, "validation_nll": score})
        if update % gen_interval == 0 or update == maximum:
            count = cfg["smoke_generation_eval_samples"] if mode == "smoke" else cfg["generation_eval_samples"]
            score = validation_generation(cfg, model, tok, writer, store, rows["validation"][:count]); row["validation_generation_f1"] = score
            if score > best_f1: best_f1 = score; save_writer(out / "best_generation.pt", writer, {"update": update, "validation_generation_f1": score})
        if len(row) > 1: evaluations.append(row); progress(f"{mode}: {kind}/{branch} {update}/{maximum}")
    summary = {"kind": kind, "branch": branch, "initialized_from": str(initial), "best_validation_nll": best_nll, "best_validation_generation_f1": best_f1, "training_seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "parameters": parameter_report(writer), "representation_losses_used": False, "tether_used": False}
    save_json(out / "history.json", history); save_json(out / "evaluations.json", evaluations); save_json(out / "summary.json", summary)
    del model, writer; torch.cuda.empty_cache()


def aggregate(rows):
    result = {}
    for condition in sorted({row["condition"] for row in rows}):
        values = [row for row in rows if row["condition"] == condition]
        result[condition] = {
            "count": len(values), "em": sum(x["em"] for x in values) / len(values), "f1": sum(x["f1"] for x in values) / len(values),
            "bridge_f1": sum(x["f1"] for x in values if x["type"] == "bridge") / max(sum(x["type"] == "bridge" for x in values), 1),
            "comparison_f1": sum(x["f1"] for x in values if x["type"] == "comparison") / max(sum(x["type"] == "comparison" for x in values), 1),
            "nll": sum(x["nll"] for x in values) / len(values), "teacher_kl": sum(x["teacher_kl"] for x in values) / len(values),
        }
    return result


@torch.no_grad()
def evaluate_condition(cfg, model, tok, store, samples, condition, writer, shuffled=False):
    writer.eval(); output = []
    for sample in samples:
        source_id = sample["shuffle_id"] if shuffled else sample["id"]
        key, value = full_source(writer, store, "test", sample, source_id)
        loss, logits, _ = ce_loss(model, sample, key, value); prediction, _ = generate(model, tok, sample, cfg, key, value)
        teacher = store.teacher("test", sample["id"])["logits"].to(logits.device).float()
        kl = (teacher.softmax(-1) * (teacher.log_softmax(-1) - logits.log_softmax(-1))).sum(-1).mean().item()
        output.append({"sample_id": sample["id"], "type": sample["type"], "condition": condition, "answer": sample["answer"], "prediction": prediction, "em": float(normalize_answer(prediction) == normalize_answer(sample["answer"])), "f1": token_f1(prediction, sample["answer"]), "nll": loss.item(), "teacher_kl": kl})
    return output


def f0(cfg, mode):
    rows = rows_for(cfg, mode); store = Store(cfg, mode, rows); model = load_model(cfg["model_4b"], cfg); tok = tokenizer(cfg["model_4b"]); output = []; representations = {}
    for kind in KINDS:
        checkpoint = root(cfg, mode, f"{kind}_stage_a") / "best.pt"; writer = writer_for(cfg, mode, kind, checkpoint)
        output += evaluate_condition(cfg, model, tok, store, rows["test"], f"{kind}_f0_correct", writer)
        output += evaluate_condition(cfg, model, tok, store, rows["test"], f"{kind}_f0_shuffled", writer, True)
        value, layers = rep_validation(cfg, writer, store, "validation", rows["validation"]); representations[kind] = {"validation_loss": value, "layers": layers}
        del writer; torch.cuda.empty_cache()
    out = root(cfg, mode, "f0"); out.mkdir(parents=True, exist_ok=True); save_json(out / "per_sample.json", output); save_json(out / "summary.json", aggregate(output)); save_json(out / "representation.json", representations)
    del model; torch.cuda.empty_cache()


def select_v2(cfg, mode):
    candidates = []
    for kind in KINDS:
        summary = load_json(root(cfg, mode, f"{kind}_ce") / "summary.json")
        candidates.append({"kind": kind, "validation_generation_f1": summary["best_validation_generation_f1"], "validation_nll": summary["best_validation_nll"], "parameters": summary["parameters"]["trainable_parameters"]})
    ranked = sorted(candidates, key=lambda row: (-row["validation_generation_f1"], row["validation_nll"], row["parameters"]))
    best_v2 = next(row for row in ranked if row["kind"].startswith("v2_"))
    report = {"selection_uses_test": False, "ranking_rule": ["max validation generation F1", "min validation NLL", "min parameters"], "all_candidates": ranked, "best_overall": ranked[0], "best_v2": best_v2}
    save_json(root(cfg, mode, "selection.json"), report)


def train_selected_kd(cfg, mode):
    selection = load_json(root(cfg, mode, "selection.json")); selected = selection["best_v2"]["kind"]
    train_stage_b(cfg, mode, "linear_continued", "kd"); train_stage_b(cfg, mode, selected, "kd")


def final_evaluate(cfg, mode):
    rows = rows_for(cfg, mode); store = Store(cfg, mode, rows); model = load_model(cfg["model_4b"], cfg); tok = tokenizer(cfg["model_4b"]); output = []
    selection = load_json(root(cfg, mode, "selection.json")); selected_v2 = selection["best_v2"]["kind"]
    evaluated = []
    for kind in KINDS:
        evaluated.append((kind, "ce"))
    evaluated.extend((("linear_continued", "kd"), (selected_v2, "kd")))
    for kind, branch in evaluated:
        checkpoint = root(cfg, mode, f"{kind}_{branch}") / "best_generation.pt"
        writer = writer_for(cfg, mode, kind, checkpoint); base = f"{kind}_{branch}"
        output += evaluate_condition(cfg, model, tok, store, rows["test"], f"{base}_correct", writer)
        output += evaluate_condition(cfg, model, tok, store, rows["test"], f"{base}_shuffled", writer, True)
        del writer; torch.cuda.empty_cache()
    summary = aggregate(output)
    reference = load_json(Path(cfg["audit_4b_dir"]) / "artifacts" / "development" / "summary.json")["conditions"]
    for name in ("question_only", "full_context_text", "official_native_cache", "shuffled_native_cache"): summary[name] = reference[name]
    floor, ceiling = reference["question_only"]["f1"], reference["full_context_text"]["f1"]
    for value in summary.values():
        if "f1" in value: value["recovery"] = (value["f1"] - floor) / max(ceiling - floor, 1e-12)
    deltas = {}
    for kind, branch in evaluated:
        correct, shuffled = summary[f"{kind}_{branch}_correct"], summary[f"{kind}_{branch}_shuffled"]
        deltas[f"{kind}_{branch}"] = {"correct_minus_shuffled_f1": correct["f1"] - shuffled["f1"], "correct_minus_no_memory_f1": correct["f1"] - floor}
    conclusion = {"validation_selected_v2": selected_v2, "delta_v2_ce_over_linear_continued_ce": summary[f"{selected_v2}_ce_correct"]["f1"] - summary["linear_continued_ce_correct"]["f1"], "deltas": deltas, "same_checkpoint_used_for_correct_and_shuffled": True, "test_used_for_selection": False}
    out = root(cfg, mode, "evaluation"); out.mkdir(parents=True, exist_ok=True); save_json(out / "per_sample.json", output); save_json(out / "summary.json", summary); save_json(out / "conclusion.json", conclusion); save_json(out / "completion.json", {"completed": True, "reader": "identity", "training_performed": True})
    del model; torch.cuda.empty_cache(); progress(f"{mode}: final evaluation completed")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--mode", choices=("smoke", "development"), required=True); parser.add_argument("action"); args = parser.parse_args()
    cfg = load_json(args.config); seed_all(cfg["seed"])
    actions = {
        "update0": update0_equivalence,
        "overfit_linear": lambda c, m: train_stage_a(c, m, "linear_continued", True),
        "overfit_v2_h": lambda c, m: train_stage_a(c, m, "v2_h", True),
        "overfit_v2_hl": lambda c, m: train_stage_a(c, m, "v2_hl", True),
        "overfit_gate": overfit_gate,
        "stagea_linear": lambda c, m: train_stage_a(c, m, "linear_continued", False),
        "stagea_v2_h": lambda c, m: train_stage_a(c, m, "v2_h", False),
        "stagea_v2_hl": lambda c, m: train_stage_a(c, m, "v2_hl", False),
        "f0": f0,
        "ce_linear": lambda c, m: train_stage_b(c, m, "linear_continued", "ce"),
        "ce_v2_h": lambda c, m: train_stage_b(c, m, "v2_h", "ce"),
        "ce_v2_hl": lambda c, m: train_stage_b(c, m, "v2_hl", "ce"),
        "select": select_v2, "kd": train_selected_kd, "evaluate": final_evaluate,
    }
    actions[args.action](cfg, args.mode)


if __name__ == "__main__": main()
