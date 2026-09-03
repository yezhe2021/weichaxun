from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from data import Store, cuda, load_json, load_model, normalize_answer, progress, representation_loss, rows_for, save_json, seed_all, token_f1, tokenizer
from receiver import answer_logits, ce_loss, generate, heterogeneous_cache, heterogeneous_forward
from writers import copy_repeat_into_learnable, load_writer, make_writer, optimizer_groups, parameter_report


def root(cfg, mode, name):
    return Path(cfg["work_dir"]) / "artifacts" / mode / name


def scales_for(cfg, mode):
    return torch.load(root(cfg, mode, "scales.pt"), map_location="cpu", weights_only=False)


def save_writer(path, writer, extra=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"writer": {name: value.detach().cpu() for name, value in writer.state_dict().items()}}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def new_writer(cfg, mode, kind, checkpoint=None, repeat_checkpoint=None):
    writer = make_writer(kind, scales_for(cfg, mode), cfg).to(cuda())
    if checkpoint is not None:
        load_writer(writer, checkpoint)
    elif repeat_checkpoint is not None:
        copy_repeat_into_learnable(writer, repeat_checkpoint)
    return writer


def sample_rep(cfg, writer, store, split, sample):
    source, target = store.source(split, sample["id"]), store.target(split, sample["id"])
    positions = target["positions"]
    sk = source["pre_key"][:, positions].to(cuda())
    sv = source["value"][:, positions].to(cuda())
    pk, pv = writer(sk, sv)
    layers = pk.shape[0]
    tk, tv = target["pre_key"][:layers].to(cuda()), target["value"][:layers].to(cuda())
    loss, metrics = representation_loss(pk, pv, tk, tv, cfg["stage_a_cosine_weight"])
    return loss, metrics, pk, pv


@torch.no_grad()
def rep_validation(cfg, writer, store, split, samples):
    writer.eval(); losses, rows = [], []
    for sample in samples:
        loss, metrics, _, _ = sample_rep(cfg, writer, store, split, sample)
        losses.append(loss.item()); rows.append(metrics)
    layers = len(rows[0])
    aggregate = [{"layer": layer, **{
        key: sum(item[layer][key] for item in rows) / len(rows)
        for key in ("k_nmse", "v_nmse", "k_cosine", "v_cosine")
    }} for layer in range(layers)]
    writer.train()
    return sum(losses) / len(losses), aggregate


def stage_a_spec(cfg, mode, name):
    if mode == "smoke":
        return cfg["smoke_updates"], cfg["smoke_gradient_accumulation"]
    if name.endswith("overfit16"):
        return cfg["overfit_updates"], cfg["gradient_accumulation"]
    if name.startswith("e2_"):
        return cfg["e2_stage_a_updates"], cfg["gradient_accumulation"]
    return cfg["stage_a_updates"], cfg["gradient_accumulation"]


def train_stage_a(cfg, mode, kind, name, overfit=False, initial=None):
    rows = rows_for(cfg, mode); store = Store(cfg, mode, rows)
    repeat_checkpoint = initial if kind == "learnable_matrix" else None
    checkpoint = initial if kind != "learnable_matrix" else None
    writer = new_writer(cfg, mode, kind, checkpoint=checkpoint, repeat_checkpoint=repeat_checkpoint).train()
    out = root(cfg, mode, name); out.mkdir(parents=True, exist_ok=True)
    maximum, grad_acc = stage_a_spec(cfg, mode, name)
    samples = list(rows["train"][:min(cfg["overfit_samples"], len(rows["train"]))] if overfit else rows["train"])
    validation, validation_split = (samples, "train") if overfit else (rows["validation"], "validation")
    linear_lr = cfg["e2_stage_a_linear_lr"] if name.startswith("e2_") else cfg["stage_a_linear_lr"]
    groups = optimizer_groups(writer, linear_lr, cfg["e2_stage_a_depth_lr"])
    optimizer = AdamW(groups, weight_decay=cfg["weight_decay"])
    initial_loss, _ = rep_validation(cfg, writer, store, validation_split, validation)
    history, evaluations, best = [], [], float("inf")
    cursor = epoch = 0
    ever_grad = {n: False for n, p in writer.named_parameters() if p.requires_grad}
    initial_identity_delta = None
    if initial is None:
        eye = torch.eye(cfg["feature_dim"], device=cuda())[None]
        initial_identity_delta = max((writer.weight_k - eye).abs().max().item(), (writer.weight_v - eye).abs().max().item())
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for update in range(1, maximum + 1):
        optimizer.zero_grad(set_to_none=True); batch = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + epoch).shuffle(samples); epoch += 1
            sample = samples[cursor]; cursor = (cursor + 1) % len(samples)
            loss, *_ = sample_rep(cfg, writer, store, "train", sample)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: NaN/Inf loss")
            (loss / grad_acc).backward(); batch.append(loss.item())
        for parameter_name, parameter in writer.named_parameters():
            if parameter.requires_grad and parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().max().item() > 0:
                ever_grad[parameter_name] = True
        norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"])
        if not torch.isfinite(norm):
            raise RuntimeError(f"{name}: NaN/Inf gradient norm")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in writer.parameters()):
            raise RuntimeError(f"{name}: NaN/Inf parameter")
        history.append({"update": update, "loss": sum(batch) / len(batch), "gradient_norm": norm.item()})
        interval = 1 if mode == "smoke" else cfg["stage_a_eval_every"]
        if update % interval == 0 or update == maximum:
            value, layer_metrics = rep_validation(cfg, writer, store, validation_split, validation)
            selected = value < best
            evaluations.append({"update": update, "validation_loss": value, "selected": selected})
            if selected:
                best = value; save_writer(out / "best.pt", writer, {"update": update, "validation_loss": value})
            progress(f"{mode}: {name} {update}/{maximum} validation={value:.6f}")
    final_loss, layer_metrics = rep_validation(cfg, writer, store, validation_split, validation)
    missing = [name for name, value in ever_grad.items() if not value]
    summary = {
        "kind": kind, "overfit_code_validation_only": overfit, "initial_checkpoint": str(initial) if initial else None,
        "initial_loss": initial_loss, "final_loss": final_loss, "loss_decreased": final_loss < initial_loss,
        "all_trainable_parameters_received_nonzero_finite_gradient": not missing, "missing_gradient_parameters": missing,
        "no_nan_or_inf": True, "best_validation_loss": best, "training_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "parameters": parameter_report(writer),
    }
    if initial_identity_delta is not None:
        eye = torch.eye(cfg["feature_dim"], device=cuda())[None]
        final_delta = max((writer.weight_k - eye).abs().max().item(), (writer.weight_v - eye).abs().max().item())
        summary["parameters_changed"] = final_delta > initial_identity_delta
    save_json(out / "history.json", history); save_json(out / "evaluations.json", evaluations)
    save_json(out / "final_layer_metrics.json", layer_metrics)
    # Checkpoint save/load and deterministic restoration are mandatory for overfit code validation.
    if overfit:
        saved = new_writer(cfg, mode, kind, checkpoint=out / "best.pt").eval()
        reloaded = new_writer(cfg, mode, kind, checkpoint=out / "best.pt").eval()
        sample = validation[0]
        _, _, before_k, before_v = sample_rep(cfg, saved, store, validation_split, sample)
        _, _, after_k, after_v = sample_rep(cfg, reloaded, store, validation_split, sample)
        difference = max((before_k.float() - after_k.float()).abs().max().item(), (before_v.float() - after_v.float()).abs().max().item())
        summary["checkpoint_reload_max_abs_difference"] = difference
        summary["checkpoint_save_load_passed"] = difference == 0.0
        del saved, reloaded
        failures = []
        if not summary["loss_decreased"]: failures.append("loss did not decrease")
        if missing: failures.append("missing gradients")
        if not summary.get("parameters_changed", False): failures.append("parameters unchanged")
        if difference != 0.0: failures.append("checkpoint restore mismatch")
        if failures: raise RuntimeError(f"{name} code validation failed: {failures}")
    save_json(out / "summary.json", summary)
    del writer; torch.cuda.empty_cache()


@torch.no_grad()
def receiver_selftest(cfg, mode):
    rows = rows_for(cfg, mode); store = Store(cfg, mode, rows); sample = rows["train"][0]
    writer = new_writer(cfg, mode, "skip").eval(); model = load_model(cfg["model_4b"], cfg)
    key, value = full_source(writer, store, "train", sample)
    prefix = key.shape[1]
    cache = heterogeneous_cache(model, key, value)
    initial_lengths = [cache.get_seq_length(i) for i in range(cfg["num_layers"])]
    token = torch.tensor([[sample["question_suffix_ids"][0]]], dtype=torch.long, device=cuda())
    position = torch.tensor([[prefix]], dtype=torch.long, device=cuda())
    logits, cache = heterogeneous_forward(model, token, position, cache)
    updated_lengths = [cache.get_seq_length(i) for i in range(cfg["num_layers"])]
    expected_initial = [prefix] * cfg["source_layers"] + [0] * (cfg["num_layers"] - cfg["source_layers"])
    expected_updated = [prefix + 1] * cfg["source_layers"] + [1] * (cfg["num_layers"] - cfg["source_layers"])
    report = {
        "passed": initial_lengths == expected_initial and updated_lengths == expected_updated and torch.isfinite(logits).all().item(),
        "initial_layer_cache_lengths": initial_lengths, "after_one_question_token_lengths": updated_lengths,
        "expected_initial": expected_initial, "expected_after_one_question_token": expected_updated,
        "question_rope_position": prefix, "zero_kv_substitute_used": False, "logits_finite": torch.isfinite(logits).all().item(),
    }
    save_json(root(cfg, mode, "receiver_heterogeneous_cache_selftest.json"), report)
    del writer, model; torch.cuda.empty_cache()
    if not report["passed"]: raise RuntimeError(f"heterogeneous receiver self-test failed: {report}")


def full_source(writer, store, split, sample, source_id=None):
    source = store.source(split, source_id or sample["id"])
    return writer(source["pre_key"].to(cuda()), source["value"].to(cuda()))


@torch.no_grad()
def validation_nll(cfg, model, writer, store, samples, skip):
    writer.eval(); values = []
    for sample in samples:
        key, value = full_source(writer, store, "validation", sample)
        loss, _, _ = ce_loss(model, sample, key, value, skip=skip); values.append(loss.item())
    writer.train(); return sum(values) / len(values)


@torch.no_grad()
def validation_generation(cfg, model, tok, writer, store, samples, skip):
    writer.eval(); values = []
    for sample in samples:
        key, value = full_source(writer, store, "validation", sample)
        prediction, _ = generate(model, tok, sample, cfg, key, value, skip=skip)
        values.append(token_f1(prediction, sample["answer"]))
    writer.train(); return sum(values) / len(values)


def train_ce(cfg, mode, kind, name, initial):
    rows = rows_for(cfg, mode); store = Store(cfg, mode, rows)
    repeat_checkpoint = initial if kind == "learnable_matrix" else None
    writer = new_writer(cfg, mode, kind, checkpoint=None if kind == "learnable_matrix" else initial, repeat_checkpoint=repeat_checkpoint).train()
    # For E2 learnable CE, the initial checkpoint already contains depth_delta and must be loaded strictly.
    if kind == "learnable_matrix" and "e2_learnable" in str(initial):
        load_writer(writer, initial)
    model = load_model(cfg["model_4b"], cfg); tok = tokenizer(cfg["model_4b"])
    out = root(cfg, mode, name); out.mkdir(parents=True, exist_ok=True)
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["stage_b_updates"]
    grad_acc = cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]
    linear_lr = cfg["stage_b_linear_lr"]
    depth_lr = cfg["e2_stage_b_depth_lr"]
    optimizer = AdamW(optimizer_groups(writer, linear_lr, depth_lr), weight_decay=cfg["weight_decay"])
    samples = list(rows["train"]); cursor = epoch = 0
    history, evaluations, best_nll, best_f1 = [], [], float("inf"), -1.0
    skip = kind == "skip"
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for update in range(1, maximum + 1):
        optimizer.zero_grad(set_to_none=True); values = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + epoch).shuffle(samples); epoch += 1
            sample = samples[cursor]; cursor = (cursor + 1) % len(samples)
            key, value = full_source(writer, store, "train", sample)
            loss, _, _ = ce_loss(model, sample, key, value, skip=skip)
            if not torch.isfinite(loss): raise RuntimeError(f"{name}: NaN/Inf CE")
            (loss / grad_acc).backward(); values.append(loss.item())
        norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"])
        if not torch.isfinite(norm): raise RuntimeError(f"{name}: NaN/Inf gradient")
        optimizer.step(); history.append({"update": update, "ce": sum(values) / len(values), "gradient_norm": norm.item()})
        row = {"update": update}
        nll_interval = 1 if mode == "smoke" else cfg["nll_eval_every"]
        gen_interval = 1 if mode == "smoke" else cfg["generation_eval_every"]
        if update % nll_interval == 0 or update == maximum:
            score = validation_nll(cfg, model, writer, store, rows["validation"], skip); row["validation_nll"] = score
            if score < best_nll:
                best_nll = score; save_writer(out / "best_nll.pt", writer, {"update": update, "validation_nll": score})
        if update % gen_interval == 0 or update == maximum:
            count = cfg["smoke_generation_eval_samples"] if mode == "smoke" else cfg["generation_eval_samples"]
            score = validation_generation(cfg, model, tok, writer, store, rows["validation"][:count], skip); row["validation_generation_f1"] = score
            if score > best_f1:
                best_f1 = score; save_writer(out / "selected.pt", writer, {"selection_rule": "highest validation generation F1", "update": update, "validation_generation_f1": score})
        if len(row) > 1:
            evaluations.append(row); progress(f"{mode}: {name} {update}/{maximum}")
    save_json(out / "history.json", history); save_json(out / "evaluations.json", evaluations)
    save_json(out / "summary.json", {
        "kind": kind, "loss": "answer CE only", "initial_checkpoint": str(initial),
        "best_validation_nll": best_nll, "best_validation_generation_f1": best_f1,
        "selected_checkpoint": "selected.pt", "selection_uses_test": False,
        "representation_loss_used": False, "kd_used": False, "tether_used": False,
        "training_seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    })
    del model, writer; torch.cuda.empty_cache()


def aggregate(rows):
    output = {}
    for condition in sorted({x["condition"] for x in rows}):
        values = [x for x in rows if x["condition"] == condition]
        output[condition] = {
            "count": len(values), "em": sum(x["em"] for x in values) / len(values),
            "f1": sum(x["f1"] for x in values) / len(values), "nll": sum(x["nll"] for x in values) / len(values),
        }
    return output


@torch.no_grad()
def evaluate_condition(cfg, model, tok, store, samples, condition, writer, shuffled=False):
    writer.eval(); output = []; skip = writer.protocol == "skip"
    for sample in samples:
        key, value = full_source(writer, store, "test", sample, sample["shuffle_id"] if shuffled else sample["id"])
        loss, _, _ = ce_loss(model, sample, key, value, skip=skip)
        prediction, _ = generate(model, tok, sample, cfg, key, value, skip=skip)
        output.append({
            "sample_id": sample["id"], "type": sample["type"], "condition": condition,
            "answer": sample["answer"], "prediction": prediction,
            "em": float(normalize_answer(prediction) == normalize_answer(sample["answer"])),
            "f1": token_f1(prediction, sample["answer"]), "nll": loss.item(),
        })
    return output


def checkpoint_for(cfg, mode, condition):
    mapping = {
        "partial28_skip_f0": ("skip", "e1_skip_stage_a/best.pt"),
        "partial28_skip_ce": ("skip", "e1_skip_ce/selected.pt"),
        "partial28_repeat_f0": ("repeat", "e1_repeat_stage_a/best.pt"),
        "partial28_repeat_ce": ("repeat", "e1_repeat_ce/selected.pt"),
        "repeat_continued_f0": ("fixed_continued", "e2_fixed_stage_a/best.pt"),
        "repeat_continued_ce": ("fixed_continued", "e2_fixed_ce/selected.pt"),
        "learnable_matrix_f0": ("learnable_matrix", "e2_learnable_stage_a/best.pt"),
        "learnable_matrix_ce": ("learnable_matrix", "e2_learnable_ce/selected.pt"),
    }
    kind, relative = mapping[condition]
    return kind, root(cfg, mode, relative.split("/")[0]) / relative.split("/")[1]


@torch.no_grad()
def evaluate_named(cfg, mode, names, output_name):
    rows = rows_for(cfg, mode); store = Store(cfg, mode, rows)
    model = load_model(cfg["model_4b"], cfg); tok = tokenizer(cfg["model_4b"]); output = []
    for condition in names:
        kind, checkpoint = checkpoint_for(cfg, mode, condition)
        writer = new_writer(cfg, mode, kind, checkpoint=checkpoint)
        output += evaluate_condition(cfg, model, tok, store, rows["test"], condition + "_correct", writer)
        output += evaluate_condition(cfg, model, tok, store, rows["test"], condition + "_shuffled", writer, True)
        del writer; torch.cuda.empty_cache()
    out = root(cfg, mode, output_name); out.mkdir(parents=True, exist_ok=True)
    save_json(out / "per_sample.json", output); save_json(out / "summary.json", aggregate(output))
    del model; torch.cuda.empty_cache()


@torch.no_grad()
def update0(cfg, mode):
    rows = rows_for(cfg, mode); samples = rows["validation"]; store = Store(cfg, mode, rows)
    checkpoint = root(cfg, mode, "e1_repeat_stage_a") / "best.pt"
    repeat = new_writer(cfg, mode, "repeat", checkpoint=checkpoint).eval()
    learned = new_writer(cfg, mode, "learnable_matrix", repeat_checkpoint=checkpoint).eval()
    model = load_model(cfg["model_4b"], cfg); tok = tokenizer(cfg["model_4b"])
    maximum_kv = maximum_logits = 0.0; generation_matches = 0; f1_matches = 0; per_sample = []
    for sample in samples:
        source = store.source("validation", sample["id"])
        sk, sv = source["pre_key"].to(cuda()), source["value"].to(cuda())
        rk, rv = repeat(sk, sv); lk, lv = learned(sk, sv)
        kv_difference = max((rk.float() - lk.float()).abs().max().item(), (rv.float() - lv.float()).abs().max().item())
        rlogits, _ = answer_logits(model, sample, rk, rv); llogits, _ = answer_logits(model, sample, lk, lv)
        logits_difference = (rlogits - llogits).abs().max().item()
        rp, _ = generate(model, tok, sample, cfg, rk, rv); lp, _ = generate(model, tok, sample, cfg, lk, lv)
        rf1, lf1 = token_f1(rp, sample["answer"]), token_f1(lp, sample["answer"])
        maximum_kv = max(maximum_kv, kv_difference); maximum_logits = max(maximum_logits, logits_difference)
        generation_matches += int(rp == lp); f1_matches += int(rf1 == lf1)
        per_sample.append({"sample_id": sample["id"], "kv_max_abs": kv_difference, "logits_max_abs": logits_difference, "repeat_prediction": rp, "learnable_prediction": lp, "generation_match": rp == lp, "f1_match": rf1 == lf1})
    count = len(samples)
    passed = maximum_kv <= cfg["update0_kv_atol"] and maximum_logits <= cfg["update0_logits_atol"] and generation_matches == count and f1_matches == count
    report = {"passed": passed, "count": count, "maximum_kv_abs_difference": maximum_kv, "maximum_logits_abs_difference": maximum_logits, "generation_matches": generation_matches, "f1_matches": f1_matches, "rows": per_sample}
    save_json(root(cfg, mode, "e2_update0.json"), report)
    del model, repeat, learned; torch.cuda.empty_cache()
    if not passed: raise RuntimeError(f"E2 update-0 equivalence failed: {report}")


def depth_outputs(cfg, mode):
    checkpoint = root(cfg, mode, "e2_learnable_ce") / "selected.pt"
    writer = new_writer(cfg, mode, "learnable_matrix", checkpoint=checkpoint).eval()
    matrix = writer.depth_matrix().detach().cpu().float()
    normalized = matrix.abs() / matrix.abs().sum(-1, keepdim=True).clamp_min(1e-12)
    rows = []
    for layer in range(matrix.shape[0]):
        values, indices = normalized[layer].topk(3)
        rows.append({"target_layer": layer, "top3_source_layers": [{"source_layer": int(i), "absolute_normalized_weight": float(v), "signed_weight": float(matrix[layer, i])} for v, i in zip(values, indices)]})
    out = root(cfg, mode, "evaluation"); out.mkdir(parents=True, exist_ok=True)
    save_json(out / "depth_matrix.json", matrix.tolist()); save_json(out / "depth_matrix_top3.json", rows)
    # Dependency-free SVG heatmap.
    cell = 14; width, height = matrix.shape[1] * cell, matrix.shape[0] * cell
    limit = matrix.abs().max().item() or 1.0
    rects = []
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x].item() / limit
            if value >= 0: color = f"rgb({int(255*(1-value))},{int(255*(1-value))},255)"
            else: color = f"rgb(255,{int(255*(1+value))},{int(255*(1+value))})"
            rects.append(f'<rect x="{x*cell}" y="{y*cell}" width="{cell}" height="{cell}" fill="{color}"/>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">' + "".join(rects) + "</svg>"
    (out / "depth_matrix_heatmap.svg").write_text(svg, encoding="utf-8")
    del writer


def final_evaluate(cfg, mode):
    names = ["partial28_skip_f0", "partial28_skip_ce", "partial28_repeat_f0", "partial28_repeat_ce", "repeat_continued_f0", "repeat_continued_ce", "learnable_matrix_f0", "learnable_matrix_ce"]
    evaluate_named(cfg, mode, names, "evaluation")
    out = root(cfg, mode, "evaluation"); summary = load_json(out / "summary.json")
    reference4 = load_json(Path(cfg["audit_4b_dir"]) / "artifacts" / "development" / "summary.json")["conditions"]
    reference17 = load_json(Path(cfg["cache_audit_1_7b_dir"]) / "artifacts" / "development" / "summary.json")["conditions"]
    summary["qwen3_4b_question_only"] = reference4["question_only"]
    summary["qwen3_4b_full_context"] = reference4["full_context_text"]
    summary["qwen3_1_7b_full_context"] = reference17["full_context_text"]
    question_f1 = reference4["question_only"]["f1"]
    comparisons = {}
    for name in names:
        correct, shuffled = summary[name + "_correct"]["f1"], summary[name + "_shuffled"]["f1"]
        comparisons[name] = {"delta_memory": correct - question_f1, "delta_shuffle": correct - shuffled, "beats_question_only": correct > question_f1, "beats_shuffled": correct > shuffled}
    comparisons["delta_matrix_ce"] = summary["learnable_matrix_ce_correct"]["f1"] - summary["repeat_continued_ce_correct"]["f1"]
    summary["comparisons"] = comparisons; save_json(out / "summary.json", summary)
    depth_outputs(cfg, mode); save_json(out / "completion.json", {"completed": True, "kd_used": False, "writer_v2_used": False, "fixed_depth_interpolation_used": False})
    progress(f"{mode}: final evaluation completed")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--mode", choices=("smoke", "development"), required=True); parser.add_argument("action"); args = parser.parse_args()
    cfg = load_json(args.config); seed_all(cfg["seed"])
    repeat_stage_a = root(cfg, args.mode, "e1_repeat_stage_a") / "best.pt"
    actions = {
        "receiver_selftest": lambda: receiver_selftest(cfg, args.mode),
        "overfit_skip": lambda: train_stage_a(cfg, args.mode, "skip", "e1_skip_overfit16", True),
        "overfit_repeat": lambda: train_stage_a(cfg, args.mode, "repeat", "e1_repeat_overfit16", True),
        "stagea_skip": lambda: train_stage_a(cfg, args.mode, "skip", "e1_skip_stage_a"),
        "stagea_repeat": lambda: train_stage_a(cfg, args.mode, "repeat", "e1_repeat_stage_a"),
        "f0_e1": lambda: evaluate_named(cfg, args.mode, ["partial28_skip_f0", "partial28_repeat_f0"], "e1_f0"),
        "ce_skip": lambda: train_ce(cfg, args.mode, "skip", "e1_skip_ce", root(cfg, args.mode, "e1_skip_stage_a") / "best.pt"),
        "ce_repeat": lambda: train_ce(cfg, args.mode, "repeat", "e1_repeat_ce", repeat_stage_a),
        "update0": lambda: update0(cfg, args.mode),
        "stagea_fixed": lambda: train_stage_a(cfg, args.mode, "fixed_continued", "e2_fixed_stage_a", initial=repeat_stage_a),
        "stagea_learnable": lambda: train_stage_a(cfg, args.mode, "learnable_matrix", "e2_learnable_stage_a", initial=repeat_stage_a),
        "f0_e2": lambda: evaluate_named(cfg, args.mode, ["repeat_continued_f0", "learnable_matrix_f0"], "e2_f0"),
        "ce_fixed": lambda: train_ce(cfg, args.mode, "fixed_continued", "e2_fixed_ce", root(cfg, args.mode, "e2_fixed_stage_a") / "best.pt"),
        "ce_learnable": lambda: train_ce(cfg, args.mode, "learnable_matrix", "e2_learnable_ce", root(cfg, args.mode, "e2_learnable_stage_a") / "best.pt"),
        "evaluate": lambda: final_evaluate(cfg, args.mode),
    }
    actions[args.action]()


if __name__ == "__main__":
    main()
