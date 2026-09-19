import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import ROOT, log, read_json, run_root, save_json, save_tensor, seed_all
from data import load_pair, load_source, manifest_rows
from protocol import final_logits
from translator import NativeKVTranslator, ResidualKVAdapter, component_loss, representation_loss


def new_translator(cfg):
    return NativeKVTranslator(
        "full34_headmix256", hidden_dim=cfg["depth_hidden_dim"],
        depth_output_dim=cfg["depth_output_dim"], head_mapping=cfg["head_mapping"])


def stage_a_checkpoint(cfg):
    selection = read_json(run_root(cfg) / "stage_a" / "selection.json")
    return Path(selection["best_accuracy"]["path"]), selection["best_accuracy"]


def tensor_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_base(cfg, frozen=True):
    path, selection = stage_a_checkpoint(cfg)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("signature") != cfg["signature"] or payload.get("architecture") != cfg["architecture"]:
        raise RuntimeError("Stage-A checkpoint protocol mismatch")
    module = new_translator(cfg).cuda()
    module.load_state_dict(payload["state"], strict=True)
    module.requires_grad_(not frozen)
    return (module.eval() if frozen else module.train()), path, selection


def choice_kl(logits, teacher_choice, ids, temperature):
    index = torch.tensor(ids, device=logits.device)
    student = F.log_softmax(logits[index].float() / temperature, -1)
    target = F.log_softmax(teacher_choice.to(logits.device).float() / temperature, -1)
    return F.kl_div(student, target.detach(), reduction="sum", log_target=True) * temperature ** 2


def receiver_logits(model, row, item, key, value):
    key = torch.cat((item["question_k"].cuda(), key), 1)
    value = torch.cat((item["question_v"].cuda(), value), 1)
    return final_logits(model, row["encoded"]["qwen"]["receiver_answer"], key, value,
                        positions=torch.arange(key.shape[1], device="cuda"), suffix_start=key.shape[1])


def teacher_path(cfg, split, row):
    return run_root(cfg) / "oracle_teacher" / split / f"{row['id']}.pt"


def load_teacher(cfg, split, row):
    payload = torch.load(teacher_path(cfg, split, row), map_location="cpu", weights_only=True)
    if payload.get("signature") != cfg["signature"]:
        raise RuntimeError("Teacher signature mismatch")
    return payload


@torch.no_grad()
def prepare_oracle_teachers(cfg, model):
    for split in ("train", "validation", "test"):
        selected = manifest_rows(cfg, split)
        for number, row in enumerate(selected, 1):
            pair = load_pair(cfg, split, row)
            logits = receiver_logits(model, row, pair, pair["target_k"][:, 1:].cuda(), pair["target_v"][:, 1:].cuda())
            ids = row["encoded"]["qwen"]["choice_ids"]
            choice = logits[ids].cpu()
            save_tensor(teacher_path(cfg, split, row), {
                "signature": cfg["signature"], "choice_logits": choice, "prediction": int(choice.argmax())
            })
            if number % 64 == 0 or number == len(selected):
                log(f"Qwen Oracle teachers {split}: {number}/{len(selected)}")


def base_output(base, source_item, gradients=False):
    context = torch.enable_grad() if gradients else torch.no_grad()
    with context, torch.amp.autocast("cuda", dtype=torch.float16):
        return base(source_item["source_k"][:, 1:].cuda()[None],
                    source_item["source_v"][:, 1:].cuda()[None])


def branch_output(kind, base, adapter, source_item, gradients=False):
    key, value = base_output(base, source_item, gradients=gradients and kind == "shared")
    if kind == "shared" or adapter is None:
        return key[0], value[0], None, None
    with torch.amp.autocast("cuda", dtype=torch.float16):
        final_k, final_v, delta_k, delta_v = adapter(key, value)
    return final_k[0], final_v[0], delta_k[0], delta_v[0]


@torch.no_grad()
def functional_metrics(cfg, model, kind, base, adapter, split):
    base.eval()
    if adapter is not None:
        adapter.eval()
    selected = manifest_rows(cfg, split)
    totals = {"accuracy": 0, "oracle_accuracy": 0, "oracle_agreement": 0, "choice_kl": 0.0}
    for row in selected:
        source_item, item, target = load_source(cfg, split, row), load_pair(cfg, split, row), load_teacher(cfg, split, row)
        key, value, _, _ = branch_output(kind, base, adapter, source_item)
        logits = receiver_logits(model, row, item, key, value)
        ids, gold = row["encoded"]["qwen"]["choice_ids"], row["gold_index"]
        prediction, oracle = int(logits[ids].argmax()), int(target["prediction"])
        totals["accuracy"] += prediction == gold
        totals["oracle_accuracy"] += oracle == gold
        totals["oracle_agreement"] += prediction == oracle
        totals["choice_kl"] += choice_kl(logits, target["choice_logits"], ids, cfg["temperature"]).item()
    return {name: value / len(selected) for name, value in totals.items()}


def source_target_batch(cfg, split, selected_rows):
    sources = [load_source(cfg, split, row) for row in selected_rows]
    pairs = [load_pair(cfg, split, row) for row in selected_rows]
    return (
        torch.stack([item["source_k"][:, 1:] for item in sources]).cuda(),
        torch.stack([item["source_v"][:, 1:] for item in sources]).cuda(),
        torch.stack([item["target_k"][:, 1:] for item in pairs]).cuda(),
        torch.stack([item["target_v"][:, 1:] for item in pairs]).cuda(),
    )


@torch.no_grad()
def representation_metrics(cfg, module, split):
    module.eval()
    selected = manifest_rows(cfg, split)
    totals = {name: 0.0 for name in ("loss", "k_nmse", "v_nmse", "k_cosine", "v_cosine")}
    for begin in range(0, len(selected), cfg["batch_size"]):
        batch = selected[begin:begin + cfg["batch_size"]]
        sk, sv, tk, tv = source_target_batch(cfg, split, batch)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pk, pv = module(sk, sv)
        loss, details = representation_loss(pk, pv, tk, tv)
        totals["loss"] += loss.item() * len(batch)
        for name, value in details.items():
            totals[name] += value.item() * len(batch)
    return {name: value / len(selected) for name, value in totals.items()}


def save_stage_a(cfg, module, step, validation):
    path = run_root(cfg) / "stage_a" / "checkpoints" / f"step_{step}.pt"
    save_tensor(path, {"signature": cfg["signature"], "architecture": cfg["architecture"], "step": step,
                       "validation": validation,
                       "state": {name: value.detach().cpu() for name, value in module.state_dict().items()}})
    return path


def load_stage_a_candidate(cfg, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("signature") != cfg["signature"] or payload.get("architecture") != cfg["architecture"]:
        raise RuntimeError("Fresh Stage-A checkpoint mismatch")
    module = new_translator(cfg).cuda().eval()
    module.load_state_dict(payload["state"], strict=True)
    return module, payload


def train_stage_a(cfg):
    seed_all(cfg["seed"])
    module = new_translator(cfg).cuda().train()
    selected = manifest_rows(cfg, "train")
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["stage_a_learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    root = run_root(cfg) / "stage_a"
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / "training_steps.jsonl").open("w", encoding="utf-8")
    candidates, step, clipped, started = [], 0, 0, time.monotonic()
    try:
        for epoch in range(1, cfg["stage_a_epochs"] + 1):
            order = list(range(len(selected)))
            random.Random(cfg["seed"] + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                batch = [selected[index] for index in order[begin:begin + cfg["batch_size"]]]
                sk, sv, tk, tv = source_target_batch(cfg, "train", batch)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pk, pv = module(sk, sv)
                loss, details = representation_loss(pk, pv, tk, tv)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite fresh Stage-A loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if not torch.isfinite(norm):
                    raise RuntimeError("Invalid fresh Stage-A gradient")
                clipped += int(norm.item() > cfg["clip"])
                scaler.step(optimizer)
                scaler.update()
                step += 1
                record = {"epoch": epoch, "step": step, "loss": loss.item(),
                          **{name: value.item() for name, value in details.items()},
                          "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                if step % 16 == 0:
                    log(f"Fresh Stage A {step}/{cfg['stage_a_steps']} loss={loss.item():.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["stage_a_steps"]:
                    validation = representation_metrics(cfg, module, "validation")
                    module.train()
                    path = save_stage_a(cfg, module, step, validation)
                    candidates.append({"step": step, "path": str(path), "representation": validation})
                    save_json(root / "candidates.json", candidates)
                    log(f"Fresh Stage A validation step={step} loss={validation['loss']:.6f}")
                if step >= cfg["stage_a_steps"]:
                    break
            if step >= cfg["stage_a_steps"]:
                break
    finally:
        stream.close()
    summary = {"stage": "A freshly retrained", "objective": "KV reconstruction only",
               "train_samples": len(selected), "epochs": cfg["stage_a_epochs"], "steps": step,
               "parameters_trained": sum(p.numel() for p in module.parameters()),
               "clip_rate": clipped / max(step, 1), "seconds": time.monotonic() - started,
               "candidates": candidates}
    save_json(root / "training_summary.json", summary)
    del module, optimizer, scaler
    torch.cuda.empty_cache()
    return summary


def select_stage_a(cfg, model):
    root = run_root(cfg) / "stage_a"
    summary = read_json(root / "training_summary.json")
    evaluated = []
    for candidate in summary["candidates"]:
        module, _ = load_stage_a_candidate(cfg, candidate["path"])
        functional = functional_metrics(cfg, model, "shared", module, None, "validation")
        evaluated.append({**candidate, "functional": functional})
        log(f"Fresh Stage A functional step={candidate['step']} acc={functional['accuracy']:.4f} KL={functional['choice_kl']:.6f}")
        del module
        torch.cuda.empty_cache()
    best_accuracy = max(evaluated, key=lambda item: (item["functional"]["accuracy"],
                                                      -item["functional"]["choice_kl"], -item["step"]))
    best_kl = min(evaluated, key=lambda item: (item["functional"]["choice_kl"], item["step"]))
    selection = {"selection": "official validation128 only", "best_accuracy": best_accuracy,
                 "best_choice_kl": best_kl, "candidates": evaluated}
    save_json(root / "selection.json", selection)
    best_module, _ = load_stage_a_candidate(cfg, best_accuracy["path"])
    head, depth = best_module.correspondence()
    save_json(root / "correspondence.json", {
        "head_definition": "Frobenius norm [target_layer,target_head,source_head]",
        "depth_definition": "Frobenius norm [target_layer,source_layer] of first depth projection",
        "head_k": head["k"].tolist(), "head_v": head["v"].tolist(),
        "depth_k": depth["k"].tolist(), "depth_v": depth["v"].tolist(),
    })
    del best_module
    torch.cuda.empty_cache()
    return selection


def checkpoint_path(cfg, kind, step):
    return run_root(cfg) / f"stage_b_{kind}" / "checkpoints" / f"step_{step}.pt"


def save_branch(cfg, kind, base, adapter, step, metrics, base_path, base_hash):
    path = checkpoint_path(cfg, kind, step)
    module = base if kind == "shared" else adapter
    save_tensor(path, {"signature": cfg["signature"], "kind": kind, "architecture": cfg["architecture"],
        "step": step, "validation": metrics, "base_checkpoint": str(base_path), "base_hash": base_hash,
        "state": {name: value.detach().cpu() for name, value in module.state_dict().items()}})
    return path


def train_branch(cfg, model, kind):
    if kind not in ("shared", "residual"):
        raise ValueError(kind)
    seed_all(cfg["seed"] + (100 if kind == "shared" else 200))
    base, base_path, base_selection = load_base(cfg, frozen=kind == "residual")
    base_hash_before = tensor_hash(base)
    adapter = ResidualKVAdapter(rank=cfg["adapter_rank"]).cuda().train() if kind == "residual" else None
    parameters = list(base.parameters()) if kind == "shared" else list(adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=1000000)
    selected = manifest_rows(cfg, "train")
    root = run_root(cfg) / f"stage_b_{kind}"
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / "training_steps.jsonl").open("w", encoding="utf-8")
    candidates, step, clipped, started = [], 0, 0, time.monotonic()
    try:
        for epoch in range(1, cfg["stage_b_epochs"] + 1):
            order = list(range(len(selected)))
            random.Random(cfg["seed"] + (100 if kind == "shared" else 200) + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                indices = order[begin:begin + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index in indices:
                    row = selected[index]
                    source_item, item, target = load_source(cfg, "train", row), load_pair(cfg, "train", row), load_teacher(cfg, "train", row)
                    key, value, _, _ = branch_output(kind, base, adapter, source_item, gradients=True)
                    logits = receiver_logits(model, row, item, key, value)
                    loss = choice_kl(logits, target["choice_logits"], row["encoded"]["qwen"]["choice_ids"], cfg["temperature"])
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite {kind} Stage-B loss")
                    scaler.scale(loss / len(indices)).backward()
                    losses.append(loss.item())
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["clip"])
                if not torch.isfinite(norm):
                    raise RuntimeError(f"Invalid {kind} gradient")
                clipped += int(norm.item() > cfg["clip"])
                scaler.step(optimizer)
                scaler.update()
                step += 1
                record = {"epoch": epoch, "step": step, "mean_choice_kl": sum(losses) / len(losses),
                          "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                if step % 16 == 0:
                    log(f"{kind} Stage B {step}/{cfg['stage_b_steps']} KL={record['mean_choice_kl']:.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["stage_b_steps"]:
                    metrics = functional_metrics(cfg, model, kind, base, adapter, "validation")
                    (base if kind == "shared" else adapter).train()
                    path = save_branch(cfg, kind, base, adapter, step, metrics, base_path, base_hash_before)
                    candidate = {"step": step, "path": str(path), "functional": metrics}
                    candidates.append(candidate)
                    save_json(root / "candidates.json", candidates)
                    log(f"{kind} validation step={step} acc={metrics['accuracy']:.4f} KL={metrics['choice_kl']:.6f}")
                if step >= cfg["stage_b_steps"]:
                    break
            if step >= cfg["stage_b_steps"]:
                break
    finally:
        stream.close()
    base_hash_after = tensor_hash(base)
    if kind == "residual" and (base_hash_after != base_hash_before or any(p.grad is not None for p in base.parameters())):
        raise RuntimeError("Frozen Stage-A Base changed during residual training")
    best_accuracy = max(candidates, key=lambda x: (x["functional"]["accuracy"], -x["functional"]["choice_kl"], -x["step"]))
    best_kl = min(candidates, key=lambda x: (x["functional"]["choice_kl"], x["step"]))
    summary = {"stage": f"B {kind}", "objective": "final-position A-D choice KL only",
        "base_checkpoint": str(base_path), "base_stage_a_selection": base_selection,
        "base_frozen": kind == "residual", "base_hash_before": base_hash_before, "base_hash_after": base_hash_after,
        "train_samples": len(selected), "parameters_trained": sum(p.numel() for p in parameters),
        "adapter_rank": cfg["adapter_rank"] if adapter else None, "steps": step, "epochs": cfg["stage_b_epochs"],
        "clip_rate": clipped / max(step, 1), "seconds": time.monotonic() - started,
        "best_accuracy": best_accuracy, "best_choice_kl": best_kl, "candidates": candidates}
    save_json(root / "training_summary.json", summary)
    del base, adapter, optimizer, scaler
    torch.cuda.empty_cache()
    return summary


def load_branch(cfg, kind):
    summary = read_json(run_root(cfg) / f"stage_b_{kind}" / "training_summary.json")
    payload = torch.load(summary["best_accuracy"]["path"], map_location="cpu", weights_only=True)
    if payload.get("signature") != cfg["signature"] or payload.get("kind") != kind:
        raise RuntimeError(f"{kind} checkpoint mismatch")
    if kind == "shared":
        base = new_translator(cfg).cuda().eval()
        base.load_state_dict(payload["state"], strict=True)
        return base, None, summary
    base, _, _ = load_base(cfg, frozen=True)
    adapter = ResidualKVAdapter(rank=cfg["adapter_rank"]).cuda().eval()
    adapter.load_state_dict(payload["state"], strict=True)
    return base, adapter, summary


def quantiles(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"min": tensor.min().item(), "p10": tensor.quantile(.1).item(), "p25": tensor.quantile(.25).item(),
            "median": tensor.median().item(), "p75": tensor.quantile(.75).item(), "p90": tensor.quantile(.9).item(),
            "p95": tensor.quantile(.95).item(), "max": tensor.max().item(), "mean": tensor.mean().item()}


def correctness_pair(first, second):
    both = sum(a and b for a, b in zip(first, second))
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    neither = len(first) - both - only_first - only_second
    d = only_first + only_second
    p = 1.0 if not d else min(1.0, 2 * sum(math.comb(d, k) for k in range(min(only_first, only_second) + 1)) / 2 ** d)
    return {"both_correct": both, "only_first_correct": only_first, "only_second_correct": only_second,
            "both_wrong": neither, "second_minus_first": (only_second - only_first) / len(first), "mcnemar_p": p}


def write_results_report(cfg, result):
    lines = [
        "# Gemma3-4B -> Qwen3-4B Full34 HeadMix256 Results", "",
        f"Protocol: `{result['protocol']}`; split: `{result['train_samples']}/{result['validation_samples']}/{result['test_samples']}`; seed `{cfg['seed']}`.", "",
        "## Native baselines and controls", "",
        "| Condition | Correct | Accuracy |", "|---|---:|---:|",
    ]
    for name, values in result.get("baselines", {}).get("metrics", {}).items():
        lines.append(f"| {name} | {values['correct']} | {100 * values['accuracy']:.2f}% |")
    lines += ["", "## Translator results", "",
              "| Method | Accuracy | Oracle agreement | Choice KL | K cosine | V cosine | Oracle gap |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    gaps = result.get("translation_gap", {})
    for name in ("stage_a", "residual", "shared"):
        values = result["metrics"][name]
        lines.append(
            f"| {name} | {100 * values['accuracy']:.2f}% | {100 * values['oracle_agreement']:.2f}% | "
            f"{values['oracle_choice_kl']:.6f} | {values['k_cosine']:.6f} | {values['v_cosine']:.6f} | "
            f"{100 * gaps.get(name, float('nan')):.2f} pp |")
    lines += ["", "## Paired correctness", "", "```json",
              json.dumps({key: value for key, value in result.items() if key.startswith("paired_")},
                         ensure_ascii=False, indent=2), "```", "",
              "The machine-readable per-sample predictions, training traces, native controls, and learned head/layer correspondence matrices are stored under `runs/study/`."]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate(cfg, model):
    modules = {kind: load_branch(cfg, kind) for kind in ("residual", "shared")}
    stage_a, stage_a_path, _ = load_base(cfg, frozen=True)
    selected = manifest_rows(cfg, "test")
    names = ("stage_a", "residual", "shared")
    totals = {name: {"accuracy": 0, "agreement": 0, "kl": 0.0,
                     "k_nmse": 0.0, "v_nmse": 0.0, "k_cosine": 0.0, "v_cosine": 0.0} for name in names}
    correct = {name: [] for name in names}
    ratios = {"k": [], "v": []}
    records = []
    for number, row in enumerate(selected, 1):
        source_item, item, target = load_source(cfg, "test", row), load_pair(cfg, "test", row), load_teacher(cfg, "test", row)
        outputs = {"stage_a": branch_output("shared", stage_a, None, source_item)[:2]}
        for kind, (base, adapter, _) in modules.items():
            outputs[kind] = branch_output(kind, base, adapter, source_item)[:2]
        residual_base, residual_adapter, _ = modules["residual"]
        bk, bv = base_output(residual_base, source_item)
        _, _, dk, dv = branch_output("residual", residual_base, residual_adapter, source_item)
        ratios["k"].append(dk.float().norm().item() / bk.float().norm().clamp_min(1e-12).item())
        ratios["v"].append(dv.float().norm().item() / bv.float().norm().clamp_min(1e-12).item())
        target_k, target_v = item["target_k"][:, 1:].cuda(), item["target_v"][:, 1:].cuda()
        record = {"id": row["id"], "gold_index": row["gold_index"], "conditions": {}}
        for name, (key, value) in outputs.items():
            logits = receiver_logits(model, row, item, key, value)
            ids, gold, oracle = row["encoded"]["qwen"]["choice_ids"], row["gold_index"], int(target["prediction"])
            prediction = int(logits[ids].argmax())
            is_correct = prediction == gold
            correct[name].append(is_correct)
            totals[name]["accuracy"] += is_correct
            totals[name]["agreement"] += prediction == oracle
            totals[name]["kl"] += choice_kl(logits, target["choice_logits"], ids, cfg["temperature"]).item()
            _, kn, kc = component_loss(key, target_k)
            _, vn, vc = component_loss(value, target_v)
            for metric, metric_value in (("k_nmse", kn), ("v_nmse", vn), ("k_cosine", kc), ("v_cosine", vc)):
                totals[name][metric] += metric_value.item()
            record["conditions"][name] = {"prediction": prediction, "correct": bool(is_correct),
                "oracle_agreement": bool(prediction == oracle), "choice_logits": logits[ids].cpu().tolist()}
        record["relative_delta_k"] = ratios["k"][-1]
        record["relative_delta_v"] = ratios["v"][-1]
        records.append(record)
        if number % 16 == 0 or number == len(selected):
            log(f"Final A/B evaluation {number}/{len(selected)}")
    metrics = {name: {"accuracy": values["accuracy"] / len(selected),
                      "oracle_agreement": values["agreement"] / len(selected),
                      "oracle_choice_kl": values["kl"] / len(selected),
                      **{key: values[key] / len(selected) for key in ("k_nmse", "v_nmse", "k_cosine", "v_cosine")}}
               for name, values in totals.items()}
    result = {"status": "completed", "protocol": cfg["protocol"], "train_samples": cfg["train_samples"],
        "validation_samples": cfg["validation_samples"], "test_samples": len(selected),
        "stage_a_checkpoint": str(stage_a_path), "metrics": metrics,
        "relative_residual_norm": {"k": quantiles(ratios["k"]), "v": quantiles(ratios["v"])},
        "paired_stage_a_vs_residual": correctness_pair(correct["stage_a"], correct["residual"]),
        "paired_stage_a_vs_shared": correctness_pair(correct["stage_a"], correct["shared"]),
        "paired_residual_vs_shared": correctness_pair(correct["residual"], correct["shared"]),
        "frozen_base_hash_verified": modules["residual"][2]["base_hash_before"] == modules["residual"][2]["base_hash_after"],
        "trained_parameters": {"stage_a": sum(p.numel() for p in stage_a.parameters()),
                               "residual": modules["residual"][2]["parameters_trained"],
                               "shared": modules["shared"][2]["parameters_trained"]}}
    baseline_path = run_root(cfg) / "baselines" / "summary.json"
    if baseline_path.exists():
        result["baselines"] = read_json(baseline_path)
        oracle_accuracy = result["baselines"]["metrics"]["qwen_native_oracle32"]["accuracy"]
        result["translation_gap"] = {
            "definition": "Qwen Native Oracle32 accuracy minus translated accuracy",
            "stage_a": oracle_accuracy - metrics["stage_a"]["accuracy"],
            "residual": oracle_accuracy - metrics["residual"]["accuracy"],
            "shared": oracle_accuracy - metrics["shared"]["accuracy"],
        }
    root = run_root(cfg) / "results"
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "comparison.json", result)
    write_results_report(cfg, result)
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    del modules, stage_a
    torch.cuda.empty_cache()
    return result
