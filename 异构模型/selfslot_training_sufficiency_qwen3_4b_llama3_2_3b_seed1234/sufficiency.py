from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_model, log, manifests, run_root, save_json, save_tensor, seed_all
from modules import SlotCompressor, full_kl
from protocol import final_logits, load_native


def arm_root(cfg, family, arm):
    return run_root(cfg) / "training" / family / arm


def checkpoint(cfg, family, arm, name):
    return arm_root(cfg, family, arm) / f"{name}.pt"


def save_checkpoint(cfg, family, arm, name, module, step, validation):
    save_tensor(checkpoint(cfg, family, arm, name), {
        "signature": cfg["signature"], "family": family, "arm": arm,
        "step": step, "validation_full_kl": validation,
        "state": {key: value.detach().cpu() for key, value in module.state_dict().items()},
    })


def restore(cfg, family, arm, module, name="best"):
    payload = torch.load(checkpoint(cfg, family, arm, name), map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]: raise RuntimeError("checkpoint signature mismatch")
    module.load_state_dict(payload["state"])
    return payload


def student_logits(cfg, family, model, compressor, row, split):
    native = load_native(cfg, family, split, row)
    key, value = compressor(native["k"].cuda(), native["v"].cuda())
    logits = final_logits(model, row["encoded"][family]["suffix"], key, value)
    return logits, native["native_logits"].cuda(), key, value


def training_loss(cfg, family, model, compressor, row, split):
    student, teacher, _, _ = student_logits(cfg, family, model, compressor, row, split)
    return full_kl(student, teacher, cfg["temperature"])


@torch.no_grad()
def mean_full_kl(cfg, family, model, compressor, rows, split):
    compressor.eval()
    values = [training_loss(cfg, family, model, compressor, row, split).item() for row in rows]
    compressor.train()
    if not values or not all(math.isfinite(x) for x in values): raise RuntimeError("invalid validation loss")
    return sum(values) / len(values)


def choice_kl(student, teacher, choice_ids):
    ids = torch.tensor(choice_ids, device=student.device)
    s = F.log_softmax(student.index_select(0, ids), dim=-1)
    t = F.log_softmax(teacher.index_select(0, ids), dim=-1)
    return F.kl_div(s, t, reduction="sum", log_target=True).item()


def pairwise_slot_cosine(tensor):
    # Average off-diagonal cosine among the 64 slots, independently by layer/head.
    x = F.normalize(tensor.float().permute(0, 2, 1, 3), dim=-1)
    matrix = torch.einsum("lhsd,lhtd->lhst", x, x)
    count = matrix.shape[-1]
    return ((matrix.sum((-1, -2)) - count) / max(count * (count - 1), 1)).mean().item()


def attention_diagnostics(compressor, native_key):
    keys = native_key.float().cuda().permute(0, 2, 1, 3)
    scale = keys.square().mean((-1, -2), keepdim=True).sqrt().clamp_min(1e-6)
    scores = torch.einsum("lhsd,lhtd->lhst", compressor.queries.float(), keys / scale) / math.sqrt(compressor.dim)
    weights = scores.softmax(-1)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
    maximum = math.log(weights.shape[-1]) if weights.shape[-1] > 1 else 1.0
    return {
        "normalized_attention_entropy": (entropy / maximum).mean().item(),
        "effective_attended_tokens": entropy.exp().mean().item(),
        "maximum_attention_weight": weights.max(-1).values.mean().item(),
    }


@torch.no_grad()
def functional_metrics(cfg, family, model, compressor, rows, split, diagnostics=False):
    compressor.eval()
    totals = {"full_kl": 0.0, "choice_kl": 0.0, "native_agreement": 0.0, "accuracy": 0.0,
              "slot_k_cosine": 0.0, "slot_v_cosine": 0.0}
    attention_rows = []
    for index, row in enumerate(rows):
        student, teacher, key, value = student_logits(cfg, family, model, compressor, row, split)
        choice_ids = row["encoded"][family]["choice_ids"]
        student_choice = student[choice_ids]
        teacher_choice = teacher[choice_ids]
        prediction, native_prediction = int(student_choice.argmax()), int(teacher_choice.argmax())
        totals["full_kl"] += full_kl(student, teacher).item()
        totals["choice_kl"] += choice_kl(student, teacher, choice_ids)
        totals["native_agreement"] += float(prediction == native_prediction)
        totals["accuracy"] += float(prediction == row["gold_index"])
        totals["slot_k_cosine"] += pairwise_slot_cosine(key)
        totals["slot_v_cosine"] += pairwise_slot_cosine(value)
        if diagnostics and index < cfg["diagnostic_samples"]:
            native = load_native(cfg, family, split, row)
            attention_rows.append(attention_diagnostics(compressor, native["k"]))
    result = {name: value / len(rows) for name, value in totals.items()}
    if attention_rows:
        result.update({name: sum(row[name] for row in attention_rows) / len(attention_rows)
                       for name in attention_rows[0]})
        result["attention_diagnostic_samples"] = len(attention_rows)
    return result


def train_arm(cfg, family, model, arm, base_state):
    spec = cfg["arms"][arm]
    all_train = manifests(cfg, "train", labels=True)
    train_rows = all_train[:spec["train_samples"]]
    if spec["validation"] == "train":
        validation_rows, validation_split = train_rows, "train"
    else:
        validation_rows, validation_split = manifests(cfg, "validation", labels=True), "validation"
    if len(train_rows) % cfg["batch_size"]: raise RuntimeError("train count must divide batch size")
    expected_steps = spec["epochs"] * len(train_rows) // cfg["batch_size"]
    if expected_steps != cfg["optimizer_steps"]:
        raise RuntimeError(f"{arm} expected {expected_steps} steps, configured {cfg['optimizer_steps']}")

    seed_all(cfg["seed"] + (0 if family == "qwen" else 1))
    compressor = SlotCompressor(36 if family == "qwen" else 28, cfg["slots"]).cuda()
    compressor.load_state_dict(base_state)
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=cfg["learning_rate"], weight_decay=0.0)
    root = arm_root(cfg, family, arm); root.mkdir(parents=True, exist_ok=True)
    initial = mean_full_kl(cfg, family, model, compressor, validation_rows, validation_split)
    best, best_step = initial, 0
    save_checkpoint(cfg, family, arm, "initial", compressor, 0, initial)
    save_checkpoint(cfg, family, arm, "best", compressor, 0, initial)
    validations = [{"step": 0, "validation_full_kl": initial, "selected": True}]
    steps, clipped, exposures, started = 0, 0, 0, time.monotonic()
    step_file = (root / "steps.jsonl").open("w", encoding="utf-8")
    try:
        for epoch in range(1, spec["epochs"] + 1):
            order = list(range(len(train_rows)))
            random.Random(cfg["seed"] + epoch).shuffle(order)
            if len(set(order)) != len(train_rows): raise RuntimeError("sampler is not without replacement")
            for begin in range(0, len(order), cfg["batch_size"]):
                indices = order[begin:begin + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True); losses = []
                for sample_index in indices:
                    loss = training_loss(cfg, family, model, compressor, train_rows[sample_index], "train")
                    if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
                    (loss / len(indices)).backward(); losses.append(loss.item())
                if compressor.queries.grad is None or not torch.isfinite(compressor.queries.grad).all():
                    raise RuntimeError("invalid slot-query gradient")
                norm = torch.nn.utils.clip_grad_norm_(compressor.parameters(), cfg["clip"])
                if not torch.isfinite(norm): raise RuntimeError("nonfinite gradient norm")
                did_clip = norm.item() > cfg["clip"]
                optimizer.step(); steps += 1; exposures += len(indices); clipped += int(did_clip)
                entry = {"epoch": epoch, "step": steps, "mean_train_full_kl": sum(losses) / len(losses),
                         "pre_clip_grad_norm": norm.item(), "clipped": did_clip,
                         "sample_ids": [train_rows[i]["id"] for i in indices]}
                step_file.write(json.dumps(entry) + "\n"); step_file.flush()
                if steps % 16 == 0: log(f"{family}/{arm} step={steps}/{cfg['optimizer_steps']} loss={entry['mean_train_full_kl']:.6f}")
                if steps % cfg["validation_interval_steps"] == 0 or steps == cfg["optimizer_steps"]:
                    score = mean_full_kl(cfg, family, model, compressor, validation_rows, validation_split)
                    selected = score < best
                    if selected:
                        best, best_step = score, steps
                        save_checkpoint(cfg, family, arm, "best", compressor, steps, score)
                    save_checkpoint(cfg, family, arm, f"step_{steps}", compressor, steps, score)
                    validations.append({"step": steps, "validation_full_kl": score, "selected": selected})
                    save_json(root / "validations.json", validations)
                    log(f"{family}/{arm} validation step={steps} KL={score:.6f} best={best:.6f}")
        save_checkpoint(cfg, family, arm, "last", compressor, steps, validations[-1]["validation_full_kl"])
    finally:
        step_file.close()
    restore(cfg, family, arm, compressor, "best")
    fit_rows, fit_split = (train_rows, "train") if arm == "overfit16" else (validation_rows, validation_split)
    best_fit = functional_metrics(cfg, family, model, compressor, fit_rows, fit_split, True)
    test_rows = manifests(cfg, "test", labels=True)
    best_test = functional_metrics(cfg, family, model, compressor, test_rows, "test", True)
    save_json(root / "summary.json", {
        "family": family, "arm": arm, "train_samples": len(train_rows), "epochs": spec["epochs"],
        "optimizer_steps": steps, "sample_exposures": exposures, "initial_validation_full_kl": initial,
        "best_validation_full_kl": best, "best_step": best_step, "relative_kl_reduction": (initial - best) / initial,
        "clip_rate": clipped / steps, "true_epoch_sampler": True, "gold_used_for_training": False,
        "same_initialization_across_arms": True, "fit_metrics": best_fit, "test_metrics": best_test,
        "seconds": time.monotonic() - started,
    })
    del compressor, optimizer
    torch.cuda.empty_cache()


def train_family(cfg, family):
    seed_all(cfg["seed"] + (0 if family == "qwen" else 1))
    base = SlotCompressor(36 if family == "qwen" else 28, cfg["slots"])
    base_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
    save_tensor(run_root(cfg) / "initialization" / f"{family}.pt", {"signature": cfg["signature"], "state": base_state})
    del base
    model = load_model(cfg, family)
    try:
        for arm in ("overfit16", "repeat128", "diverse1024"):
            train_arm(cfg, family, model, arm, base_state)
    finally:
        del model
        torch.cuda.empty_cache()


def compare(cfg):
    results = {}
    for family in ("qwen", "llama"):
        results[family] = {arm: json.loads((arm_root(cfg, family, arm) / "summary.json").read_text())
                           for arm in cfg["arms"]}
    save_json(run_root(cfg) / "results/summary.json", {
        "design": "equal 512 optimizer steps and 4096 sample exposures per arm",
        "interpretation": {
            "overfit16": "capacity and optimization sanity check on the fitted examples",
            "repeat128": "more repeated optimization with the original data size",
            "diverse1024": "more data diversity at equal optimizer steps and exposures",
        },
        "results": results,
    })
    log("Saved training-sufficiency comparison")
