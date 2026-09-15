from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import digest, load_model, log, read_json, run_root, save_json, save_tensor, seed_all
from modules import SoftLocalSlotCompressor, full_kl
from protocol import final_logits


def source_cfg(cfg):
    return read_json(Path(cfg["source_root"]) / "run_config.json")


def rows(cfg, split, count):
    payload = read_json(Path(cfg["source_root"]) / "manifests" / f"{split}.json")
    expected = source_cfg(cfg)["signature"]
    if payload["signature"] != expected: raise RuntimeError("source manifest signature mismatch")
    if len(payload["rows"]) < count: raise RuntimeError(f"source {split} has too few rows")
    return payload["rows"][:count]


def native(cfg, family, split, row):
    path = Path(cfg["source_root"]) / "cache" / family / split / f"{row['id']}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = source_cfg(cfg)["signature"]
    if payload["signature"] != expected: raise RuntimeError("source cache signature mismatch")
    if payload["tokens_hash"] != digest(row["encoded"][family]): raise RuntimeError("source cache token mismatch")
    return payload


def strength_key(strength):
    return f"lambda_{float(strength):g}"


def checkpoint(cfg, family, strength, name):
    return run_root(cfg) / "checkpoints" / family / strength_key(strength) / f"{name}.pt"


def save_checkpoint(cfg, family, strength, name, module, step, loss):
    save_tensor(checkpoint(cfg, family, strength, name), {
        "signature": cfg["signature"], "family": family, "slots": cfg["slots"],
        "locality_strength": float(strength),
        "step": step, "validation_full_kl": loss,
        "state": {key: value.detach().cpu() for key, value in module.state_dict().items()},
    })


def restore(cfg, family, strength, name, module):
    payload = torch.load(checkpoint(cfg, family, strength, name), map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]: raise RuntimeError("checkpoint signature mismatch")
    module.load_state_dict(payload["state"])
    return payload


def forward(cfg, family, model, module, row, split):
    payload = native(cfg, family, split, row)
    native_key, native_value = payload["k"].cuda(), payload["v"].cuda()
    if native_key.shape[1] < 2: raise RuntimeError("token0 bypass requires at least two context tokens")
    # token0 is a fixed Native bypass; the learned slots cover only tokens 1..T-1.
    slot_key, slot_value = module(native_key[:, 1:], native_value[:, 1:])
    key = torch.cat((native_key[:, :1], slot_key), dim=1)
    value = torch.cat((native_value[:, :1], slot_value), dim=1)
    positions = torch.arange(cfg["slots"] + 1, device=key.device)
    logits = final_logits(model, row["encoded"][family]["suffix"], key, value,
                          positions=positions, suffix_start=cfg["slots"] + 1)
    return logits, payload["native_logits"].cuda(), slot_key, slot_value, payload


def loss(cfg, family, model, module, row, split):
    student, teacher, _, _, _ = forward(cfg, family, model, module, row, split)
    return full_kl(student, teacher, cfg["temperature"])


@torch.no_grad()
def validation_loss(cfg, family, model, module, selected, split):
    module.eval(); values = [loss(cfg, family, model, module, row, split).item() for row in selected]; module.train()
    if not values or not all(math.isfinite(x) for x in values): raise RuntimeError("invalid validation loss")
    return sum(values) / len(values)


def off_diagonal_slot_cosine(tensor):
    normalized = F.normalize(tensor.float().permute(0, 2, 1, 3), dim=-1)
    matrix = torch.einsum("lhsd,lhtd->lhst", normalized, normalized)
    count = matrix.shape[-1]
    return ((matrix.sum((-1, -2)) - count) / max(count * (count - 1), 1)).mean().item()


def attention_stats(module, native_key):
    weights = module.attention(native_key.cuda())
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
    maximum = math.log(weights.shape[-1]) if weights.shape[-1] > 1 else 1.0
    token_position = (torch.arange(weights.shape[-1], device=weights.device) + 0.5) / weights.shape[-1]
    slot_center = (torch.arange(module.slots, device=weights.device) + 0.5) / module.slots
    distance = (slot_center[:, None] - token_position[None, :]).abs()
    attended_distance = (weights * distance[None, None]).sum(-1).mean().item()
    return ((entropy / maximum).mean().item(), entropy.exp().mean().item(),
            weights.max(-1).values.mean().item(), attended_distance)


def choice_kl(student, teacher, ids):
    index = torch.tensor(ids, device=student.device)
    s, t = F.log_softmax(student[index], -1), F.log_softmax(teacher[index], -1)
    return F.kl_div(s, t, reduction="sum", log_target=True).item()


@torch.no_grad()
def evaluate(cfg, family, model, module, selected, split):
    module.eval()
    names = ("full_kl", "choice_kl", "native_agreement", "accuracy", "slot_k_cosine", "slot_v_cosine",
             "normalized_attention_entropy", "effective_attended_tokens", "maximum_attention_weight",
             "mean_attended_normalized_distance", "native_accuracy", "without_token0_full_kl",
             "without_token0_choice_kl", "without_token0_native_agreement", "without_token0_accuracy")
    totals = {name: 0.0 for name in names}; lengths = []
    for row in selected:
        student, teacher, key, value, payload = forward(cfg, family, model, module, row, split)
        ids = row["encoded"][family]["choice_ids"]
        prediction, native_prediction = int(student[ids].argmax()), int(teacher[ids].argmax())
        no_token0 = final_logits(
            model, row["encoded"][family]["suffix"], key, value,
            positions=torch.arange(1, cfg["slots"] + 1, device=key.device),
            suffix_start=cfg["slots"] + 1)
        no_token0_prediction = int(no_token0[ids].argmax())
        entropy, effective, maximum, distance = attention_stats(module, payload["k"][:, 1:])
        totals["full_kl"] += full_kl(student, teacher, cfg["temperature"]).item()
        totals["choice_kl"] += choice_kl(student, teacher, ids)
        totals["native_agreement"] += float(prediction == native_prediction)
        totals["accuracy"] += float(prediction == row["gold_index"])
        totals["native_accuracy"] += float(native_prediction == row["gold_index"])
        totals["without_token0_full_kl"] += full_kl(no_token0, teacher, cfg["temperature"]).item()
        totals["without_token0_choice_kl"] += choice_kl(no_token0, teacher, ids)
        totals["without_token0_native_agreement"] += float(no_token0_prediction == native_prediction)
        totals["without_token0_accuracy"] += float(no_token0_prediction == row["gold_index"])
        totals["slot_k_cosine"] += off_diagonal_slot_cosine(key)
        totals["slot_v_cosine"] += off_diagonal_slot_cosine(value)
        totals["normalized_attention_entropy"] += entropy
        totals["effective_attended_tokens"] += effective
        totals["maximum_attention_weight"] += maximum
        totals["mean_attended_normalized_distance"] += distance
        lengths.append(len(row["encoded"][family]["prefix"]))
    result = {name: value / len(selected) for name, value in totals.items()}
    result["sample_count"] = len(selected)
    result["mean_native_prefix_tokens"] = sum(lengths) / len(lengths)
    result["native_prefix_shorter_than_slots_fraction"] = sum(length < module.slots for length in lengths) / len(lengths)
    return result


def train_one(cfg, family, strength, model, initial_queries):
    train_rows = rows(cfg, "train", cfg["train_samples"])
    validation_rows = rows(cfg, "validation", cfg["validation_samples"])
    test_rows = rows(cfg, "test", cfg["test_samples"])
    layers = 36 if family == "qwen" else 28
    module = SoftLocalSlotCompressor(layers, cfg["slots"], locality_strength=strength).cuda()
    with torch.no_grad(): module.queries.copy_(initial_queries)
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["learning_rate"], weight_decay=0)
    initial = validation_loss(cfg, family, model, module, validation_rows, "validation")
    best, best_step = initial, 0
    save_checkpoint(cfg, family, strength, "initial", module, 0, initial)
    save_checkpoint(cfg, family, strength, "best", module, 0, initial)
    validations = [{"step": 0, "validation_full_kl": initial, "selected": True}]
    root = run_root(cfg) / "training" / family / strength_key(strength); root.mkdir(parents=True, exist_ok=True)
    step_file = (root / "steps.jsonl").open("w", encoding="utf-8")
    step, epoch, clipped, exposures, started = 0, 0, 0, 0, time.monotonic()
    try:
        while step < cfg["optimizer_steps"]:
            epoch += 1; order = list(range(len(train_rows)))
            random.Random(cfg["seed"] + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                indices = order[begin:begin + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True); values = []
                for index in indices:
                    value = loss(cfg, family, model, module, train_rows[index], "train")
                    if not torch.isfinite(value): raise RuntimeError("nonfinite training loss")
                    (value / len(indices)).backward(); values.append(value.item())
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if module.queries.grad is None or not torch.isfinite(norm): raise RuntimeError("invalid query gradient")
                did_clip = norm.item() > cfg["clip"]; clipped += int(did_clip)
                optimizer.step(); step += 1; exposures += len(indices)
                record = {"epoch": epoch, "step": step, "mean_train_full_kl": sum(values) / len(values),
                          "pre_clip_grad_norm": norm.item(), "clipped": did_clip}
                step_file.write(json.dumps(record) + "\n"); step_file.flush()
                if step % 16 == 0: log(f"{family} lambda={strength:g} step={step}/{cfg['optimizer_steps']} loss={record['mean_train_full_kl']:.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["optimizer_steps"]:
                    score = validation_loss(cfg, family, model, module, validation_rows, "validation")
                    selected = score < best
                    if selected:
                        best, best_step = score, step
                        save_checkpoint(cfg, family, strength, "best", module, step, score)
                    validations.append({"step": step, "validation_full_kl": score, "selected": selected})
                    save_json(root / "validations.json", validations)
                    log(f"{family} lambda={strength:g} validation step={step} KL={score:.6f}")
                if step == cfg["optimizer_steps"]: break
        save_checkpoint(cfg, family, strength, "last", module, step, validations[-1]["validation_full_kl"])
    finally:
        step_file.close()
    restore(cfg, family, strength, "best", module)
    summary = {
        "family": family, "slots": cfg["slots"], "locality_strength": float(strength),
        "train_samples": len(train_rows), "validation_samples": len(validation_rows),
        "test_samples": len(test_rows), "optimizer_steps": step, "sample_exposures": exposures,
        "initial_validation_full_kl": initial, "best_validation_full_kl": best, "best_step": best_step,
        "relative_validation_kl_reduction": (initial - best) / initial, "clip_rate": clipped / step,
        "initialization_seed": cfg["seed"], "gold_used_for_training": False,
        "native_token0_bypass": True, "slot_pool_excludes_token0": True,
        "validation_metrics": evaluate(cfg, family, model, module, validation_rows, "validation"),
        "test_metrics": evaluate(cfg, family, model, module, test_rows, "test"),
        "seconds": time.monotonic() - started,
    }
    save_json(root / "summary.json", summary)
    del module, optimizer; torch.cuda.empty_cache()
    return summary


def run(cfg):
    result = {}
    for family in ("qwen", "llama"):
        seed_all(cfg["seed"] + (0 if family == "qwen" else 1))
        layers = 36 if family == "qwen" else 28
        initializer = SoftLocalSlotCompressor(layers, cfg["slots"], locality_strength=0.0)
        initial_queries = initializer.queries.detach().cpu().clone(); del initializer
        model = load_model(cfg, family); result[family] = {}
        try:
            for strength in cfg["locality_strengths"]:
                result[family][strength_key(strength)] = train_one(cfg, family, strength, model, initial_queries.cuda())
        finally:
            del model; torch.cuda.empty_cache()
    save_json(run_root(cfg) / "results/summary.json", {
        "experiment": "token0_plus_slot64_joint_training",
        "controlled": "fixed Native token0 bypass plus 64 learned slots over tokens 1..T-1; 1024 train examples, four true epochs, 512 optimizer steps and 4096 exposures",
        "locality_formula": "-lambda * abs((t+0.5)/T - (i+0.5)/64)",
        "position_protocol": "token0=0, slots=1..64, receiver suffix starts at 65",
        "results": result,
    })
    log("ALL TOKEN0 PLUS 64-SLOT JOINT TRAINING COMPLETED")
