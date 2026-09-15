from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import digest, load_model, log, read_json, run_root, save_json, save_tensor, seed_all
from modules import HybridChunkCompressor, full_kl
from protocol import final_logits, query_context_importance


def source_cfg(cfg):
    return read_json(Path(cfg["source_root"]) / "run_config.json")


def rows(cfg, split, count):
    payload = read_json(Path(cfg["source_root"]) / "manifests" / f"{split}.json")
    if payload["signature"] != source_cfg(cfg)["signature"] or len(payload["rows"]) < count:
        raise RuntimeError(f"Invalid source manifest: {split}")
    return payload["rows"][:count]


def native(cfg, family, split, row):
    path = Path(cfg["source_root"]) / "cache" / family / split / f"{row['id']}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != source_cfg(cfg)["signature"]:
        raise RuntimeError("Source cache signature mismatch")
    if payload["tokens_hash"] != digest(row["encoded"][family]):
        raise RuntimeError("Source cache token mismatch")
    return payload


def stable_seed(cfg, family, row):
    text = f"{cfg['seed']}:{family}:{row['id']}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def selection_path(cfg, family, split, row):
    return run_root(cfg) / "selection_cache" / family / split / f"{row['id']}.pt"


@torch.no_grad()
def prepare_selections(cfg, family, model, split, selected_rows):
    selector = HybridChunkCompressor(36 if family == "qwen" else 28, cfg["chunks"])
    for number, row in enumerate(selected_rows, 1):
        path = selection_path(cfg, family, split, row)
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("signature") == cfg["signature"]:
                continue
        cache = native(cfg, family, split, row)
        fields = row["encoded"][family]
        importance = query_context_importance(model, fields["prefix"], fields["suffix"], cache["k"])
        post0 = importance[1:].float()
        choices = {arm: selector.select(post0, arm, stable_seed(cfg, family, row)).cpu()
                   for arm in cfg["arms"]}
        save_tensor(path, {"signature": cfg["signature"], "importance": importance,
                           "selected": choices, "prefix_tokens": len(fields["prefix"])})
        log(f"{family} selector {split}: {number}/{len(selected_rows)}")


def load_all_selections(cfg, family, split_rows):
    return {split: {row["id"]: torch.load(selection_path(cfg, family, split, row),
                                               map_location="cpu", weights_only=True)
                    for row in selected_rows}
            for split, selected_rows in split_rows.items()}


def checkpoint(cfg, family, arm, name):
    return run_root(cfg) / "checkpoints" / family / arm / f"{name}.pt"


def save_checkpoint(cfg, family, arm, name, module, step, loss):
    save_tensor(checkpoint(cfg, family, arm, name), {
        "signature": cfg["signature"], "family": family, "arm": arm, "step": step,
        "validation_full_kl": loss,
        "state": {k: v.detach().cpu() for k, v in module.state_dict().items()}})


def restore(cfg, family, arm, name, module):
    payload = torch.load(checkpoint(cfg, family, arm, name), map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]:
        raise RuntimeError("Checkpoint signature mismatch")
    module.load_state_dict(payload["state"]); return payload


def schema_positions(module, selected, prefix_tokens, mode, device):
    if mode == "canonical":
        return torch.arange(1 + 2 * module.slots, device=device), 1 + 2 * module.slots
    if mode != "original":
        raise ValueError(mode)
    positions = [0]
    for index, (start, end) in zip(selected.tolist(), module.chunk_bounds(prefix_tokens - 1)):
        positions.extend([index + 1 if index >= 0 else 0,
                          1 + (start + end - 1) // 2 if end > start else 0])
    return torch.tensor(positions, device=device), prefix_tokens


def forward(cfg, family, model, module, row, split, selection, arm, position_mode="canonical"):
    payload = native(cfg, family, split, row)
    key, value = payload["k"].cuda(), payload["v"].cuda()
    selected = selection["selected"][arm].to(key.device)
    outputs = module(key[:, 1:], value[:, 1:], selected)
    hybrid_key, hybrid_value, prefix_mask = module.interleave(key[:, :1], value[:, :1], outputs)
    positions, suffix_start = schema_positions(module, selected, key.shape[1], position_mode, key.device)
    student = final_logits(model, row["encoded"][family]["suffix"], hybrid_key, hybrid_value,
                           positions=positions, suffix_start=suffix_start,
                           prefix_attention_mask=prefix_mask)
    return student, payload["native_logits"].cuda(), outputs, selected, selection


def loss(cfg, family, model, module, row, split, selection, arm):
    student, teacher, *_ = forward(cfg, family, model, module, row, split, selection, arm)
    return full_kl(student, teacher, cfg["temperature"])


@torch.no_grad()
def validation_loss(cfg, family, model, module, selected_rows, selections, split, arm):
    module.eval()
    values = [loss(cfg, family, model, module, row, split, selections[row["id"]], arm).item()
              for row in selected_rows]
    module.train()
    if not values or not all(math.isfinite(v) for v in values):
        raise RuntimeError("Invalid validation loss")
    return sum(values) / len(values)


def choice_kl(student, teacher, ids):
    index = torch.tensor(ids, device=student.device)
    return F.kl_div(F.log_softmax(student[index], -1), F.log_softmax(teacher[index], -1),
                    reduction="sum", log_target=True).item()


@torch.no_grad()
def evaluate(cfg, family, model, module, selected_rows, selections, split, arm, position_mode="canonical"):
    module.eval(); totals = {k: 0.0 for k in (
        "full_kl", "choice_kl", "native_agreement", "accuracy", "native_accuracy",
        "selected_attention_mass", "selected_token_normalized_position",
        "selected_within_chunk_fraction", "remaining_tokens_per_valid_slot", "empty_slot_fraction")}
    records = []
    for row in selected_rows:
        student, teacher, outputs, selected, selection = forward(
            cfg, family, model, module, row, split, selections[row["id"]], arm, position_mode)
        _, _, _, _, _, slot_valid = outputs
        ids = row["encoded"][family]["choice_ids"]
        prediction, native_prediction = int(student[ids].argmax()), int(teacher[ids].argmax())
        importance = selection["importance"][1:].float()
        valid_indices = selected[selected >= 0].cpu()
        selected_mass = importance[valid_indices].sum().item() / importance.sum().clamp_min(1e-12).item()
        positions, within, remainder = [], [], []
        for index, (start, end) in zip(selected.tolist(), module.chunk_bounds(importance.numel())):
            if index >= 0:
                positions.append((index + 1) / selection["prefix_tokens"])
                within.append(0.0 if end - start <= 1 else (index - start) / (end - start - 1))
                remainder.append(max(end - start - 1, 0))
        values = {
            "full_kl": full_kl(student, teacher, cfg["temperature"]).item(),
            "choice_kl": choice_kl(student, teacher, ids),
            "native_agreement": float(prediction == native_prediction),
            "accuracy": float(prediction == row["gold_index"]),
            "native_accuracy": float(native_prediction == row["gold_index"]),
            "selected_attention_mass": selected_mass,
            "selected_token_normalized_position": sum(positions) / len(positions),
            "selected_within_chunk_fraction": sum(within) / len(within),
            "remaining_tokens_per_valid_slot": sum(remainder) / max(len(remainder), 1),
            "empty_slot_fraction": float((~slot_valid).float().mean()),
        }
        for key, value in values.items(): totals[key] += value
        records.append({"id": row["id"], "family": family, "arm": arm,
                        "position_mode": position_mode, "prediction": prediction,
                        "native_prediction": native_prediction, "gold_index": row["gold_index"],
                        "selected_offsets_after_token0": selected.cpu().tolist(), **values})
    result = {k: v / len(selected_rows) for k, v in totals.items()}
    result["retention"] = result["accuracy"] / result["native_accuracy"] if result["native_accuracy"] else None
    result["sample_count"] = len(selected_rows)
    return result, records


def train_one(cfg, family, arm, model, initial_queries, split_rows, selections):
    layers = 36 if family == "qwen" else 28
    module = HybridChunkCompressor(layers, cfg["chunks"]).cuda()
    with torch.no_grad(): module.queries.copy_(initial_queries)
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["learning_rate"], weight_decay=0)
    initial = validation_loss(cfg, family, model, module, split_rows["validation"],
                              selections["validation"], "validation", arm)
    best, best_step = initial, 0
    save_checkpoint(cfg, family, arm, "initial", module, 0, initial)
    save_checkpoint(cfg, family, arm, "best", module, 0, initial)
    validations = [{"step": 0, "validation_full_kl": initial, "selected": True}]
    root = run_root(cfg) / "training" / family / arm; root.mkdir(parents=True, exist_ok=True)
    stream = (root / "steps.jsonl").open("w", encoding="utf-8")
    step = clipped = exposures = 0; started = time.monotonic()
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            order = list(range(len(split_rows["train"])))
            random.Random(cfg["seed"] + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                indices = order[begin:begin + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True); values = []
                for index in indices:
                    row = split_rows["train"][index]
                    value = loss(cfg, family, model, module, row, "train",
                                 selections["train"][row["id"]], arm)
                    if not torch.isfinite(value): raise RuntimeError("Nonfinite loss")
                    (value / len(indices)).backward(); values.append(value.item())
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if module.queries.grad is None or not torch.isfinite(norm): raise RuntimeError("Invalid gradient")
                did_clip = norm.item() > cfg["clip"]; clipped += int(did_clip)
                optimizer.step(); step += 1; exposures += len(indices)
                record = {"epoch": epoch, "step": step, "mean_train_full_kl": sum(values) / len(values),
                          "pre_clip_grad_norm": norm.item(), "clipped": did_clip}
                stream.write(json.dumps(record) + "\n"); stream.flush()
                if step % 16 == 0: log(f"{family} {arm} step={step}/{cfg['optimizer_steps']} loss={record['mean_train_full_kl']:.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["optimizer_steps"]:
                    score = validation_loss(cfg, family, model, module, split_rows["validation"],
                                            selections["validation"], "validation", arm)
                    chosen = score < best
                    if chosen:
                        best, best_step = score, step
                        save_checkpoint(cfg, family, arm, "best", module, step, score)
                    validations.append({"step": step, "validation_full_kl": score, "selected": chosen})
                    save_json(root / "validations.json", validations)
                    log(f"{family} {arm} validation step={step} KL={score:.6f}")
                if step >= cfg["optimizer_steps"]: break
            if step >= cfg["optimizer_steps"]: break
        save_checkpoint(cfg, family, arm, "last", module, step, validations[-1]["validation_full_kl"])
    finally:
        stream.close()
    restore(cfg, family, arm, "best", module)
    validation_metrics, _ = evaluate(cfg, family, model, module, split_rows["validation"],
                                     selections["validation"], "validation", arm)
    test_metrics, records = evaluate(cfg, family, model, module, split_rows["test"],
                                     selections["test"], "test", arm)
    oracle_metrics = None
    if arm == "query":
        oracle_metrics, oracle_records = evaluate(cfg, family, model, module, split_rows["test"],
                                                  selections["test"], "test", arm, "original")
        records.extend(oracle_records)
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as output:
        for record in records: output.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"family": family, "arm": arm, "train_samples": len(split_rows["train"]),
               "optimizer_steps": step, "sample_exposures": exposures,
               "initial_validation_full_kl": initial, "best_validation_full_kl": best,
               "best_step": best_step, "clip_rate": clipped / step,
               "gold_used_for_training": False, "selector_trainable": False,
               "validation_metrics": validation_metrics, "test_metrics": test_metrics,
               "original_position_oracle_test_metrics": oracle_metrics,
               "seconds": time.monotonic() - started}
    save_json(root / "summary.json", summary)
    del module, optimizer; torch.cuda.empty_cache()
    return summary


def comparison_results(cfg):
    result = {}
    for name, path in cfg.get("comparison_summaries", {}).items():
        p = Path(path); result[name] = read_json(p) if p.exists() else {"unavailable": str(p)}
    return result


def run(cfg):
    split_rows = {split: rows(cfg, split, cfg[f"{split}_samples"])
                  for split in ("train", "validation", "test")}
    result = {}
    for family in ("qwen", "llama"):
        seed_all(cfg["seed"] + (0 if family == "qwen" else 1))
        layers = 36 if family == "qwen" else 28
        initializer = HybridChunkCompressor(layers, cfg["chunks"])
        initial_queries = initializer.queries.detach().cpu().clone(); del initializer
        model = load_model(cfg, family)
        try:
            for split, selected_rows in split_rows.items():
                prepare_selections(cfg, family, model, split, selected_rows)
            selections = load_all_selections(cfg, family, split_rows)
            result[family] = {}
            for arm in cfg["arms"]:
                seed_all(cfg["seed"] + (0 if family == "qwen" else 1))
                result[family][arm] = train_one(cfg, family, arm, model, initial_queries.cuda(),
                                                split_rows, selections)
        finally:
            del model; torch.cuda.empty_cache()
    save_json(run_root(cfg) / "results" / "summary.json", {
        "experiment": "queryaware_hybrid32x32",
        "schema": "[token0,N0,S0,...,N31,S31] at canonical positions 0..64",
        "selector": "mean_layer,head max_query softmax_context(native RoPE QK/sqrt(d))",
        "training_loss": "final-position full-vocabulary KL; no gold labels",
        "results": result, "existing_comparisons": comparison_results(cfg)})
    log("ALL QUERY-AWARE HYBRID32x32 EXPERIMENTS COMPLETED")
