from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import digest, load_model, log, read_json, run_root, save_json, seed_all
from modules import HybridChunkCompressor, full_kl
from protocol import final_logits


CONDITIONS = ("compact_native_only", "position_matched_native_only", "real_slot_hybrid")


def source_cfg(cfg):
    return read_json(Path(cfg["source_root"]) / "run_config.json")


def test_rows(cfg):
    payload = read_json(Path(cfg["source_root"]) / "manifests" / "test.json")
    if payload["signature"] != source_cfg(cfg)["signature"] or len(payload["rows"]) < cfg["test_samples"]:
        raise RuntimeError("Invalid source test manifest")
    return payload["rows"][:cfg["test_samples"]]


def native(cfg, family, row):
    path = Path(cfg["source_root"]) / "cache" / family / "test" / f"{row['id']}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != source_cfg(cfg)["signature"]:
        raise RuntimeError("Native cache signature mismatch")
    if payload["tokens_hash"] != digest(row["encoded"][family]):
        raise RuntimeError("Native cache token mismatch")
    return payload


def load_selection(cfg, family, row):
    run_cfg = read_json(Path(cfg["hybrid_root"]) / "run_config.json")
    path = Path(cfg["hybrid_root"]) / "selection_cache" / family / "test" / f"{row['id']}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != run_cfg["signature"]:
        raise RuntimeError("Selection cache signature mismatch")
    return payload["selected"]["query"]


def load_query_module(cfg, family):
    layers = 36 if family == "qwen" else 28
    module = HybridChunkCompressor(layers, cfg["chunks"]).cuda()
    path = Path(cfg["hybrid_root"]) / "checkpoints" / family / "query" / "best.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["family"] != family or payload["arm"] != "query":
        raise RuntimeError("Wrong Hybrid Query checkpoint")
    module.load_state_dict(payload["state"]); return module.eval(), payload


def native_entries(module, key, value, selected):
    outputs = module(key[:, 1:], value[:, 1:], selected)
    return outputs[0], outputs[1], outputs[4], outputs


def build_condition(module, key, value, selected, condition):
    nk, nv, native_valid, outputs = native_entries(module, key, value, selected)
    device = key.device
    if condition == "compact_native_only":
        return (torch.cat((key[:, :1], nk), 1), torch.cat((value[:, :1], nv), 1),
                torch.cat((torch.ones(1, dtype=torch.bool, device=device), native_valid)),
                torch.arange(1 + module.slots, device=device), 1 + module.slots)
    if condition == "position_matched_native_only":
        zero_k, zero_v = torch.zeros_like(nk), torch.zeros_like(nv)
        memory_k = torch.stack((nk, zero_k), 2).flatten(1, 2)
        memory_v = torch.stack((nv, zero_v), 2).flatten(1, 2)
        pair_mask = torch.stack((native_valid, torch.zeros_like(native_valid)), 1).flatten()
        return (torch.cat((key[:, :1], memory_k), 1), torch.cat((value[:, :1], memory_v), 1),
                torch.cat((torch.ones(1, dtype=torch.bool, device=device), pair_mask)),
                torch.arange(1 + 2 * module.slots, device=device), 1 + 2 * module.slots)
    if condition == "real_slot_hybrid":
        memory_k, memory_v, mask = module.interleave(key[:, :1], value[:, :1], outputs)
        return memory_k, memory_v, mask, torch.arange(1 + 2 * module.slots, device=device), 1 + 2 * module.slots
    raise ValueError(condition)


def choice_kl(student, teacher, ids):
    index = torch.tensor(ids, device=student.device)
    return F.kl_div(F.log_softmax(student[index], -1), F.log_softmax(teacher[index], -1),
                    reduction="sum", log_target=True).item()


@torch.no_grad()
def evaluate_family(cfg, family, model, module, rows):
    totals = {condition: {name: 0.0 for name in
              ("accuracy", "native_accuracy", "native_agreement", "full_kl", "choice_kl")}
              for condition in CONDITIONS}
    records = []
    for number, row in enumerate(rows, 1):
        payload = native(cfg, family, row)
        key, value, teacher = payload["k"].cuda(), payload["v"].cuda(), payload["native_logits"].cuda()
        selected = load_selection(cfg, family, row).cuda()
        ids = row["encoded"][family]["choice_ids"]
        native_prediction = int(teacher[ids].argmax())
        record = {"id": row["id"], "family": family, "gold_index": row["gold_index"],
                  "native_prediction": native_prediction, "conditions": {}}
        for condition in CONDITIONS:
            mk, mv, mask, positions, suffix_start = build_condition(module, key, value, selected, condition)
            student = final_logits(model, row["encoded"][family]["suffix"], mk, mv,
                                   positions=positions, suffix_start=suffix_start,
                                   prefix_attention_mask=mask)
            prediction = int(student[ids].argmax())
            values = {"accuracy": float(prediction == row["gold_index"]),
                      "native_accuracy": float(native_prediction == row["gold_index"]),
                      "native_agreement": float(prediction == native_prediction),
                      "full_kl": full_kl(student, teacher, cfg["temperature"]).item(),
                      "choice_kl": choice_kl(student, teacher, ids)}
            record["conditions"][condition] = {"prediction": prediction, **values}
            for name, value_ in values.items(): totals[condition][name] += value_
        records.append(record)
        if number % 16 == 0: log(f"{family} slot-contribution evaluation: {number}/{len(rows)}")
    metrics = {condition: {name: value / len(rows) for name, value in values.items()}
               for condition, values in totals.items()}
    for values in metrics.values():
        values["retention"] = (values["accuracy"] / values["native_accuracy"]
                               if values["native_accuracy"] else None)
        values["sample_count"] = len(rows)
    return metrics, records


def existing_slot32(cfg, family):
    payload = read_json(cfg["slot32_summary"])
    return payload["results"][family]["hard_chunk"]["test_metrics"]


def paired_effect(records, left, right):
    deltas = [row["conditions"][left]["accuracy"] - row["conditions"][right]["accuracy"]
              for row in records]
    return {"accuracy_difference": sum(deltas) / len(deltas),
            "left_correct_right_wrong": sum(delta > 0 for delta in deltas),
            "left_wrong_right_correct": sum(delta < 0 for delta in deltas),
            "same_correctness": sum(delta == 0 for delta in deltas)}


def run(cfg):
    rows = test_rows(cfg); results, all_records = {}, []
    for family in ("qwen", "llama"):
        seed_all(cfg["seed"] + (family == "llama"))
        model, module = load_model(cfg, family), None
        try:
            module, checkpoint = load_query_module(cfg, family)
            metrics, records = evaluate_family(cfg, family, model, module, rows)
            results[family] = {"slot32_existing": existing_slot32(cfg, family), **metrics,
                "slot_net_contribution_C_minus_D": paired_effect(records, "real_slot_hybrid", "position_matched_native_only"),
                "position_protocol_effect_B1_minus_D": paired_effect(records, "compact_native_only", "position_matched_native_only"),
                "query_checkpoint_step": checkpoint["step"]}
            all_records.extend(records)
        finally:
            del model, module; torch.cuda.empty_cache()
    root = run_root(cfg); (root / "results").mkdir(parents=True, exist_ok=True)
    with (root / "results" / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as output:
        for row in all_records: output.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(root / "results" / "summary.json", {
        "experiment": "queryaware_hybrid32x32_slot_contribution_ablation",
        "primary_contrast": "real_slot_hybrid minus position_matched_native_only",
        "no_retraining": True, "test_samples": len(rows), "results": results})
    log("ALL SLOT CONTRIBUTION ABLATIONS COMPLETED")
