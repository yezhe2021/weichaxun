from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from anchors import build_anchors, selected_mass, self_anchors
from common import digest, load_model, read_json, run_root, save_json, seed_all, tokenizer, log
from modules_stub import full_kl
from offsets import token_spans
from protocol import final_logits


def source_cfg(cfg): return read_json(Path(cfg["source_root"]) / "run_config.json")


def rows(cfg):
    payload = read_json(Path(cfg["source_root"]) / "manifests" / "test.json")
    if payload["signature"] != source_cfg(cfg)["signature"] or len(payload["rows"]) < cfg["test_samples"]:
        raise RuntimeError("Invalid source manifest")
    return payload["rows"][:cfg["test_samples"]]


def native(cfg, family, row):
    payload = torch.load(Path(cfg["source_root"]) / "cache" / family / "test" / f"{row['id']}.pt",
                         map_location="cpu", weights_only=True)
    if payload["signature"] != source_cfg(cfg)["signature"] or payload["tokens_hash"] != digest(row["encoded"][family]):
        raise RuntimeError("Native cache mismatch")
    return payload


def importance(cfg, family, row):
    run_cfg = read_json(Path(cfg["hybrid_root"]) / "run_config.json")
    payload = torch.load(Path(cfg["hybrid_root"]) / "selection_cache" / family / "test" / f"{row['id']}.pt",
                         map_location="cpu", weights_only=True)
    if payload["signature"] != run_cfg["signature"]: raise RuntimeError("Importance cache mismatch")
    return payload["importance"].float()


def choice_kl(student, teacher, ids):
    index = torch.tensor(ids, device=student.device)
    return F.kl_div(F.log_softmax(student[index], -1), F.log_softmax(teacher[index], -1),
                    reduction="sum", log_target=True).item()


def evaluate_memory(cfg, family, model, row, indices):
    payload = native(cfg, family, row); key, value = payload["k"].cuda(), payload["v"].cuda()
    chosen = torch.tensor(indices, device=key.device)
    memory_k, memory_v = torch.cat((key[:, :1], key[:, chosen]), 1), torch.cat((value[:, :1], value[:, chosen]), 1)
    student, teacher = final_logits(model, row["encoded"][family]["suffix"], memory_k, memory_v,
                                    positions=torch.arange(1 + len(indices), device=key.device),
                                    suffix_start=1 + len(indices)), payload["native_logits"].cuda()
    ids = row["encoded"][family]["choice_ids"]
    prediction, native_prediction = int(student[ids].argmax()), int(teacher[ids].argmax())
    return {"prediction": prediction, "native_prediction": native_prediction,
            "accuracy": float(prediction == row["gold_index"]),
            "native_accuracy": float(native_prediction == row["gold_index"]),
            "native_agreement": float(prediction == native_prediction),
            "full_kl": full_kl(student, teacher, cfg["temperature"]).item(),
            "choice_kl": choice_kl(student, teacher, ids)}


def alignment_summary(records):
    anchors = [anchor for row in records for anchor in row["shared_anchors"]]
    distances = sorted(anchor["char_distance"] for anchor in anchors)
    quantile = lambda q: distances[min(round(q * (len(distances) - 1)), len(distances) - 1)]
    return {"anchor_count": len(anchors),
            "target_contains_anchor_rate": sum(a["contains_anchor"] for a in anchors) / len(anchors),
            "positive_span_overlap_rate": sum(a["span_overlap"] > 0 for a in anchors) / len(anchors),
            "mean_span_iou": sum(a["span_iou"] for a in anchors) / len(anchors),
            "mean_char_distance": sum(distances) / len(distances), "p50_char_distance": quantile(.5),
            "p90_char_distance": quantile(.9), "max_char_distance": max(distances),
            "source_region_fallback_rate": sum(a["source_region_fallback"] for a in anchors) / len(anchors),
            "source_duplicate_rate": sum((32 - len(set(a["source_index"] for a in row["shared_anchors"]))) / 32 for row in records) / len(records),
            "target_duplicate_rate": sum((32 - len(set(a["target_index"] for a in row["shared_anchors"]))) / 32 for row in records) / len(records)}


def aggregate(records, key):
    names = ("accuracy", "native_accuracy", "native_agreement", "full_kl", "choice_kl")
    result = {name: sum(row["metrics"][key][name] for row in records) / len(records) for name in names}
    result["retention"] = result["accuracy"] / result["native_accuracy"] if result["native_accuracy"] else None
    result["sample_count"] = len(records); return result


def old_token_chunk(cfg, family):
    return read_json(cfg["old_native_only_summary"])["results"][family]["compact_native_only"]


@torch.no_grad()
def run(cfg):
    selected_rows = rows(cfg); toks = {family: tokenizer(path) for family, path in cfg["models"].items()}
    prepared, audit = [], {family: {"rows": 0, "id_matches": 0, "special_spans": 0} for family in toks}
    for row in selected_rows:
        spans, scores = {}, {}
        for family, tok in toks.items():
            spans[family] = token_spans(tok, row["prefix_text"], row["encoded"][family]["prefix"])
            scores[family] = importance(cfg, family, row)
            if len(spans[family]) != scores[family].numel(): raise RuntimeError("Importance/span length mismatch")
            audit[family]["rows"] += 1; audit[family]["id_matches"] += 1
            audit[family]["special_spans"] += sum(span.special for span in spans[family])
        shared = build_anchors(row["prefix_text"], spans["llama"], spans["qwen"], scores["llama"], cfg["regions"])
        qwen_self = self_anchors(row["prefix_text"], spans["qwen"], scores["qwen"], cfg["regions"])
        llama_self = self_anchors(row["prefix_text"], spans["llama"], scores["llama"], cfg["regions"])
        prepared.append({"row": row, "shared_anchors": shared, "qwen_self": qwen_self,
                         "llama_self": llama_self, "scores": scores})
    for family in audit: audit[family]["id_match_rate"] = audit[family]["id_matches"] / audit[family]["rows"]
    save_json(run_root(cfg) / "audit" / "offsets.json", audit)
    records = []
    for family in ("qwen", "llama"):
        seed_all(cfg["seed"] + (family == "llama")); model = load_model(cfg, family)
        try:
            for number, item in enumerate(prepared, 1):
                row = item["row"]
                record = next((x for x in records if x["id"] == row["id"]), None)
                if record is None:
                    shared = item["shared_anchors"]
                    record = {"id": row["id"], "gold_index": row["gold_index"], "shared_anchors": shared,
                              "diagnostics": {"llama_selected_mass": selected_mass(item["scores"]["llama"], [a["source_index"] for a in shared]),
                                              "mapped_qwen_selected_mass": selected_mass(item["scores"]["qwen"], [a["target_index"] for a in shared]),
                                              "qwen_self_selected_mass": selected_mass(item["scores"]["qwen"], [a["source_index"] for a in item["qwen_self"]])},
                              "metrics": {}}
                    records.append(record)
                if family == "qwen":
                    record["metrics"]["qwen_raw_region_self"] = evaluate_memory(
                        cfg, family, model, row, [a["source_index"] for a in item["qwen_self"]])
                    record["metrics"]["llama_driven_mapped_qwen"] = evaluate_memory(
                        cfg, family, model, row, [a["target_index"] for a in item["shared_anchors"]])
                else:
                    record["metrics"]["llama_raw_region_self"] = evaluate_memory(
                        cfg, family, model, row, [a["source_index"] for a in item["llama_self"]])
                if number % 16 == 0: log(f"{family} raw-anchor evaluation: {number}/{len(prepared)}")
        finally:
            del model; torch.cuda.empty_cache()
    root = run_root(cfg) / "results"; root.mkdir(parents=True, exist_ok=True)
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as output:
        for record in records: output.write(json.dumps(record, ensure_ascii=False) + "\n")
    diagnostics = {name: sum(row["diagnostics"][name] for row in records) / len(records)
                   for name in records[0]["diagnostics"]}
    save_json(root / "summary.json", {"experiment": "rawtext_anchor_oracle_llama_to_qwen",
        "no_training": True, "schema": "33 entries [model-native token0,32 raw-text anchors], canonical positions 0..32",
        "offset_audit": audit, "alignment": alignment_summary(records), "attention_mass": diagnostics,
        "functional": {"qwen_old_token_chunk_self": old_token_chunk(cfg, "qwen"),
                       "qwen_raw_region_self": aggregate(records, "qwen_raw_region_self"),
                       "llama_driven_mapped_qwen": aggregate(records, "llama_driven_mapped_qwen"),
                       "llama_old_token_chunk_self": old_token_chunk(cfg, "llama"),
                       "llama_raw_region_self": aggregate(records, "llama_raw_region_self")}})
    log("ALL RAW-TEXT ANCHOR ORACLE EVALUATIONS COMPLETED")
