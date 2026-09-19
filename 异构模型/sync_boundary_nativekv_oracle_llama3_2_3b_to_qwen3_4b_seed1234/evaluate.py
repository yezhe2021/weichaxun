import json
from collections import Counter

import torch

from common import load_qwen, log, run_root, save_json, tokenizer
from data import importance, native, rawanchor, rows
from offsets import token_spans
from protocol import final_logits
from units import build_synchronized_units, select_units, trigger_units


CONDITIONS = (
    "qwen_full", "current_rawanchor", "sync_trigger",
    "sync_max_duplicate", "sync_max_unique", "sync_sum_duplicate", "sync_sum_unique",
    "qwen_self_sync_max_unique",
)


def memory(qwen_native, indices):
    selected = [0] + list(indices)
    if len(selected) != 33: raise RuntimeError("Oracle memory must contain token0 + 32 entries")
    return qwen_native["k"][:, selected].cuda(), qwen_native["v"][:, selected].cuda()


def predict(model, row, key, value):
    logits = final_logits(model, row["encoded"]["qwen"]["suffix"], key, value,
                          positions=torch.arange(33, device="cuda"), suffix_start=33)
    ids = row["encoded"]["qwen"]["choice_ids"]
    return int(logits[ids].argmax()), logits[ids].cpu().tolist()


def selected_mass(score, units, family):
    indices = set()
    for unit in units:
        indices.update(unit.llama_indices if family == "llama" else unit.qwen_indices)
    denominator = score[1:].sum().clamp_min(1e-12)
    return float(score[list(indices)].sum() / denominator)


def unit_statistics(units):
    shapes = Counter()
    for unit in units:
        a, b = len(unit.llama_indices), len(unit.qwen_indices)
        if a == 1 and b == 1: shapes["one_to_one"] += 1
        elif a == 1 or b == 1: shapes["one_to_many"] += 1
        else: shapes["many_to_many"] += 1
    n = len(units)
    return {"count": n,
            "mean_llama_tokens": sum(len(x.llama_indices) for x in units) / n,
            "mean_qwen_tokens": sum(len(x.qwen_indices) for x in units) / n,
            "mean_bytes": sum(x.byte_end - x.byte_start for x in units) / n,
            "mean_characters": sum(x.char_end - x.char_start for x in units) / n,
            **{name: shapes[name] / n for name in ("one_to_one", "one_to_many", "many_to_many")}}


@torch.no_grad()
def run_oracle(cfg):
    toks = {family: tokenizer(path) for family, path in cfg["models"].items()}
    model = load_qwen(cfg)
    totals = {condition: 0 for condition in CONDITIONS}; records = []; audits = []
    root = run_root(cfg); (root / "units").mkdir(parents=True, exist_ok=True)
    (root / "audit").mkdir(parents=True, exist_ok=True)
    units_path = root / "units" / "units.jsonl"
    units_path.write_text("", encoding="utf-8")
    try:
        for number, row in enumerate(rows(cfg), 1):
            text = row["prefix_text"]
            spans = {family: token_spans(toks[family], text, row["encoded"][family]["prefix"])
                     for family in ("llama", "qwen")}
            units = build_synchronized_units(text, spans["llama"], spans["qwen"])
            scores = {family: importance(cfg, family, row) for family in ("llama", "qwen")}
            qnative, old = native(cfg, "qwen", row), rawanchor(cfg, row)
            anchor_chars = [anchor["anchor_char"] for anchor in old["metadata"]["anchors"]]
            selected = {
                "sync_trigger": trigger_units(units, anchor_chars),
                "sync_max_duplicate": select_units(units, scores["llama"], "llama", "max", cfg["regions"], False),
                "sync_max_unique": select_units(units, scores["llama"], "llama", "max", cfg["regions"], True),
                "sync_sum_duplicate": select_units(units, scores["llama"], "llama", "sum", cfg["regions"], False),
                "sync_sum_unique": select_units(units, scores["llama"], "llama", "sum", cfg["regions"], True),
                "qwen_self_sync_max_unique": select_units(units, scores["qwen"], "qwen", "max", cfg["regions"], True),
            }
            gold = row["gold_index"]
            ids = row["encoded"]["qwen"]["choice_ids"]
            predictions = {"qwen_full": int(qnative["native_logits"][ids].argmax())}
            logits = {"qwen_full": qnative["native_logits"][ids].tolist()}
            predictions["current_rawanchor"], logits["current_rawanchor"] = predict(
                model, row, old["target_k"].cuda(), old["target_v"].cuda())
            for condition, chosen in selected.items():
                indices = [unit.qwen_right for unit in chosen]
                predictions[condition], logits[condition] = predict(model, row, *memory(qnative, indices))
            for condition in CONDITIONS: totals[condition] += predictions[condition] == gold
            selection_audit = {}
            for condition, chosen in selected.items():
                selection_audit[condition] = {
                    "unit_ids": [unit.unit_id for unit in chosen],
                    "qwen_right_indices": [unit.qwen_right for unit in chosen],
                    "unique_units": len(set(unit.unit_id for unit in chosen)),
                    "duplicate_entries": 32 - len(set(unit.unit_id for unit in chosen)),
                    "selected_llama_attention_mass": selected_mass(scores["llama"], chosen, "llama"),
                    "selected_qwen_attention_mass": selected_mass(scores["qwen"], chosen, "qwen"),
                }
            unit_stats = unit_statistics(units)
            audits.append({"id": row["id"], "unit_statistics": unit_stats, "selection": selection_audit})
            records.append({"id": row["id"], "gold_index": gold, "predictions": predictions,
                            "correct": {key: value == gold for key, value in predictions.items()},
                            "choice_logits": logits})
            with units_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"id": row["id"], "units": [unit.json() for unit in units]}, ensure_ascii=False) + "\n")
            if number % 16 == 0: log(f"Synchronized Oracle: {number}/{cfg['test_samples']}")
    finally:
        del model; torch.cuda.empty_cache()
    count = cfg["test_samples"]
    accuracy = {condition: totals[condition] / count for condition in CONDITIONS}
    transitions = {}
    for condition in CONDITIONS[2:]:
        transitions[condition] = {
            "raw_wrong_to_sync_correct": sum((not r["correct"]["current_rawanchor"]) and r["correct"][condition] for r in records),
            "raw_correct_to_sync_wrong": sum(r["correct"]["current_rawanchor"] and (not r["correct"][condition]) for r in records),
        }
    summary = {"primary_metric": "accuracy", "sample_count": count, "accuracy": accuracy,
               "correct_counts": totals, "paired_transitions_vs_current_rawanchor": transitions}
    save_json(root / "results" / "summary.json", summary)
    with (root / "results" / "per_sample.jsonl").open("w", encoding="utf-8") as output:
        for record in records: output.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (root / "audit" / "per_sample_alignment.jsonl").open("w", encoding="utf-8") as output:
        for audit in audits: output.write(json.dumps(audit, ensure_ascii=False) + "\n")
    aggregate = {}
    stat_keys = ("count", "mean_llama_tokens", "mean_qwen_tokens", "mean_bytes", "mean_characters",
                 "one_to_one", "one_to_many", "many_to_many")
    for key in stat_keys: aggregate[key] = sum(a["unit_statistics"][key] for a in audits) / count
    aggregate["samples_with_fewer_units_than_budget"] = sum(
        a["unit_statistics"]["count"] < cfg["regions"] for a in audits)
    aggregate["minimum_units_in_sample"] = min(a["unit_statistics"]["count"] for a in audits)
    aggregate["selection"] = {}
    for condition in selected:
        aggregate["selection"][condition] = {
            key: sum(a["selection"][condition][key] for a in audits) / count
            for key in ("unique_units", "duplicate_entries", "selected_llama_attention_mass", "selected_qwen_attention_mass")}
    save_json(root / "audit" / "alignment_audit.json", aggregate)
    log("SYNCHRONIZED BOUNDARY NATIVE-KV ORACLE COMPLETED")
