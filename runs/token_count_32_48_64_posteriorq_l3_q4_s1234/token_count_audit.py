from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def variable_length_translation(module, source_k, source_v):
    """Run the unchanged token-independent maps without the legacy T=32 guard."""
    if source_k.shape != source_v.shape or source_k.ndim != 4:
        raise RuntimeError(f"Bad source shapes: {source_k.shape}, {source_v.shape}")
    if source_k.shape[0] != module.source_layers or source_k.shape[2:] != (module.heads, module.dim):
        raise RuntimeError(f"Translator architecture mismatch: {source_k.shape}")
    key = source_k.cuda()[None]
    value = source_v.cuda()[None]
    with torch.amp.autocast("cuda", dtype=torch.float16):
        predicted_k = module._heads(module._depth(key, module.k_depth), module.k_head)
        predicted_v = module._heads(module._depth(value, module.v_depth), module.v_head)
    return predicted_k[0], predicted_v[0]


def average(records, count, condition, metric):
    return sum(row["counts"][str(count)][condition][metric] for row in records) / len(records)


@torch.no_grad()
def run(limit=None):
    audit_cfg = read_json(ROOT / "config.json")
    baseline_root = Path(audit_cfg["baseline_experiment_root"])
    sys.path.insert(0, str(baseline_root))

    from anchors import build_anchors, selected_mass
    from common import load_model, seed_all, tokenizer
    from data import load_pair, load_source, manifest_rows
    from experiment import choice_kl, load_checkpoint, receiver_logits
    from offsets import token_spans
    from protocol import capture_decision

    cfg = read_json(baseline_root / "runs" / "study" / "run_config.json")
    if cfg.get("protocol") != "sender_question_options_posterior_question_router_v1":
        raise RuntimeError(f"Unexpected baseline protocol: {cfg.get('protocol')}")
    seed_all(audit_cfg["seed"])
    counts = [int(value) for value in audit_cfg["token_counts"]]
    rows = manifest_rows(cfg, "test")
    if limit is not None:
        rows = rows[:limit]
    output_root = ROOT / "runs" / ("smoke" if limit else "study")
    cache_root = output_root / "cache" / "source"
    save_json(output_root / "status.json", {"status": "running", "stage": "cache_llama"})

    toks = {family: tokenizer(path) for family, path in cfg["models"].items()}
    llama = load_model(cfg, "llama")
    try:
        for number, row in enumerate(rows, 1):
            fields = row["encoded"]["llama"]
            full_k, full_v, observed_importance, _ = capture_decision(
                llama, fields["sender_full"], len(fields["body"]), fields["choice_ids"]
            )
            baseline_source = load_source(cfg, "test", row)
            importance = baseline_source["importance"]
            max_delta = float((observed_importance - importance).abs().max())
            if max_delta > 2e-3:
                raise RuntimeError(f"Routing replay drift for {row['id']}: {max_delta}")
            spans = {
                family: token_spans(toks[family], row["body"], row["encoded"][family]["body"])
                for family in ("llama", "qwen")
            }
            payload = {"id": row["id"], "routing_replay_max_delta": max_delta, "counts": {}}
            for count in counts:
                anchors = build_anchors(
                    row["body"], spans["llama"], spans["qwen"], importance, count,
                    *row["options_char_span"]
                )
                source_indices = [anchor["source_index"] for anchor in anchors]
                target_indices = [anchor["target_index"] for anchor in anchors]
                payload["counts"][str(count)] = {
                    "source_k": full_k[:, source_indices].contiguous(),
                    "source_v": full_v[:, source_indices].contiguous(),
                    "source_indices": source_indices,
                    "target_indices": target_indices,
                    "anchors": anchors,
                    "unique_source_tokens": len(set(source_indices)),
                    "unique_target_tokens": len(set(target_indices)),
                    "selected_attention_mass": selected_mass(
                        importance, source_indices, fields["option_token_indices"]
                    ),
                }
            cache_root.mkdir(parents=True, exist_ok=True)
            torch.save(payload, cache_root / f"{row['id']}.pt")
            if number % 16 == 0 or number == len(rows):
                log(f"Llama scalable source cache: {number}/{len(rows)}")
    finally:
        del llama
        torch.cuda.empty_cache()

    save_json(output_root / "status.json", {"status": "running", "stage": "evaluate"})
    stage_b = read_json(baseline_root / "runs" / "study" / "stage_b" / "training_summary.json")
    candidate = stage_b[audit_cfg["checkpoint_alias"]]
    module, checkpoint = load_checkpoint(cfg, candidate["path"])
    module.eval()
    qwen = load_model(cfg, "qwen")
    records = []
    try:
        for number, row in enumerate(rows, 1):
            fields = row["encoded"]["qwen"]
            full_k, full_v, _, _ = capture_decision(
                qwen, fields["sender_full"], len(fields["body"]), fields["choice_ids"]
            )
            pair = load_pair(cfg, "test", row)
            scalable = torch.load(cache_root / f"{row['id']}.pt", map_location="cpu", weights_only=True)
            choice_ids = fields["choice_ids"]
            choice_tensor = torch.tensor(choice_ids, device="cuda", dtype=torch.long)
            gold = row["gold_index"]
            record = {"id": row["id"], "gold_index": gold, "counts": {}}
            for count in counts:
                item = scalable["counts"][str(count)]
                target_indices = item["target_indices"]
                native = (
                    full_k[:, target_indices].cuda(),
                    full_v[:, target_indices].cuda(),
                )
                translated = variable_length_translation(module, item["source_k"], item["source_v"])
                logits = {}
                for condition, content in (("native_oracle", native), ("translated", translated)):
                    memory_k = torch.cat((pair["question_k"].cuda(), content[0]), 1)
                    memory_v = torch.cat((pair["question_v"].cuda(), content[1]), 1)
                    logits[condition] = receiver_logits(qwen, row, memory_k, memory_v)
                teacher_choice = logits["native_oracle"][choice_tensor].detach()
                oracle_prediction = int(teacher_choice.argmax())
                result = {
                    "unique_source_tokens": item["unique_source_tokens"],
                    "unique_target_tokens": item["unique_target_tokens"],
                    "selected_attention_mass": item["selected_attention_mass"],
                }
                for condition in ("native_oracle", "translated"):
                    choice = logits[condition][choice_tensor]
                    prediction = int(choice.argmax())
                    result[condition] = {
                        "prediction": prediction,
                        "accuracy": float(prediction == gold),
                        "oracle_agreement": float(prediction == oracle_prediction),
                        "oracle_choice_kl": float(
                            choice_kl(logits[condition], teacher_choice.cpu(), choice_ids, cfg["temperature"]).item()
                        ),
                        "choice_logits": choice.cpu().tolist(),
                    }
                record["counts"][str(count)] = result
            records.append(record)
            if number % 16 == 0 or number == len(rows):
                log(f"Token-count evaluation: {number}/{len(rows)}")
    finally:
        del qwen, module
        torch.cuda.empty_cache()

    metrics = {}
    for count in counts:
        metrics[str(count)] = {
            condition: {
                metric: average(records, count, condition, metric)
                for metric in ("accuracy", "oracle_agreement", "oracle_choice_kl")
            }
            for condition in ("native_oracle", "translated")
        }
        metrics[str(count)]["selection"] = {
            "mean_unique_source_tokens": sum(
                row["counts"][str(count)]["unique_source_tokens"] for row in records
            ) / len(records),
            "mean_unique_target_tokens": sum(
                row["counts"][str(count)]["unique_target_tokens"] for row in records
            ) / len(records),
            "mean_selected_attention_mass": sum(
                row["counts"][str(count)]["selected_attention_mass"] for row in records
            ) / len(records),
        }

    evaluation_root = output_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    with (evaluation_root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    save_json(evaluation_root / "summary.json", metrics)

    baseline_results = read_json(baseline_root / "runs" / "study" / "results" / "comparison.json")
    comparison = {
        "experiment": ROOT.name,
        "status": "completed",
        "samples": len(rows),
        "checkpoint": {"alias": audit_cfg["checkpoint_alias"], "step": checkpoint["step"]},
        "purpose": "No-training scaling audit from 32 to 48/64 selected option KV tokens.",
        "invariants": {
            "translator_parameters": "unchanged posterior-Question Full28-Diagonal checkpoint",
            "translator_operation": "same per-token maps; only legacy T=32 shape guard bypassed",
            "sender_routing_query": "last token of repeated posterior Question",
            "receiver": "native Question + external selected Options KV + native Answer:",
            "selection": "M normalized option-text regions; max routing attention per region",
        },
        "baseline_32": {
            "native_oracle": baseline_results["main_table"]["Llama-selected -> Qwen Native Oracle"],
            "translated": baseline_results["main_table"]["Translated Full28-Diagonal"],
        },
        "scaled": metrics,
    }
    save_json(output_root / "results" / "comparison.json", comparison)
    save_json(output_root / "status.json", {"status": "completed", "stage": "all"})
    log("TOKEN-COUNT SCALING AUDIT COMPLETED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.limit)


if __name__ == "__main__":
    main()
