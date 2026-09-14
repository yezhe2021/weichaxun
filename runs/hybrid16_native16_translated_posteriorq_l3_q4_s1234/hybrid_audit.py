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


def mean_metrics(records, condition):
    keys = ("accuracy", "oracle_agreement", "oracle_choice_kl")
    return {
        key: sum(row["conditions"][condition][key] for row in records) / len(records)
        for key in keys
    }


def hybrid_content(translated, native, native_indices):
    translated_k, translated_v = translated
    native_k, native_v = native
    if translated_k.shape != native_k.shape or translated_v.shape != native_v.shape:
        raise RuntimeError(
            f"Hybrid shape mismatch: translated={translated_k.shape}/{translated_v.shape}, "
            f"native={native_k.shape}/{native_v.shape}"
        )
    if translated_k.shape[1] != 32:
        raise RuntimeError(f"Expected exactly 32 selected KV positions, got {translated_k.shape[1]}")
    index = torch.tensor(native_indices, device=translated_k.device, dtype=torch.long)
    key, value = translated_k.clone(), translated_v.clone()
    key[:, index] = native_k[:, index]
    value[:, index] = native_v[:, index]
    return key, value


@torch.no_grad()
def run():
    audit_cfg = read_json(ROOT / "config.json")
    baseline_root = Path(audit_cfg["baseline_experiment_root"])
    sys.path.insert(0, str(baseline_root))

    from common import load_model, seed_all
    from data import load_pair, load_source, manifest_rows
    from experiment import choice_kl, load_checkpoint, load_teacher, student_logits, translated_content

    cfg = read_json(baseline_root / "runs" / "study" / "run_config.json")
    if cfg.get("protocol") != "sender_question_options_posterior_question_router_v1":
        raise RuntimeError(f"Unexpected baseline protocol: {cfg.get('protocol')}")
    seed_all(audit_cfg["seed"])
    output_root = ROOT / "runs" / "study"
    save_json(output_root / "status.json", {"status": "running", "stage": "evaluate"})

    stage_b = read_json(baseline_root / "runs" / "study" / "stage_b" / "training_summary.json")
    baseline_results = read_json(baseline_root / "runs" / "study" / "results" / "comparison.json")
    rows = manifest_rows(cfg, "test")
    model = load_model(cfg, "qwen")
    all_results = {}
    try:
        for alias in audit_cfg["checkpoint_aliases"]:
            candidate = stage_b[alias]
            module, payload = load_checkpoint(cfg, candidate["path"])
            module.eval()
            records = []
            for number, row in enumerate(rows, 1):
                source = load_source(cfg, "test", row)
                pair = load_pair(cfg, "test", row)
                teacher = load_teacher(cfg, "test", row)
                translated = translated_content(module, source)
                native = (
                    pair["target_k"][:, 1:].cuda(),
                    pair["target_v"][:, 1:].cuda(),
                )
                choice_ids = row["encoded"]["qwen"]["choice_ids"]
                ids = torch.tensor(choice_ids, device="cuda", dtype=torch.long)
                gold = row["gold_index"]
                oracle_prediction = teacher["prediction"]
                record = {
                    "id": row["id"],
                    "gold_index": gold,
                    "oracle_prediction": oracle_prediction,
                    "conditions": {},
                }
                for name, pattern in audit_cfg["patterns"].items():
                    content = hybrid_content(translated, native, pattern["native_indices"])
                    logits = student_logits(model, row, pair, content)
                    choice_logits = logits[ids]
                    prediction = int(choice_logits.argmax())
                    record["conditions"][name] = {
                        "prediction": prediction,
                        "accuracy": float(prediction == gold),
                        "oracle_agreement": float(prediction == oracle_prediction),
                        "oracle_choice_kl": float(
                            choice_kl(logits, teacher["choice_logits"], choice_ids, cfg["temperature"]).item()
                        ),
                        "choice_logits": choice_logits.cpu().tolist(),
                    }
                records.append(record)
                if number % 16 == 0 or number == len(rows):
                    log(f"{alias}: {number}/{len(rows)}")
            metrics = {name: mean_metrics(records, name) for name in audit_cfg["patterns"]}
            alias_root = output_root / "evaluation" / alias
            alias_root.mkdir(parents=True, exist_ok=True)
            with (alias_root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            save_json(alias_root / "summary.json", {
                "checkpoint_step": payload["step"],
                "checkpoint_path": candidate["path"],
                "samples": len(rows),
                "conditions": metrics,
            })
            all_results[alias] = {
                "checkpoint_step": payload["step"],
                "conditions": metrics,
            }
            del module
            torch.cuda.empty_cache()
    finally:
        del model
        torch.cuda.empty_cache()

    comparison = {
        "experiment": ROOT.name,
        "status": "completed",
        "purpose": "Replace exactly 16 of 32 translated option KV positions with position-matched native Qwen KV.",
        "protocol_controls": {
            "selected_anchors": "unchanged from posterior-Question Sender baseline",
            "receiver": "native Question + 32 external Options KV + native Answer:",
            "cache_length": 32,
            "position_ids": "unchanged",
            "mask": "unchanged",
            "kv_replacement": "K and V replaced together at the same memory position",
        },
        "patterns": audit_cfg["patterns"],
        "existing_controls": {
            "full_translated_best_accuracy": baseline_results["main_table"]["Translated Full28-Diagonal"],
            "full_native_oracle": baseline_results["main_table"]["Llama-selected -> Qwen Native Oracle"],
            "zero": baseline_results["main_table"]["Zero"],
            "no_memory": baseline_results["main_table"]["No-memory"],
        },
        "hybrid_results": all_results,
    }
    save_json(output_root / "results" / "comparison.json", comparison)
    save_json(output_root / "status.json", {"status": "completed", "stage": "evaluate"})
    log("HYBRID 16-NATIVE / 16-TRANSLATED AUDIT COMPLETED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", default="evaluate", choices=("evaluate",))
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
