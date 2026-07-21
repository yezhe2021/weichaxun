import argparse
import csv
from pathlib import Path

from p3d_common import read_json, write_json


def load_if(path): return read_json(path) if Path(path).exists() else {"status": "unavailable"}


def reader_metrics(result, condition="correct"):
    return result.get("conditions", {}).get(condition, {})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); args = parser.parse_args(); root = Path(args.root)
    readers = {name: load_if(root / f"readers/{name}/eval/SUCCESS.json") for name in ("native16", "canonical16", "canonical36")}
    baselines = {
        "q4_question_only": load_if(root / "baselines/q4_question_only/SUCCESS.json"),
        "q4_full_text": load_if(root / "baselines/q4_full_text/SUCCESS.json"),
        "q8_full_text": load_if(root / "baselines/q8_full_text/SUCCESS.json"),
        "old_synthetic_reader": load_if(root / "baselines/old_synthetic_reader/SUCCESS.json"),
    }
    native, canonical = reader_metrics(readers["native16"]), reader_metrics(readers["canonical16"])
    native_zero, canonical_zero = reader_metrics(readers["native16"], "zero"), reader_metrics(readers["canonical16"], "zero")
    denominator = native.get("f1", 0.0) - native_zero.get("f1", 0.0)
    retention = (canonical.get("f1", 0.0) - canonical_zero.get("f1", 0.0)) / denominator if abs(denominator) > 1e-8 else None
    native_success = native.get("f1", 0.0) >= native_zero.get("f1", 0.0) + 0.10 and native.get("f1", 0.0) >= reader_metrics(readers["native16"], "shuffled").get("f1", 0.0) + 0.10
    canonical_success = canonical.get("f1", 0.0) >= canonical_zero.get("f1", 0.0) + 0.10 and canonical.get("f1", 0.0) >= reader_metrics(readers["canonical16"], "shuffled").get("f1", 0.0) + 0.10
    if native_success and not canonical_success: verdict = "canonical_writer_mapping_bottleneck"
    elif not native_success: verdict = "real_reader_injection_or_training_bottleneck"
    elif canonical_success: verdict = "canonical_reader_onboarding_passed"
    else: verdict = "mixed_or_inconclusive"
    result = {"status": "complete", "experiment": "P3-D Frozen Multi-Layer Canonical Writer to Qwen3-4B Real Reader Onboarding", "verdict": verdict, "canonical_uniform16_relative_native_retention": retention, "readers": readers, "baselines": baselines}
    write_json(root / "SUCCESS.json", result)
    rows = []
    for name, value in readers.items():
        for condition, metrics in value.get("conditions", {}).items(): rows.append({"method": name, "condition": condition, "n": metrics.get("n"), "em": metrics.get("exact_match"), "f1": metrics.get("f1"), "rejection_accuracy": metrics.get("rejection_accuracy")})
    for name, value in baselines.items():
        metrics = value.get("metrics", {}); rows.append({"method": name, "condition": value.get("condition", "correct"), "n": metrics.get("n"), "em": metrics.get("exact_match"), "f1": metrics.get("f1"), "rejection_accuracy": None})
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "condition", "n", "em", "f1", "rejection_accuracy")); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
