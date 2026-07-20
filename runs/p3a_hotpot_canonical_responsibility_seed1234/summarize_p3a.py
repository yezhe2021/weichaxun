import argparse
import json
from pathlib import Path

from p3a_common import write_json


def condition(result, name): return next(row for row in result["conditions"] if row["condition"] == name)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); args = parser.parse_args(); root = Path(args.root)
    paths = {
        "q4_question_only": root / "baselines/q4_question_only/SUCCESS.json", "q4_full_text": root / "baselines/q4_full_text/SUCCESS.json",
        "q8_full_text": root / "baselines/q8_full_text/SUCCESS.json", "old_synthetic_reader": root / "readers/old_synthetic/eval/SUCCESS.json",
        "new_hotpot_reader": root / "readers/hotpot/full/eval/SUCCESS.json",
        **{f"probe_{source}": root / f"probes/{source}/eval/SUCCESS.json" for source in ("hidden", "raw_kv", "pca_kv", "canonical")},
    }
    results = {}
    for name, path in paths.items():
        with path.open(encoding="utf-8") as handle: results[name] = json.load(handle)
    probe = {source: condition(results[f"probe_{source}"], "correct")["f1"] for source in ("hidden", "raw_kv", "pca_kv", "canonical")}
    old = condition(results["old_synthetic_reader"], "correct")["f1"]; new = condition(results["new_hotpot_reader"], "correct")["f1"]
    qonly = condition(results["q4_question_only"], "question_only")["f1"]
    zero = condition(results["new_hotpot_reader"], "zero")["f1"]
    raw_high = max(probe["hidden"], probe["raw_kv"]) >= 0.30; canonical_high = probe["canonical"] >= 0.30
    new_success = new >= qonly + 0.10 and new >= zero + 0.10
    if raw_high and probe["canonical"] + 0.15 < max(probe["hidden"], probe["raw_kv"]): verdict = "writer_cross_task_information_loss"
    elif canonical_high and new_success and old + 0.10 < new: verdict = "legacy_reader_synthetic_task_overfit"
    elif canonical_high and not new_success: verdict = "reader_injection_or_open_answer_execution_problem"
    elif not raw_high and not canonical_high: verdict = "last_layer_states_not_readable_by_controlled_probe"
    else: verdict = "mixed_or_inconclusive"
    write_json(root / "SUCCESS.json", {"status": "complete", "experiment": "P3-A HotpotQA Canonical Evidence-KV Responsibility Decomposition", "samples": {"train": 512, "dev": 500}, "verdict": verdict, "probe_correct_f1": probe, "old_reader_correct_f1": old, "new_reader_correct_f1": new, "results": results})


if __name__ == "__main__": main()
