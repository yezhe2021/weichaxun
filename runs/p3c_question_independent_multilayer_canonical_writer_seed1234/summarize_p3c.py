import argparse
import csv
from pathlib import Path

import numpy as np

from p3b_common import read_json, write_json
from p3c_common import LAYER_CONFIGS, SEEDS


def condition(result, name):
    return next(row for row in result["conditions"] if row["condition"] == name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for layer_config in LAYER_CONFIGS:
        for seed in SEEDS:
            branch = root / "branches" / layer_config / f"seed{seed}"
            result = read_json(branch / "fresh_probe/SUCCESS.json")
            correct, zero, shuffled, mismatch = (condition(result, name) for name in ("correct", "zero", "shuffled", "kv_mismatch"))
            rows.append({
                "layer_config": layer_config, "layers": len(LAYER_CONFIGS[layer_config]), "seed": seed,
                "correct_em": correct["current_answer_em"], "correct_f1": correct["current_answer_f1"],
                "zero_f1": zero["current_answer_f1"], "shuffled_current_f1": shuffled["current_answer_f1"],
                "shuffled_source_f1": shuffled["source_memory_f1"], "mismatch_f1": mismatch["current_answer_f1"],
                "start_accuracy": correct["start_accuracy"], "end_accuracy": correct["end_accuracy"],
                "supporting_sentence_recall": correct["supporting_sentence_recall"], "retention": result["retention"],
                "writer_checkpoint": str(branch / "writer/writer_best.pt"),
            })
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregate = {}
    for layer_config in LAYER_CONFIGS:
        selected = [row for row in rows if row["layer_config"] == layer_config]
        aggregate[layer_config] = {
            key: {"mean": float(np.mean([row[key] for row in selected])), "std": float(np.std([row[key] for row in selected]))}
            for key in ("correct_em", "correct_f1", "zero_f1", "shuffled_current_f1", "shuffled_source_f1", "mismatch_f1", "retention")
        }
    gap = aggregate["all36"]["correct_f1"]["mean"] - aggregate["uniform16"]["correct_f1"]["mean"]
    if gap <= 0.03:
        layer_recommendation = "uniform16_is_stable_efficiency_candidate"
    else:
        layer_recommendation = "retain_all36_as_public_protocol"
    all36 = aggregate["all36"]
    fresh_success = all36["correct_f1"]["mean"] > all36["zero_f1"]["mean"] + 0.10 and all36["correct_f1"]["mean"] > all36["shuffled_current_f1"]["mean"] + 0.05
    summary = {
        "status": "complete", "rows": rows, "aggregate": aggregate,
        "all36_minus_uniform16_f1": gap, "layer_recommendation": layer_recommendation,
        "fresh_probe_interface_passed": fresh_success,
        "interpretation": (
            "Frozen Writer remains independently readable" if fresh_success else
            "Canonical Writer did not establish a probe-independent executable interface"
        ),
        "gate": read_json(root / "gate/SUCCESS.json"),
    }
    write_json(root / "SUCCESS.json", summary)


if __name__ == "__main__":
    main()
