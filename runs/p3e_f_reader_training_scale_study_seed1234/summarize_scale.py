import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from p3e_f_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary512", required=True)
    parser.add_argument("--summary1024", required=True)
    parser.add_argument("--summary2048", required=True)
    parser.add_argument("--semantic-status", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for scale, path in ((512, args.summary512), (1024, args.summary1024), (2048, args.summary2048)):
        summary = read_json(path)
        conditions = summary["conditions"]
        records.append({
            "train_samples": scale,
            "question_only_em": conditions["question_only"]["em"],
            "question_only_f1": conditions["question_only"]["f1"],
            "correct_em": conditions["correct_canonical"]["em"],
            "correct_f1": conditions["correct_canonical"]["f1"],
            "correct_bridge_f1": conditions["correct_canonical"]["by_type"]["bridge"]["f1"],
            "correct_comparison_f1": conditions["correct_canonical"]["by_type"]["comparison"]["f1"],
            "shuffled_em": conditions["hard_shuffled_canonical"]["em"],
            "shuffled_f1": conditions["hard_shuffled_canonical"]["f1"],
            "oracle_em": conditions["oracle_support_canonical"]["em"],
            "oracle_f1": conditions["oracle_support_canonical"]["f1"],
            "reader_off_em": conditions["reader_off"]["em"],
            "reader_off_f1": conditions["reader_off"]["f1"],
            "correct_shuffled_f1_gap": summary["correct_shuffled_f1_gap"],
            "prediction_switch_rate": summary["prediction_switch_rate"],
            "eos_rate": conditions["correct_canonical"]["eos_rate"],
            "supporting_fact_attention_mass": conditions["correct_canonical"].get(
                "supporting_fact_attention_mass"
            ),
            "manual_strict_semantic_accuracy": None,
            "manual_lenient_semantic_accuracy": None,
            "manual_cpw": None,
        })
    with (output / "scale_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    x = [row["train_samples"] for row in records]
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, [row["correct_f1"] for row in records], marker="o", label="Correct Canonical")
    plt.plot(x, [row["shuffled_f1"] for row in records], marker="o", label="Hard shuffled")
    plt.plot(x, [row["correct_shuffled_f1_gap"] for row in records], marker="o", label="F1 gap")
    plt.xlabel("Reader training samples")
    plt.ylabel("HotpotQA F1")
    plt.xticks(x)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "scale_curve.png", dpi=180)
    plt.close()
    result = {
        "status": "complete", "experiment": "P3-E-F Reader Training Scale Study",
        "historical512_reused": True, "new_training_scales": [1024, 2048],
        "independent_initialization_checkpoint": True,
        "validation_samples": 64, "records": records,
        "manual_semantic_review": read_json(args.semantic_status),
        "interpretation_policy": {
            "increasing_correct_with_low_shuffled": "training-data limited",
            "1024_gain_then_2048_plateau": "approaching current Reader capacity",
            "little_gain_or_correct_and_shuffled_rise_together": "structure/objective/memory bottleneck",
        },
    }
    write_json(output / "scale_summary.json", result)
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
