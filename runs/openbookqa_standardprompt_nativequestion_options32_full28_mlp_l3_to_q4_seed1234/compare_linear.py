import json
import math
from pathlib import Path

from common import ROOT, read_json, save_json


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mcnemar(first_only, second_only):
    discordant = first_only + second_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def compare(cfg):
    root = ROOT / "runs" / cfg["mode"] / "results"
    mlp_path = ROOT / "runs" / cfg["mode"] / "evaluation" / "translated" / "best_accuracy" / "per_sample_metrics.jsonl"
    linear_path = Path(cfg["linear_baseline_per_sample"])
    if cfg["mode"] != "study" or not linear_path.exists():
        return None
    mlp = {row["id"]: row for row in read_jsonl(mlp_path)}
    linear = {row["id"]: row for row in read_jsonl(linear_path)}
    if set(mlp) != set(linear):
        raise RuntimeError("Linear and MLP test IDs differ")
    ids = list(mlp)
    both = only_linear = only_mlp = neither = 0
    records = []
    for sample_id in ids:
        a = bool(linear[sample_id]["conditions"]["old_translated"]["correct"])
        b = bool(mlp[sample_id]["conditions"]["translated"]["accuracy"])
        both += a and b; only_linear += a and not b; only_mlp += b and not a; neither += not a and not b
        records.append({"id": sample_id, "linear_correct": a, "mlp_correct": b,
                        "linear_prediction": linear[sample_id]["conditions"]["old_translated"]["prediction"],
                        "mlp_prediction": mlp[sample_id]["conditions"]["translated"]["prediction"]})
    result = {"sample_count": len(ids), "linear_accuracy": (both + only_linear) / len(ids),
              "mlp_accuracy": (both + only_mlp) / len(ids),
              "mlp_minus_linear": (only_mlp - only_linear) / len(ids),
              "both_correct": both, "only_linear_correct": only_linear,
              "only_mlp_correct": only_mlp, "both_wrong": neither,
              "mcnemar_exact_two_sided_p": mcnemar(only_linear, only_mlp),
              "only_linear_ids": [row["id"] for row in records if row["linear_correct"] and not row["mlp_correct"]],
              "only_mlp_ids": [row["id"] for row in records if row["mlp_correct"] and not row["linear_correct"]]}
    save_json(root / "linear_vs_mlp_paired.json", result)
    with (root / "linear_vs_mlp_per_sample.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    comparison_path = root / "comparison.json"
    comparison = read_json(comparison_path)
    comparison["linear_vs_mlp_paired"] = result
    save_json(comparison_path, comparison)
    return result


if __name__ == "__main__":
    from common import configuration
    compare(configuration("study"))
