import argparse
import statistics

from p3d3_common import read_json, write_json


def extract(path):
    result = read_json(path); correct = result["conditions"]["correct_learned_writer16"]
    return {"path": path, "correct_em": correct["em"], "correct_f1": correct["f1"],
            "bridge_f1": correct.get("by_type", {}).get("bridge", {}).get("f1"),
            "comparison_f1": correct.get("by_type", {}).get("comparison", {}).get("f1"),
            "correct_shuffled_gap": result["correct_shuffled_f1_gap"], "prediction_switch_rate": result["prediction_switch_rate"],
            "reader_off_consistency": result["reader_off_exact_output_consistency"]}


def aggregate(rows):
    keys = ["correct_em", "correct_f1", "bridge_f1", "comparison_f1", "correct_shuffled_gap", "prediction_switch_rate", "reader_off_consistency"]
    return {key: {"mean": statistics.mean(row[key] for row in rows if row[key] is not None),
                  "std": statistics.pstdev(row[key] for row in rows if row[key] is not None),
                  "minimum": min(row[key] for row in rows if row[key] is not None)} for key in keys}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--random", nargs=2, required=True); parser.add_argument("--weak", nargs=2, required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); random_rows, weak_rows = [extract(path) for path in args.random], [extract(path) for path in args.weak]
    write_json(args.out, {"status": "complete", "experiment": "C3-A strict fresh Reader initialization audit",
        "fully_random": {"runs": random_rows, "aggregate": aggregate(random_rows)}, "weak_pair": {"runs": weak_rows, "aggregate": aggregate(weak_rows)},
        "claim_rule": "Public learnable onboarding is judged from both fully-random seeds; weak-pair runs are an optimization-prior ablation."})


if __name__ == "__main__": main()
