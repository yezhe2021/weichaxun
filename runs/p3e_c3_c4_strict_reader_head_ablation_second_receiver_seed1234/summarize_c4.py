import argparse
import statistics

from p3d3_common import read_json, write_json


def extract(path):
    result = read_json(path); correct = result["conditions"]["correct_canonical16"]
    return {"path": path, "correct_em": correct["em"], "correct_f1": correct["f1"], "bridge_f1": correct.get("by_type", {}).get("bridge", {}).get("f1"),
            "comparison_f1": correct.get("by_type", {}).get("comparison", {}).get("f1"), "correct_shuffled_gap": result["correct_shuffled_f1_gap"],
            "question_only_gain": result["correct_question_only_f1_gain"], "prediction_switch_rate": result["prediction_switch_rate"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--runs", nargs=2, required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); rows = [extract(path) for path in args.runs]; keys = [key for key in rows[0] if key != "path"]
    aggregate = {key: {"mean": statistics.mean(row[key] for row in rows if row[key] is not None), "std": statistics.pstdev(row[key] for row in rows if row[key] is not None),
                       "minimum": min(row[key] for row in rows if row[key] is not None)} for key in keys}
    write_json(args.out, {"status": "complete", "experiment": "C4 second Receiver onboarding with same frozen C2 Writer", "runs": rows, "aggregate": aggregate,
                          "writer_update_count": 0, "receiver": "Qwen3.5-4B eight genuine full-attention layers"})


if __name__ == "__main__": main()
