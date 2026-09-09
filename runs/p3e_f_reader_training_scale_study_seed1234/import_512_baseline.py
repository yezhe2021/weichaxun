import argparse
from pathlib import Path

from p3e_f_common import read_json, read_jsonl, write_json, write_jsonl


MAPPING = {
    "question_only": "question_only",
    "correct_learned_writer16": "correct_canonical",
    "hard_shuffled_learned_writer16": "hard_shuffled_canonical",
    "oracle_support_learned_writer16": "oracle_support_canonical",
    "reader_off": "reader_off",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in read_jsonl(args.source_results):
        if row["condition"] in MAPPING:
            item = dict(row)
            item["condition"] = MAPPING[row["condition"]]
            rows.append(item)
    source = read_json(args.source_summary)
    conditions = {MAPPING[name]: value for name, value in source["conditions"].items() if name in MAPPING}
    summary = {
        "status": "complete", "experiment": "P3-E-F reused train512 baseline",
        "train_scale": 512, "validation_samples": 64, "conditions": conditions,
        "correct_shuffled_f1_gap": (
            conditions["correct_canonical"]["f1"] -
            conditions["hard_shuffled_canonical"]["f1"]
        ),
        "prediction_switch_rate": source["prediction_switch_rate"],
        "reader_off_exact_output_consistency": source["reader_off_exact_output_consistency"],
        "reader_gates": source["reader_gates"],
        "reader_canonical_head_usage": source["reader_canonical_head_usage"],
        "reused_without_retraining": True,
        "source_results": args.source_results,
        "source_summary": args.source_summary,
    }
    write_jsonl(output / "per_sample_generation.jsonl", rows)
    write_json(output / "SUCCESS.json", summary)


if __name__ == "__main__":
    main()
