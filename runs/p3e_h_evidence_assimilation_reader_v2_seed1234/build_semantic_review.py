import argparse
import csv
import random
from pathlib import Path

from p3e_f_common import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    source = []
    for row in read_jsonl(args.results):
        if row["condition"] not in {"current_c1_reader", "assimilation_reader_v2",
                                    "hard_shuffled_assimilation"}:
            continue
        source.append({
            "sample_id": row["id"], "type": row["type"], "gold_answer": row["answer"],
            "prediction": row["output"]["prediction"], "raw_generation": row["output"]["text"],
            "_condition": row["condition"], "C_P_W": "", "strict_semantic_correct": "",
            "lenient_semantic_correct": "", "review_notes": "",
        })
    random.Random(args.seed).shuffle(source)
    rows, key = [], []
    for index, row in enumerate(source):
        blind_id = f"H{index:04d}"
        condition = row.pop("_condition")
        rows.append({"blind_id": blind_id, **row})
        key.append({"blind_id": blind_id, "sample_id": row["sample_id"], "condition": condition})
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output.with_name("semantic_review_key.json"), {"seed": args.seed, "mapping": key})
    write_json(output.with_suffix(".json"), {
        "status": "pending_manual_review", "rows": len(rows),
        "strict_semantic_accuracy": None, "lenient_semantic_accuracy": None,
        "C_P_W_counts": None,
    })


if __name__ == "__main__":
    main()
