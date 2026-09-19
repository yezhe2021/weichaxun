import argparse
import csv
import random
from pathlib import Path

from p3e_f_common import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval512", required=True)
    parser.add_argument("--eval1024", required=True)
    parser.add_argument("--eval2048", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for scale, path in ((512, args.eval512), (1024, args.eval1024), (2048, args.eval2048)):
        for row in read_jsonl(path):
            if row["condition"] not in {"correct_canonical", "hard_shuffled_canonical"}:
                continue
            source_rows.append({
                "sample_id": row["id"], "type": row["type"], "gold_answer": row["answer"],
                "prediction": row["output"]["prediction"], "raw_generation": row["output"]["text"],
                "_scale": scale, "_condition": row["condition"],
                "C_P_W": "", "strict_semantic_correct": "", "lenient_semantic_correct": "",
                "review_notes": "",
            })
    random.Random(args.seed).shuffle(source_rows)
    rows, key = [], []
    for index, source in enumerate(source_rows):
        blind_id = f"B{index:04d}"
        key.append({"blind_id": blind_id, "train_scale": source.pop("_scale"),
                    "condition": source.pop("_condition"), "sample_id": source["sample_id"]})
        rows.append({"blind_id": blind_id, **source})
    fieldnames = list(rows[0])
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output.with_name("semantic_review_key.json"), {"seed": args.seed, "mapping": key})
    write_json(output.with_suffix(".json"), {
        "status": "pending_manual_review", "rows": len(rows),
        "labels": {"C": "correct", "P": "partially/semantically acceptable", "W": "wrong"},
        "strict_rule": "C only", "lenient_rule": "C or P",
        "note": "Automatic experiment completion does not fabricate manual C/P/W labels.",
    })


if __name__ == "__main__":
    main()
