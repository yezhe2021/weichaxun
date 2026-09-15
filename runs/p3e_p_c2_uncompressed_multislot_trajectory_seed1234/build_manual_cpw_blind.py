import argparse
import csv

from p3d3_common import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    fields = [
        "id",
        "type",
        "condition",
        "question",
        "gold_answer",
        "source_answer",
        "generation",
        "prediction",
        "C_P_W",
        "strict_correct",
        "lenient_correct",
        "notes",
    ]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in load_jsonl(args.records):
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    main()

