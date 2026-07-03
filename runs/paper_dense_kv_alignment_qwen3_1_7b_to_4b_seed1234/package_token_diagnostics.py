import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ("paper_rec_then_mixed_generation", "mse_only", "mse_then_ce", "q_aware_functional")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_for(method):
    if method.startswith("paper_"):
        return "paper"
    if method == "q_aware_functional":
        return "ours"
    return "baseline"


def main():
    parser = argparse.ArgumentParser(description="Package token-level diagnostic summaries")
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    rows = []
    completed = []
    for method in METHODS:
        summary_path = input_root / method / "summary.json"
        if not summary_path.is_file():
            continue
        completed.append(method)
        payload = load_json(summary_path)
        for row in payload["diagnostic_table"]:
            rows.append(
                {
                    "dataset": args.dataset_label,
                    "method_group": group_for(method),
                    **row,
                }
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "method_comparison.csv", rows)
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "dataset": args.dataset_label,
                "completed_methods": completed,
                "required_methods": list(METHODS),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
