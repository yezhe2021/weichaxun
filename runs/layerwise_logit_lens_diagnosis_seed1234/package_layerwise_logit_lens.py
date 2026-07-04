import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Package layerwise logit-lens diagnosis summaries")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "summary_all"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary in sorted(root.glob("*/layerwise_logit_lens_summary.csv")):
        rows.extend(read_csv(summary))
    write_csv(out / "layerwise_logit_lens_summary_all.csv", rows)
    payload = {
        "status": "complete",
        "root": str(root),
        "summary_rows": len(rows),
        "summary_file": "layerwise_logit_lens_summary_all.csv",
    }
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
