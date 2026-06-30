import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGIMES = ("mse_only", "mse_then_ce", "q_aware_functional")


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


def main():
    comparison = []
    train_metadata = {}
    completed = []
    for regime in REGIMES:
        eval_summary = ROOT / "eval" / regime / "summary.json"
        train_meta = ROOT / "train" / regime / "metadata.json"
        if not eval_summary.is_file():
            continue
        payload = load_json(eval_summary)
        completed.append(regime)
        if train_meta.is_file():
            train_metadata[regime] = load_json(train_meta)
        for row in payload["diagnostic_table"]:
            comparison.append({"regime": regime, **row})
    out = ROOT / "summary"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "regime_comparison.csv", comparison)
    with open(out / "train_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(train_metadata, handle, indent=2, ensure_ascii=False)
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "completed_regimes": completed,
                "required_regimes": list(REGIMES),
                "primary_comparison": "q_aware_functional vs mse_then_ce",
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
