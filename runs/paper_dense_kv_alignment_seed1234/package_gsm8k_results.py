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


def method_group(method):
    if method.startswith("paper_"):
        return "paper"
    if method == "q_aware_functional":
        return "ours"
    return "baseline"


def main():
    comparison = []
    completed = []
    for method in METHODS:
        eval_summary = ROOT / "eval_gsm8k" / method / "summary.json"
        if not eval_summary.is_file():
            continue
        payload = load_json(eval_summary)
        completed.append(method)
        for row in payload["diagnostic_table"]:
            comparison.append({"method_group": method_group(method), **row})
    out = ROOT / "summary_gsm8k"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "method_comparison.csv", comparison)
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "dataset": "gsm8k",
                "completed_methods": completed,
                "required_methods": list(METHODS),
                "paper_method": "paper_rec_then_mixed_generation",
                "baselines": ["mse_only", "mse_then_ce"],
                "ours": ["q_aware_functional"],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
