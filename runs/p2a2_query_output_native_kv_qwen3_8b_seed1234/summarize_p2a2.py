import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Combine the three P2-A2 Reader configurations")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for name in ("output_only", "query_only", "query_output"):
        with open(root / name / "SUCCESS.json", encoding="utf-8") as handle:
            result = json.load(handle)
        condition_map = {row["condition"]: row for row in result["conditions"]}
        row = {
            "config_name": name,
            "query_rank": result["query_rank"],
            "output_rank": result["output_rank"],
            "kv_paired_consistency": result["kv_paired_consistency"],
            "full_text_paired_consistency": result["full_text_prefilled_final_paired_consistency"],
        }
        for condition, values in condition_map.items():
            row[f"{condition}_target_em"] = values["target_em"]
            row[f"{condition}_eos_rate"] = values["eos_rate"]
        rows.append(row)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "comparison.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", "configurations": rows}, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
