import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = ("fp16_fp16", "int4_fp16", "fp16_int4", "int4_int4")


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten(prefix, payload):
    return {f"{prefix}{key}": value for key, value in payload.items()}


def main():
    summary_dir = ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    representation_rows = []
    translation_rows = []
    for group in GROUPS:
        representation = load(ROOT / "representation" / group / "summary.json")
        representation_rows.append({"group": group, **representation["summary"]})
        for regime in ("mse_then_ce", "ce_only_small"):
            evaluation = load(ROOT / "translation" / group / regime / "eval" / "summary.json")
            translation_rows.append(
                {
                    "group": group,
                    "regime": regime,
                    **evaluation["summary"],
                    "receiver_reference": "same_group_receiver_native",
                }
            )
    for filename, rows in (
        ("representation_comparison.csv", representation_rows),
        ("translation_comparison.csv", translation_rows),
    ):
        fields = sorted({key for row in rows for key in row})
        with open(summary_dir / filename, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    drift = {
        label: load(ROOT / "same_model_drift" / label / "summary.json")
        for label in ("qwen3_0_6b", "qwen3_1_7b")
    }
    with open(summary_dir / "same_model_drift.json", "w", encoding="utf-8") as handle:
        json.dump(drift, handle, indent=2, ensure_ascii=False)
    with open(summary_dir / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "groups": list(GROUPS),
                "main_regime": "mse_then_ce",
                "auxiliary_regime": "ce_only_small",
                "cross_receiver_absolute_ce_comparison": "forbidden",
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
