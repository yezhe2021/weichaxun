import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ("mse_then_ce", "mse_only")


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
    stage1_rows = []
    stage2_rows = []
    manifests = {}
    completed = []
    for label in LABELS:
        result_dir = ROOT / "results" / label
        summary_path = result_dir / "summary.json"
        if not summary_path.is_file():
            continue
        payload = load_json(summary_path)
        manifests[label] = load_json(result_dir / "checkpoint_manifest.json")
        completed.append(label)
        for row in payload["stage1_cache_swap_summary"]:
            stage1_rows.append({"checkpoint": label, **row})
        for row in payload["stage2_readout_probe_summary"]:
            stage2_rows.append({"checkpoint": label, **row})

    out = ROOT / "summary"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stage1_checkpoint_comparison.csv", stage1_rows)
    write_csv(out / "stage2_checkpoint_comparison.csv", stage2_rows)
    with open(out / "checkpoint_manifests.json", "w", encoding="utf-8") as handle:
        json.dump(manifests, handle, indent=2, ensure_ascii=False)
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "completed_checkpoints": completed,
                "stage1_conditions": [
                    "native",
                    "translated",
                    "native_k_translated_v",
                    "translated_k_native_v",
                ],
                "stage2_conditions": [
                    "native",
                    "translated",
                    "native_k_translated_v",
                    "translated_k_native_v",
                ],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
