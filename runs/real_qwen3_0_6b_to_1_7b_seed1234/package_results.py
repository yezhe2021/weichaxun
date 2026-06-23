import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_NAMES = ["mse_only", "ce_only", "mse_ce", "mse_then_ce"]
EVAL_NAMES = TRAIN_NAMES


def require_file(path, min_bytes=1):
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.stat().st_size < min_bytes:
        raise ValueError(f"File is empty or too small: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
    }


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    checked = []
    for name in ["README.md", "run_all.sh", "real_kv_common.py", "real_kv_translator.py", "train_real_kv_translator.py", "eval_real_kv_translation.py"]:
        checked.append(require_file(ROOT / name))
    for name in TRAIN_NAMES:
        checked.append(require_file(ROOT / "train" / name / "metadata.json"))
        checked.append(require_file(ROOT / "train" / name / "train_history.jsonl"))
        checked.append(require_file(ROOT / "train" / name / "checkpoint_epoch1.pt", min_bytes=1024))
    for name in EVAL_NAMES:
        checked.append(require_file(ROOT / "eval" / name / "summary.json"))
        checked.append(require_file(ROOT / "eval" / name / "diagnostic_table.csv"))
        checked.append(require_file(ROOT / "eval" / name / "per_example.jsonl"))
        checked.append(require_file(ROOT / "eval" / name / "per_layer.jsonl"))
    summary_dir = ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in EVAL_NAMES:
        for row in load_csv(ROOT / "eval" / name / "diagnostic_table.csv"):
            row = dict(row)
            row["training_objective"] = name
            rows.append(row)
    all_rows_path = summary_dir / "all_diagnostic_rows.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with open(all_rows_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    checked.append(require_file(all_rows_path))
    success = {
        "status": "success",
        "experiment": "real_qwen3_0_6b_to_1_7b_context_kv",
        "root": str(ROOT),
        "checked_files": checked,
        "notes": [
            "Main experiment translates context C KV only.",
            "Q and answer teacher forcing are processed natively by the Qwen3-1.7B receiver.",
            "Checkpoint and per-layer files may be kept on server if too large for GitHub.",
        ],
    }
    with open(summary_dir / "SUCCESS.json", "w", encoding="utf-8") as f:
        json.dump(success, f, indent=2, ensure_ascii=False)
    print(json.dumps(success, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
