import csv
import json
from pathlib import Path


ROOT = Path("runs/formal_qwen3_1_7b_seed1234")
NAMES = ["autoencoder", "mse_only", "ce_only", "mse_ce", "rope_mse_ce"]


def main():
    expected = [
        ROOT / "run_all.sh",
        ROOT / "controls" / "summary.json",
        ROOT / "controls" / "diagnostic_table.csv",
        ROOT / "controls" / "per_example.jsonl",
        ROOT / "controls" / "per_layer.jsonl",
    ]
    for name in NAMES:
        expected.extend([
            ROOT / "train" / name / "checkpoint_epoch1.pt",
            ROOT / "train" / name / "metadata.json",
            ROOT / "train" / name / "train_history.jsonl",
            ROOT / "eval" / name / "summary.json",
            ROOT / "eval" / name / "diagnostic_table.csv",
            ROOT / "eval" / name / "per_example.jsonl",
            ROOT / "eval" / name / "per_layer.jsonl",
        ])
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing or empty files:\n" + "\n".join(missing))

    rows = []
    controls = json.loads((ROOT / "controls" / "summary.json").read_text(encoding="utf-8"))
    for row in controls["diagnostic_table"]:
        rows.append({"group": "controls", **row})
    for name in NAMES:
        summary = json.loads((ROOT / "eval" / name / "summary.json").read_text(encoding="utf-8"))
        for row in summary["diagnostic_table"]:
            rows.append({"group": name, **row})
    fields = sorted({key for row in rows for key in row})
    with open(ROOT / "summary" / "all_diagnostic_rows.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "status": "complete",
        "model": "/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B",
        "train_samples": 512,
        "eval_samples": 64,
        "max_context_tokens": 256,
        "max_new_tokens": 16,
        "seed": 1234,
        "experiments": NAMES,
        "expected_file_count": len(expected),
        "missing_files": missing,
    }
    (ROOT / "summary" / "SUCCESS.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
