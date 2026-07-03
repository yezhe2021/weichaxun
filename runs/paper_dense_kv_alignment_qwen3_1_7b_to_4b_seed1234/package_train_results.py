import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ("paper_rec_then_mixed_generation", "mse_only", "mse_then_ce", "q_aware_functional")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_history(path):
    rows = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def method_group(method):
    if method.startswith("paper_"):
        return "paper"
    if method == "q_aware_functional":
        return "ours"
    return "baseline"


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    summary_rows = []
    last_step_rows = []
    completed = []
    for method in METHODS:
        train_dir = ROOT / "train" / method
        metadata_path = train_dir / "metadata.json"
        history_path = train_dir / "train_history.jsonl"
        if not metadata_path.is_file():
            continue
        metadata = load_json(metadata_path)
        history = load_history(history_path)
        completed.append(method)
        summary_rows.append(
            {
                "method_group": method_group(method),
                "method": method,
                "stage_name": metadata.get("stage_name"),
                "stage_index": metadata.get("stage_index"),
                "global_step": metadata.get("global_step"),
                "val_rec": metadata.get("val_rec"),
                "val_aware_ce": metadata.get("val_aware_ce"),
                "val_unaware_ce": metadata.get("val_unaware_ce"),
                "val_mixed_gen": metadata.get("val_mixed_gen"),
                "val_qaware_logit_kl": metadata.get("val_qaware_logit_kl"),
                "val_qaware_route": metadata.get("val_qaware_route"),
                "val_qaware_readout": metadata.get("val_qaware_readout"),
                "metadata": str(metadata_path),
                "train_history": str(history_path),
            }
        )
        if history:
            last = history[-1]
            last_step_rows.append(
                {
                    "method_group": method_group(method),
                    "method": method,
                    "last_stage": last.get("stage"),
                    "last_step": last.get("step"),
                    "last_loss": last.get("loss"),
                    "last_rec_loss": last.get("receiver_cache_reconstruction_loss"),
                    "last_generation_loss": last.get("generation_loss"),
                    "last_context_aware_ce": last.get("context_aware_ce"),
                    "last_context_unaware_ce": last.get("context_unaware_ce"),
                    "last_logit_kl_loss": last.get("logit_kl_loss"),
                    "last_route_loss": last.get("route_loss"),
                    "last_readout_loss": last.get("readout_loss"),
                }
            )
    out = ROOT / "summary_train"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "train_method_comparison.csv", summary_rows)
    write_csv(out / "train_last_step_comparison.csv", last_step_rows)
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "completed_methods": completed,
                "required_methods": list(METHODS),
                "outputs": ["train_method_comparison.csv", "train_last_step_comparison.csv"],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
