import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ("mse_only", "mse_then_ce", "paper_rec_then_mixed_generation", "q_aware_functional")
METHOD_GROUP = {
    "mse_only": "baseline",
    "mse_then_ce": "baseline",
    "paper_rec_then_mixed_generation": "paper",
    "q_aware_functional": "ours",
}
DATASETS = (
    ("hotpotqa", ROOT / "eval"),
    ("gsm8k", ROOT / "eval_gsm8k"),
)
KEY_METRICS = (
    "translated_ce",
    "ce_delta",
    "top1_match",
    "answer_f1",
    "final_answer_exact_match",
    "logit_kl",
    "route_overlap",
    "attention_output_cos",
    "readout_loss",
    "kv_joint_consistency",
)
TRAIN_METRICS = (
    "val_rec",
    "val_aware_ce",
    "val_unaware_ce",
    "val_mixed_gen",
    "val_qaware_logit_kl",
    "val_qaware_route",
    "val_qaware_readout",
)


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


def read_eval_rows():
    rows = []
    for dataset, eval_root in DATASETS:
        for method in METHODS:
            summary_path = eval_root / method / "summary.json"
            if not summary_path.is_file():
                continue
            payload = load_json(summary_path)
            for row in payload.get("diagnostic_table", []):
                rows.append(
                    {
                        "dataset": dataset,
                        "method_group": METHOD_GROUP.get(method, "unknown"),
                        **row,
                    }
                )
    return rows


def read_train_rows():
    summary_path = ROOT / "summary_train" / "train_method_comparison.csv"
    if summary_path.is_file():
        with open(summary_path, "r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    rows = []
    for method in METHODS:
        metadata_path = ROOT / "train" / method / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = load_json(metadata_path)
        rows.append(
            {
                "method_group": METHOD_GROUP.get(method, "unknown"),
                "method": method,
                "stage_name": metadata.get("stage_name"),
                "global_step": metadata.get("global_step"),
                **{key: metadata.get(key) for key in TRAIN_METRICS},
            }
        )
    return rows


def index_eval(eval_rows):
    indexed = {}
    for row in eval_rows:
        key = (row.get("dataset"), row.get("method"), row.get("receiver_prompt_mode"))
        indexed[key] = row
    return indexed


def make_wide_summary(train_rows, eval_rows):
    eval_index = index_eval(eval_rows)
    train_index = {row.get("method"): row for row in train_rows}
    wide = []
    for method in METHODS:
        row = {"method_group": METHOD_GROUP.get(method, "unknown"), "method": method}
        train = train_index.get(method, {})
        for metric in ("stage_name", "global_step", *TRAIN_METRICS):
            if metric in train:
                row[f"train_{metric}"] = train[metric]
        for dataset, _ in DATASETS:
            for mode in ("context_aware", "context_unaware"):
                metrics = eval_index.get((dataset, method, mode), {})
                for metric in KEY_METRICS:
                    if metric in metrics:
                        row[f"{dataset}_{mode}_{metric}"] = metrics[metric]
        wide.append(row)
    return wide


def fmt(value, digits=4):
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def make_report(train_rows, eval_rows, wide_rows):
    lines = [
        "# Qwen3-1.7B -> Qwen3-4B Dense KV Alignment Summary",
        "",
        "Generated from existing train/eval result files. No model inference is run by this packaging script.",
        "",
        "## Train Summary",
        "",
    ]
    train_cols = ["method_group", "method", "stage_name", "global_step", "val_rec", "val_aware_ce", "val_unaware_ce", "val_mixed_gen"]
    lines.append(markdown_table(train_rows, train_cols))
    lines.append("")
    for dataset in ("hotpotqa", "gsm8k"):
        lines.append(f"## {dataset.upper()} Evaluation")
        lines.append("")
        selected = [row for row in eval_rows if row.get("dataset") == dataset]
        cols = [
            "method_group",
            "method",
            "receiver_prompt_mode",
            "translated_ce",
            "ce_delta",
            "top1_match",
            "answer_f1",
            "final_answer_exact_match",
            "logit_kl",
            "route_overlap",
            "readout_loss",
        ]
        lines.append(markdown_table(selected, cols))
        lines.append("")
    lines.extend(
        [
            "## Output Files",
            "",
            "- `eval_all_datasets.csv`: long-form evaluation table across HotpotQA and GSM8K.",
            "- `train_summary.csv`: training-stage validation summary.",
            "- `overall_wide_summary.csv`: one row per method with train, HotpotQA, and GSM8K key metrics.",
            "- `overall_report.md`: this readable summary.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    out = ROOT / "summary_all"
    out.mkdir(parents=True, exist_ok=True)
    train_rows = read_train_rows()
    eval_rows = read_eval_rows()
    wide_rows = make_wide_summary(train_rows, eval_rows)
    write_csv(out / "train_summary.csv", train_rows)
    write_csv(out / "eval_all_datasets.csv", eval_rows)
    write_csv(out / "overall_wide_summary.csv", wide_rows)
    (out / "overall_report.md").write_text(
        make_report(train_rows, eval_rows, wide_rows),
        encoding="utf-8",
    )
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "methods": list(METHODS),
                "datasets": [name for name, _ in DATASETS],
                "outputs": [
                    "train_summary.csv",
                    "eval_all_datasets.csv",
                    "overall_wide_summary.csv",
                    "overall_report.md",
                ],
                "eval_rows": len(eval_rows),
                "train_rows": len(train_rows),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
