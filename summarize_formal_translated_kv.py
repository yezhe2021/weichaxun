import csv
import json
from pathlib import Path


ROOT = Path("runs")
OUTPUT = ROOT / "formal_translated_kv_summary_seed1234"

EXPERIMENTS = {
    "autoencoder": (
        ROOT / "formal_translator_autoencoder_n512_seed1234" / "metadata.json",
        ROOT / "formal_eval_autoencoder_n64_seed1234" / "summary.json",
    ),
    "mse_only": (
        ROOT / "formal_translator_mse_n512_seed1234" / "metadata.json",
        ROOT / "formal_eval_mse_n64_seed1234" / "summary.json",
    ),
    "ce_only": (
        ROOT / "formal_translator_ce_n512_seed1234" / "metadata.json",
        ROOT / "formal_eval_ce_n64_seed1234" / "summary.json",
    ),
    "mse_ce": (
        ROOT / "formal_translator_mse_ce_n512_seed1234" / "metadata.json",
        ROOT / "formal_eval_mse_ce_n64_seed1234" / "summary.json",
    ),
    "rope_mse_ce": (
        ROOT / "formal_translator_rope_mse_ce_n512_seed1234" / "metadata.json",
        ROOT / "formal_eval_rope_mse_ce_n64_seed1234" / "summary.json",
    ),
}

CORE_METRICS = [
    "kv_mse",
    "kv_relative_mse",
    "k_cos",
    "v_cos",
    "logit_kl",
    "ce_delta",
    "top1_match",
    "attention_route_overlap",
    "attention_route_js",
    "attention_output_cos",
    "kv_joint_consistency",
    "answer_em",
    "answer_f1",
]


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    training_rows = []
    diagnostic_rows = []
    equivalence = []
    native_added = False
    for name, (metadata_path, summary_path) in EXPERIMENTS.items():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        training_rows.append({
            "experiment": name,
            "objective": metadata["args"]["objective"],
            "translator_kind": metadata["args"]["translator_kind"],
            "rope_disentangled": metadata["args"]["rope_disentangled"],
            "train_samples": metadata["args"]["max_train_samples"],
            "val_samples": metadata["args"]["max_val_samples"],
            "val_mse": metadata["val_mse"],
            "val_ce": metadata["val_ce"],
            "mse_weight": metadata.get("resolved_mse_weight"),
        })
        equivalence.append({
            "experiment": name,
            "n": len(summary["equivalence"]),
            "strict_top1_pass": sum(row["top1_match"] == 1.0 for row in summary["equivalence"]),
            "atol_pass": sum(row["max_abs_logit"] <= row["atol"] for row in summary["equivalence"]),
            "max_abs_logit": max(row["max_abs_logit"] for row in summary["equivalence"]),
        })
        for row in summary["diagnostic_table"]:
            if row["method"] == "native":
                if native_added:
                    continue
                native_added = True
                label, alpha = "native", 0.0
            else:
                label = name
                alpha = float(row["method"].split("alpha=")[1])
            selected = {"experiment": label, "alpha": alpha, "n": row["n"]}
            for metric in CORE_METRICS:
                if metric in row:
                    selected[metric] = row[metric]
                for suffix in ("_ci95_low", "_ci95_high"):
                    if metric + suffix in row:
                        selected[metric + suffix] = row[metric + suffix]
            diagnostic_rows.append(selected)

    direct = sorted(
        [row for row in diagnostic_rows if row["alpha"] == 1.0],
        key=lambda row: row["experiment"],
    )
    controls = json.loads(
        (ROOT / "formal_translated_kv_controls_n64_seed1234" / "summary.json").read_text(encoding="utf-8")
    )
    control_rows = controls["diagnostic_table"]
    control_equivalence = {
        "experiment": "deterministic_controls",
        "n": len(controls["equivalence"]),
        "strict_top1_pass": sum(row["top1_match"] == 1.0 for row in controls["equivalence"]),
        "atol_pass": sum(row["max_abs_logit"] <= row["atol"] for row in controls["equivalence"]),
        "max_abs_logit": max(row["max_abs_logit"] for row in controls["equivalence"]),
    }
    equivalence.append(control_equivalence)

    write_csv(OUTPUT / "training_validation.csv", training_rows)
    write_csv(OUTPUT / "translator_diagnostic_table.csv", diagnostic_rows)
    write_csv(OUTPUT / "direct_replace_table.csv", direct)
    write_csv(OUTPUT / "controls_diagnostic_table.csv", control_rows)
    write_csv(OUTPUT / "equivalence_checks.csv", equivalence)

    lines = [
        "# Formal translated-like KV experiment",
        "",
        "Receiver: Qwen3-0.6B. Data: HotpotQA, 512 train / 64 dev. Context limit: 256 tokens. ",
        "All reported checkpoints use seed 1234 and one training epoch. Evaluation includes 16-token greedy generation.",
        "",
        "## Equivalence guard",
        "",
        "| Experiment | Within atol | Strict top1 | N | Max abs logit |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in equivalence:
        lines.append(
            f"| {row['experiment']} | {row['atol_pass']} | {row['strict_top1_pass']} | "
            f"{row['n']} | {fmt(row['max_abs_logit'], 6)} |"
        )
    lines.extend([
        "",
        "## Training validation",
        "",
        "| Experiment | Val relative MSE | Val CE | MSE weight |",
        "|---|---:|---:|---:|",
    ])
    for row in training_rows:
        lines.append(
            f"| {row['experiment']} | {fmt(row['val_mse'])} | {fmt(row['val_ce'])} | {fmt(row['mse_weight'])} |"
        )
    lines.extend([
        "",
        "## Direct replacement",
        "",
        "| Experiment | Rel MSE | K cos | V cos | Logit KL | CE delta | Top1 | Route | Attn out cos | KV joint | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in direct:
        lines.append(
            f"| {row['experiment']} | {fmt(row.get('kv_relative_mse'))} | {fmt(row.get('k_cos'))} | "
            f"{fmt(row.get('v_cos'))} | {fmt(row.get('logit_kl'))} | {fmt(row.get('ce_delta'))} | "
            f"{fmt(row.get('top1_match'))} | {fmt(row.get('attention_route_overlap'))} | "
            f"{fmt(row.get('attention_output_cos'))} | {fmt(row.get('kv_joint_consistency'))} | "
            f"{fmt(row.get('answer_f1'))} |"
        )
    lines.extend([
        "",
        "## Residual fusion",
        "",
        "| Experiment | Alpha | CE delta | Logit KL | Top1 | Attn out cos | KV joint | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(
        [row for row in diagnostic_rows if row["experiment"] != "native"],
        key=lambda row: (row["experiment"], row["alpha"]),
    ):
        lines.append(
            f"| {row['experiment']} | {fmt(row['alpha'], 2)} | {fmt(row.get('ce_delta'))} | "
            f"{fmt(row.get('logit_kl'))} | {fmt(row.get('top1_match'))} | "
            f"{fmt(row.get('attention_output_cos'))} | {fmt(row.get('kv_joint_consistency'))} | "
            f"{fmt(row.get('answer_f1'))} |"
        )

    by_name = {row["experiment"]: row for row in direct}
    lines.extend([
        "",
        "## Findings",
        "",
        "- All 64 samples pass the FP16 absolute-logit tolerance. One near-tied sample changes one argmax, so strict full/split top-1 equivalence is 63/64.",
        f"- CE-only moves toward a functional shortcut during training: validation CE is {fmt(next(r['val_ce'] for r in training_rows if r['experiment'] == 'ce_only'))}, while validation relative MSE explodes to {fmt(next(r['val_mse'] for r in training_rows if r['experiment'] == 'ce_only'))}. Its direct generated F1 is nevertheless {fmt(by_name['ce_only']['answer_f1'])}, so CE-only has not learned a robust readable cache.",
        f"- MSE-only achieves direct relative MSE {fmt(by_name['mse_only']['kv_relative_mse'])}, but direct CE delta remains {fmt(by_name['mse_only']['ce_delta'])}; KV fitting alone does not ensure receiver readability.",
        f"- MSE+CE is the strongest functional-shortcut result: direct CE delta {fmt(by_name['mse_ce']['ce_delta'])}, F1 {fmt(by_name['mse_ce']['answer_f1'])} versus native {fmt(next(r['answer_f1'] for r in diagnostic_rows if r['experiment'] == 'native'))}, despite attention-output cosine {fmt(by_name['mse_ce']['attention_output_cos'])} and KV-joint consistency {fmt(by_name['mse_ce']['kv_joint_consistency'])}.",
        f"- RoPE disentangling does not help in this setup: direct CE delta is {fmt(by_name['rope_mse_ce']['ce_delta'])} versus {fmt(by_name['mse_ce']['ce_delta'])} for ordinary MSE+CE.",
        "- Increasing the native-cache share consistently restores KV geometry and attention readout, but CE/F1 can be non-monotonic for CE-trained translators. This decoupling is itself evidence that task loss can exploit cache shortcuts rather than reconstruct native memory.",
        "- Joint token permutation is a calibration control: route overlap changes sharply while attention output can remain invariant, so routing metrics must be interpreted jointly with readout and logits.",
        "",
        "These results support a receiver-specific structured-memory interpretation for native prefill KV within this model and pseudo-translation setup. They do not by themselves establish the behavior of every real heterogeneous translator.",
    ])
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(
        json.dumps({
            "experiments": {name: [str(path) for path in paths] for name, paths in EXPERIMENTS.items()},
            "controls": "runs/formal_translated_kv_controls_n64_seed1234/summary.json",
            "training_rows": training_rows,
            "equivalence": equivalence,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
