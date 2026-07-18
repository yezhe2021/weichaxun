import argparse
import csv
import json
from pathlib import Path


VARIANTS = ("matched_task_only", "reader_aligned")


def condition(conditions, name, field="target_em"):
    return conditions[name][field]


def main():
    parser = argparse.ArgumentParser(description="Summarize the two P2-G2 reverse Writer variants")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--native-gate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    payloads = {}
    for variant in VARIANTS:
        path = Path(args.eval_root) / variant / "SUCCESS.json"
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        payloads[variant] = payload
        conditions = {row["condition"]: row for row in payload["conditions"]}
        writer_base = condition(conditions, "writer_8b_to_4b_base")
        writer_cf = condition(conditions, "writer_8b_to_4b_counterfactual")
        rows.append(
            {
                "variant": variant,
                "full_text_base_em": condition(conditions, "full_text_base"),
                "full_text_cf_em": condition(conditions, "full_text_counterfactual"),
                "full_text_paired": payload["paired_consistency"]["full_text"],
                "native_4b_base_em": condition(conditions, "native_4b_base"),
                "native_4b_cf_em": condition(conditions, "native_4b_counterfactual"),
                "native_4b_paired": payload["paired_consistency"]["native_4b"],
                "raw_8b_base_em": condition(conditions, "raw_minimal_8b_base"),
                "raw_8b_cf_em": condition(conditions, "raw_minimal_8b_counterfactual"),
                "raw_8b_paired": payload["paired_consistency"]["raw_minimal_8b"],
                "writer_base_em": writer_base,
                "writer_cf_em": writer_cf,
                "writer_mean_em": 0.5 * (writer_base + writer_cf),
                "writer_paired": payload["paired_consistency"]["writer_8b_to_4b"],
                "writer_switch_rate": payload["prediction_switch_rate"]["writer_8b_to_4b"],
                "shuffled_original_answer_em": condition(conditions, "writer_shuffled"),
                "shuffled_memory_answer_hit": condition(
                    conditions, "writer_shuffled", "memory_answer_hit_rate"
                ),
                "k_mismatched_original_answer_em": condition(conditions, "writer_key_mismatched"),
                "v_mismatched_original_answer_em": condition(conditions, "writer_value_mismatched"),
                "zero_original_answer_em": condition(conditions, "zero_kv"),
                "reader_off_original_answer_em": condition(conditions, "reader_off"),
                "route_kl": payload["route_readout_summary"]["route_kl_mean"],
                "readout_cosine": payload["route_readout_summary"]["readout_cosine_mean"],
                "writer_target_attention_mass": payload["route_readout_summary"][
                    "writer_target_attention_mass_mean"
                ],
                "native_target_attention_mass": payload["route_readout_summary"][
                    "native_target_attention_mass_mean"
                ],
                "mean_em_recovery": payload["native_gap_recovery"]["mean_em_writer_vs_raw"],
                "paired_recovery": payload["native_gap_recovery"][
                    "paired_consistency_writer_vs_raw"
                ],
            }
        )

    with open(args.native_gate, encoding="utf-8") as handle:
        native_gate = json.load(handle)
    by_variant = {row["variant"]: row for row in rows}
    task = by_variant["matched_task_only"]
    aligned = by_variant["reader_aligned"]
    success = (
        aligned["writer_base_em"] >= 0.85
        and aligned["writer_cf_em"] >= 0.85
        and aligned["writer_paired"] >= 0.80
        and aligned["writer_paired"] > task["writer_paired"]
    )

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "comparison.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "experiment": "P2-G2 Qwen3-8B sender to frozen Qwen3-4B Query Reader",
                "variants": rows,
                "native_reader_gate": native_gate,
                "success_criteria_met": success,
                "success_criteria": {
                    "reader_aligned_base_em_at_least": 0.85,
                    "reader_aligned_cf_em_at_least": 0.85,
                    "reader_aligned_paired_at_least": 0.80,
                    "reader_aligned_paired_exceeds_task_only": True,
                },
                "claim_boundary": (
                    "This tests direction reversal and Reader-aligned transfer on the current synthetic task. "
                    "It does not establish capability amplification or receiver-independent Canonical Evidence-KV."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
