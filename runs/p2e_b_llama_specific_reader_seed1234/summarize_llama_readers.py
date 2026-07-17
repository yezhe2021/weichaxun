import argparse
import csv
import json
from pathlib import Path


VARIANTS = ("minimal_reader", "routed_reader")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for variant in VARIANTS:
        with open(Path(args.eval_root) / variant / "SUCCESS.json", encoding="utf-8") as handle:
            result = json.load(handle)
        conditions = {row["condition"]: row for row in result["conditions"]}
        rows.append(
            {
                "variant": variant,
                "base_em": conditions["llama_reader_base"]["target_em"],
                "counterfactual_em": conditions["llama_reader_counterfactual"]["target_em"],
                "paired_consistency": result["paired_consistency"]["llama_specific_reader"],
                "native_paired_consistency": result["paired_consistency"]["native_8b"],
                "full_text_paired_consistency": result["paired_consistency"]["full_text"],
                "shuffled_target_em": conditions["llama_reader_shuffled"]["target_em"],
                "shuffled_memory_hit": conditions["llama_reader_shuffled"]["memory_answer_hit_rate"],
                "mismatched_target_em": conditions["llama_reader_mismatched"]["target_em"],
                "zero_em": conditions["llama_reader_zero"]["target_em"],
                "reader_off_em": conditions["llama_reader_off"]["target_em"],
                "target_attention_mass": result["reader_diagnostics"]["mean_target_attention_mass"],
            }
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
                "experiment": "Experiment B: Llama-specific Reader",
                "variants": rows,
                "claim_boundary": (
                    "This tests a Qwen3-8B Reader specialized for evidence-token-only raw Llama Native-KV. "
                    "It does not establish receiver-independent Canonical Evidence-KV."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
