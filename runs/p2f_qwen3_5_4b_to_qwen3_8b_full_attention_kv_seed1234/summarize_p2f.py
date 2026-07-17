import argparse
import csv
import json
from pathlib import Path


VARIANTS = ("matched_task_only", "reader_aligned")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--sender-gate", required=True)
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
        rows.append(
            {
                "variant": variant,
                "writer_base_em": conditions["writer_qwen35_4b_base"]["target_em"],
                "writer_cf_em": conditions["writer_qwen35_4b_counterfactual"]["target_em"],
                "writer_paired": payload["paired_consistency"]["writer_qwen35_4b"],
                "native_paired": payload["paired_consistency"]["native_8b"],
                "raw_paired": payload["paired_consistency"]["raw_minimal_qwen35_4b"],
                "route_kl": payload["route_readout_summary"]["route_kl_mean"],
                "readout_cosine": payload["route_readout_summary"]["readout_cosine_mean"],
                "binding_error": payload["structure_summary"]["token_binding_error"],
            }
        )
    with open(args.sender_gate, encoding="utf-8") as handle:
        sender_gate = json.load(handle)
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
                "experiment": "P2-F Qwen3.5-4B full-attention KV to frozen Qwen3-8B Query Reader",
                "variants": rows,
                "qwen35_direct_answerability_gate": sender_gate,
                "claim_boundary": (
                    "Only the eight genuine Qwen3.5 full-attention layers are transferred. "
                    "Linear-attention recurrent states are excluded, not converted into fake KV."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
