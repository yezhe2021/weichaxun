import argparse
import csv
import json
from pathlib import Path


VARIANTS = ("task_only", "shared_span_relation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--llama-gate-reference", default="")
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
                "writer_base_em": conditions["writer_llama3_2_3b_base"]["target_em"],
                "writer_cf_em": conditions["writer_llama3_2_3b_counterfactual"]["target_em"],
                "writer_paired": payload["paired_consistency"]["writer_llama3_2_3b"],
                "native_paired": payload["paired_consistency"]["native_8b"],
                "raw_paired": payload["paired_consistency"]["raw_minimal_llama3_2_3b"],
                "route_kl": payload["route_readout_summary"]["route_kl_mean"],
                "readout_cosine": payload["route_readout_summary"]["readout_cosine_mean"],
                "binding_error": payload["structure_summary"]["token_binding_error"],
            }
        )
    llama_gate = None
    if args.llama_gate_reference and Path(args.llama_gate_reference).is_file():
        with open(args.llama_gate_reference, encoding="utf-8") as handle:
            llama_gate = json.load(handle)
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
                "experiment": "P2-E Llama-3.2-3B to frozen Qwen3-8B Query Reader",
                "variants": rows,
                "llama_direct_answerability_gate": llama_gate,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
