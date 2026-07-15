import argparse
import csv
import json
from pathlib import Path


def load_if_exists(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Summarize P2-C2 enhanced Writer ablations")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--current-p2c1", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for variant in ("global_only", "global_head", "full_staged"):
        result = load_if_exists(Path(args.eval_root) / variant / "SUCCESS.json")
        if result is None:
            continue
        condition_map = {row["condition"]: row for row in result["conditions"]}
        rows.append(
            {
                "variant": variant,
                "writer_base_em": condition_map["enhanced_writer_1_7b_base"]["target_em"],
                "writer_counterfactual_em": condition_map["enhanced_writer_1_7b_counterfactual"]["target_em"],
                "paired_consistency": result["paired_consistency"]["enhanced_writer_1_7b"],
                "route_kl": result["route_readout_summary"]["route_kl_mean"],
                "target_attention_mass": result["route_readout_summary"]["writer_target_attention_mass_mean"],
                "readout_cosine": result["route_readout_summary"]["readout_cosine_mean"],
                "paired_recovery": result["native_gap_recovery"]["paired_consistency_writer_vs_raw"],
                "success_level": result["success_level"],
            }
        )
    current = load_if_exists(Path(args.current_p2c1)) if args.current_p2c1 else None
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(output / "comparison.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "enhanced_variants": rows,
                "current_p2c1_reference": current,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
