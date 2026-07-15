import argparse
import csv
import json
from pathlib import Path


VARIANTS = (
    "task_only",
    "shared_routing",
    "binding_relation",
    "shared_routing_relation",
)


def load_json(path):
    if not path or not Path(path).is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def condition_value(condition_map, condition, field="target_em"):
    row = condition_map.get(condition)
    return None if row is None else row.get(field)


def main():
    parser = argparse.ArgumentParser(description="Summarize the four P2-C3 Writer variants")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--p2c1-reference", default="")
    parser.add_argument("--p2c2-reference", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    results = {}
    for variant in VARIANTS:
        result = load_json(Path(args.eval_root) / variant / "SUCCESS.json")
        if result is None:
            continue
        results[variant] = result
        conditions = {row["condition"]: row for row in result["conditions"]}
        route = result["route_readout_summary"]
        structure = result["structure_summary"]
        routing = result["kv_routing_difference"]
        rows.append(
            {
                "variant": variant,
                "base_em": condition_value(conditions, "enhanced_writer_1_7b_base"),
                "counterfactual_em": condition_value(
                    conditions, "enhanced_writer_1_7b_counterfactual"
                ),
                "paired_consistency": result["paired_consistency"]["enhanced_writer_1_7b"],
                "shuffled_em": condition_value(conditions, "writer_shuffled"),
                "value_mismatched_em": condition_value(conditions, "writer_value_mismatched"),
                "key_mismatched_em": condition_value(conditions, "writer_key_mismatched"),
                "zero_em": condition_value(conditions, "zero_kv"),
                "reader_off_em": condition_value(conditions, "reader_off"),
                "shuffled_memory_answer_hit": condition_value(
                    conditions, "writer_shuffled", "memory_answer_hit_rate"
                ),
                "mismatched_memory_answer_hit": condition_value(
                    conditions, "writer_value_mismatched", "memory_answer_hit_rate"
                ),
                "route_kl": route["route_kl_mean"],
                "readout_cosine": route["readout_cosine_mean"],
                "target_attention_mass": route["writer_target_attention_mass_mean"],
                "kv_layer_routing_l1": routing["layer_l1"],
                "kv_head_routing_l1": routing["head_l1"],
                "kv_layer_support_disagreement": routing["layer_support_disagreement"],
                "token_binding_error": structure["token_binding_error"],
                "key_relation_error": structure["key_relation_error"],
                "value_relation_error": structure["value_relation_error"],
                "readout_relation_error": structure["readout_relation_error"],
                "native_paired_recovery": result["native_gap_recovery"][
                    "paired_consistency_writer_vs_raw"
                ],
            }
        )

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
                "variants": rows,
                "variant_results": results,
                "p2c1_reference": load_json(args.p2c1_reference),
                "p2c2_reference": load_json(args.p2c2_reference),
                "selection_rule": (
                    "Prefer free-running base/counterfactual EM and paired consistency; "
                    "treat geometric diagnostics as secondary."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
