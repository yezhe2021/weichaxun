import argparse
import csv
from pathlib import Path

from p3d_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); parser.add_argument("--native-result", required=True); args = parser.parse_args(); root = Path(args.root)
    oracle = read_json(root / "step1_oracle/SUCCESS.json"); configurations = {}
    native = read_json(args.native_result); native_correct = native["conditions"]["correct"]["f1"]
    rows = []
    for name in ("uniform8", "midlate8", "key4", "all36"):
        result = read_json(root / f"step3_capacity/{name}/eval/SUCCESS.json"); configurations[name] = result
        for condition, metrics in result["conditions"].items():
            rows.append({"configuration": name, "condition": condition, "em": metrics["em"], "f1": metrics["f1"], "bridge_f1": metrics.get("bridge_f1"), "comparison_f1": metrics.get("comparison_f1"), "compatibility_score": metrics["compatibility_score"]})
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    candidates = []
    for name, result in configurations.items():
        correct = result["conditions"]["correct"]; shuffled = result["conditions"]["hard_shuffled"]
        checks = {"correct_f1_at_least_0_38": correct["f1"] >= 0.38, "gap_at_least_0_10": correct["f1"] - shuffled["f1"] >= 0.10, "bridge_f1_at_least_0_25": correct.get("bridge_f1", 0.0) >= 0.25, "within_0_03_native": correct["f1"] >= native_correct - 0.03}
        candidates.append({"configuration": name, "checks": checks, "minimum_success": all(checks.values())})
    if "causal_gaps" in oracle:
        ordinary = oracle["causal_gaps"]["ordinary"]
        oracle_layer = oracle["causal_gaps"]["oracle_token_layer"]
    else:
        standard_mode = oracle["modes"]["standard"]
        oracle_mode = oracle["modes"]["oracle_token_layer"]
        ordinary = {"correct_minus_shuffled_f1": standard_mode["correct_minus_shuffled_f1"], "bridge_f1": standard_mode["correct"].get("bridge", {}).get("f1")}
        oracle_layer = {"correct_minus_shuffled_f1": oracle_mode["correct_minus_shuffled_f1"], "bridge_f1": oracle_mode["correct"].get("bridge", {}).get("f1")}
    oracle_decisive = (oracle_layer["correct_minus_shuffled_f1"] - ordinary["correct_minus_shuffled_f1"] >= 0.05) or ((oracle_layer.get("bridge_f1") or 0.0) - (ordinary.get("bridge_f1") or 0.0) >= 0.05)
    if any(item["minimum_success"] for item in candidates): disposition = "question_independent_route_meets_minimum_standard"
    elif oracle_decisive: disposition = "redesign_external_reader_before_more_question_independent_training"
    else: disposition = "stop_question_independent_route_and_consider_q_aware_sender"
    write_json(root / "STEPS_1_3_SUCCESS.json", {"status": "complete", "experiment": "P3-D2 Question-Independent Reader Grounding and Execution Compatibility", "completed_steps": ["oracle_grounding", "receiver_native_grounded_reader", "reader_capacity_and_injection_comparison"], "step4_executed": False, "oracle_grounding_decisive": oracle_decisive, "native_reference_f1": native_correct, "candidates": candidates, "disposition_after_step3": disposition, "oracle": oracle, "configurations": configurations})


if __name__ == "__main__": main()
