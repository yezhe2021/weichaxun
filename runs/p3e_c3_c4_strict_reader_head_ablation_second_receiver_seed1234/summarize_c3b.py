import argparse

from p3d3_common import read_json, write_json


def compact(result):
    pairs = {}
    for pair, modes in result["pairs"].items():
        pairs[pair] = {mode: {"correct_f1": values["correct"]["f1"],
                              "bridge_f1": values["correct"].get("by_type", {}).get("bridge", {}).get("f1"),
                              "comparison_f1": values["correct"].get("by_type", {}).get("comparison", {}).get("f1"),
                              "correct_shuffled_gap": values["correct_shuffled_f1_gap"]} for mode, values in modes.items()}
    return {"seed": result["seed"], "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--runs", nargs=2, required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); runs = [compact(read_json(path)) for path in args.runs]; stable = {}
    for pair in map(str, range(8)):
        seed_rows = []
        for run in runs:
            modes = run["pairs"][pair]; both = modes["both"]["correct_f1"]; first, second, swap = modes["first_only"]["correct_f1"], modes["second_only"]["correct_f1"], modes["swap"]["correct_f1"]
            seed_rows.append({"seed": run["seed"], "both_advantage_over_best_single": both - max(first, second),
                              "single_head_asymmetry": abs(first - second), "swap_drop": both - swap})
        stable[pair] = {"seeds": seed_rows,
                        "stable_differentiation": all(max(row["both_advantage_over_best_single"], row["single_head_asymmetry"], row["swap_drop"]) >= 0.02 for row in seed_rows)}
    write_json(args.out, {"status": "complete", "experiment": "C3-B paired Canonical-head functional differentiation", "runs": runs,
                          "stable_pair_diagnostics": stable, "stable_differentiated_pair_count": sum(value["stable_differentiation"] for value in stable.values())})


if __name__ == "__main__": main()
