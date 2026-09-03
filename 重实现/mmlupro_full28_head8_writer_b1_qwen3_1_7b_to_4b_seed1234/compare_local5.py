from __future__ import annotations

import argparse
from pathlib import Path

from common import load_json, save_json


def ratio(full28: float, local5: float) -> float:
    return full28 / local5 if local5 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    root = Path(cfg["work_dir"])
    references = cfg["local5_reference"]

    full_a = load_json(root / "checkpoints/quick/full28/stage_a/summary.json")
    full_b = load_json(root / "checkpoints/quick/full28/stage_b_final/summary.json")
    full_eval = load_json(root / "artifacts/evaluation/formal/summary.json")
    local_a = load_json(references["stage_a_summary"])
    local_b = load_json(references["stage_b_summary"])
    local_eval = load_json(references["formal_summary"])

    full_condition = full_eval["conditions"]["full28_b1_final_kl_correct"]
    local_condition = local_eval["conditions"]["d2_b1_final_kl_correct"]
    full_a_loss = float(full_a["best_validation_loss"])
    local_a_loss = float(local_a["best_validation_loss"])
    full_b_kl = float(full_b["best_validation_functional_kl"])
    local_b_kl = float(local_b["best_validation_functional_kl"])

    save_json(root / "artifacts/comparison_to_local5.json", {
        "protocol": {
            "same_seed": True,
            "same_splits": True,
            "same_cache": True,
            "same_stage_a_objective": "KV_reconstruction",
            "same_stage_b_objective": "final_position_full_vocabulary_KL",
            "full28_source_layers_per_target": 28,
            "local5_source_layers_per_target": 5,
        },
        "stage_a": {
            "full28_best_validation_kv_loss": full_a_loss,
            "local5_best_validation_kv_loss": local_a_loss,
            "full28_minus_local5": full_a_loss - local_a_loss,
            "full28_over_local5": ratio(full_a_loss, local_a_loss),
            "full28_better": full_a_loss < local_a_loss,
        },
        "stage_b": {
            "full28_best_validation_final_kl": full_b_kl,
            "local5_best_validation_final_kl": local_b_kl,
            "full28_minus_local5": full_b_kl - local_b_kl,
            "full28_over_local5": ratio(full_b_kl, local_b_kl),
            "full28_better": full_b_kl < local_b_kl,
        },
        "formal_test": {
            "full28_accuracy": full_condition["accuracy"],
            "local5_accuracy": local_condition["accuracy"],
            "full28_minus_local5_accuracy": full_condition["accuracy"] - local_condition["accuracy"],
            "full28_native_agreement": full_condition["native_agreement"],
            "local5_native_agreement": local_condition["native_agreement"],
            "full28_minus_local5_native_agreement": (
                full_condition["native_agreement"] - local_condition["native_agreement"]
            ),
            "full28_native_choice_kl": full_condition["native_to_condition_choice_kl"],
            "local5_native_choice_kl": local_condition["native_to_condition_choice_kl"],
        },
        "local5_references": references,
    })


if __name__ == "__main__":
    main()
