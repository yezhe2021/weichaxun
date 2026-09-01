from __future__ import annotations

import argparse
from pathlib import Path

from common import load_json, save_json


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    root = Path(cfg["work_dir"])
    references = cfg["forward_reference"]

    reverse_a = load_json(root / "checkpoints/quick/full36/stage_a/summary.json")
    reverse_b = load_json(root / "checkpoints/quick/full36/stage_b_final/summary.json")
    reverse_eval = load_json(root / "artifacts/evaluation/formal/summary.json")
    forward_a = load_json(references["stage_a_summary"])
    forward_b = load_json(references["stage_b_summary"])
    forward_eval = load_json(references["formal_summary"])

    reverse_conditions = reverse_eval["conditions"]
    forward_conditions = forward_eval["conditions"]
    reverse_writer = reverse_conditions["full36_b1_final_kl_correct"]
    reverse_native = reverse_conditions["1_7b_native_full_kv"]
    forward_writer = forward_conditions["full28_b1_final_kl_correct"]
    forward_native = forward_conditions["4b_native_full_kv"]

    save_json(root / "artifacts/comparison_to_forward_full28.json", {
        "protocol": {
            "same_seed": True,
            "same_splits": True,
            "same_paired_full_kv_cache": True,
            "same_stage_a_objective": "KV_reconstruction",
            "same_stage_b_objective": "final_position_full_vocabulary_KL",
            "same_train_validation_test_sizes": [1024, 128, 128],
            "same_writer_parameter_count": 33030144,
            "forward_direction": "Qwen3-1.7B_28_layers_to_Qwen3-4B_36_layers",
            "reverse_direction": "Qwen3-4B_36_layers_to_Qwen3-1.7B_28_layers",
            "warning": "Raw accuracy ceilings differ because the frozen Receiver differs by direction.",
        },
        "stage_a": {
            "reverse_full36_best_validation_kv_loss": reverse_a["best_validation_loss"],
            "forward_full28_best_validation_kv_loss": forward_a["best_validation_loss"],
            "reverse_minus_forward": (
                reverse_a["best_validation_loss"] - forward_a["best_validation_loss"]
            ),
        },
        "stage_b": {
            "reverse_full36_best_validation_final_kl": reverse_b["best_validation_functional_kl"],
            "forward_full28_best_validation_final_kl": forward_b["best_validation_functional_kl"],
            "reverse_minus_forward": (
                reverse_b["best_validation_functional_kl"]
                - forward_b["best_validation_functional_kl"]
            ),
            "reverse_clip_rate": reverse_b["clip_rate"],
            "forward_clip_rate": forward_b["clip_rate"],
        },
        "formal_test": {
            "reverse_native_receiver_accuracy": reverse_native["accuracy"],
            "reverse_writer_accuracy": reverse_writer["accuracy"],
            "reverse_accuracy_retention": safe_ratio(
                reverse_writer["accuracy"], reverse_native["accuracy"]
            ),
            "reverse_native_agreement": reverse_writer["native_agreement"],
            "reverse_native_choice_kl": reverse_writer["native_to_condition_choice_kl"],
            "forward_native_receiver_accuracy": forward_native["accuracy"],
            "forward_writer_accuracy": forward_writer["accuracy"],
            "forward_accuracy_retention": safe_ratio(
                forward_writer["accuracy"], forward_native["accuracy"]
            ),
            "forward_native_agreement": forward_writer["native_agreement"],
            "forward_native_choice_kl": forward_writer["native_to_condition_choice_kl"],
        },
        "forward_references": references,
    })


if __name__ == "__main__":
    main()
