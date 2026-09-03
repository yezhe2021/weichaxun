import argparse
from pathlib import Path

from p3e_f_common import read_json, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    errors = []
    required = [
        root / "smoke16/stage_a/TRAIN_SUCCESS.json",
        root / "smoke16/stage_b/TRAIN_SUCCESS.json",
        root / "formal512/stage_a/TRAIN_SUCCESS.json",
        root / "formal512/stage_b/TRAIN_SUCCESS.json",
        root / "stage_a_validation64/SUCCESS.json",
        root / "eval64/SUCCESS.json",
        root / "eval64/per_sample_generation.jsonl",
        root / "review/semantic_review_blinded.csv",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing:{path}")
    if not errors:
        stage_a = read_json(root / "formal512/stage_a/TRAIN_SUCCESS.json")
        stage_b = read_json(root / "formal512/stage_b/TRAIN_SUCCESS.json")
        evaluation = read_json(root / "eval64/SUCCESS.json")
        rows = read_jsonl(root / "eval64/per_sample_generation.jsonl")
        if stage_a["samples"] != 512 or stage_b["samples"] != 512:
            errors.append("formal_training_not_512")
        if stage_a["receiver_trainable_parameters"] != 0:
            errors.append("receiver_trainable_in_stage_a")
        if stage_b["receiver_trainable_parameters"] != 0:
            errors.append("receiver_trainable_in_stage_b")
        if len(stage_a["history"]) != stage_a["epochs"]:
            errors.append("stage_a_epoch_checkpoint_history_mismatch")
        if len(stage_b["history"]) != stage_b["epochs"]:
            errors.append("stage_b_epoch_checkpoint_history_mismatch")
        if evaluation["samples"] != 64 or len(rows) != 64 * 8:
            errors.append("validation_condition_count_mismatch")
        if evaluation["input_ids_exact_embeddings_generation_consistency"] != 1.0:
            errors.append("exact_embedding_generation_mismatch")
        if evaluation["soft_prefix_off_question_only_consistency"] != 1.0:
            errors.append("soft_prefix_off_mismatch")
        for condition, metrics in evaluation["conditions"].items():
            if metrics["n"] != 64:
                errors.append(f"condition_count:{condition}")
    result = {
        "status": "complete" if not errors else "failed",
        "errors": errors,
        "diagnostic_claim": (
            "P3-E-J diagnoses Canonical decodability versus frozen-Receiver "
            "executability; Soft Prefix is not claimed as the final communication method."
        ),
    }
    write_json(root / "audit/SUCCESS.json", result)
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
