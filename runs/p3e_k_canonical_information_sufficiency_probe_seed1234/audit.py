import argparse
from pathlib import Path

from p3e_k_common import read_json, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    errors = []
    required = [
        root / "cache/train/SUCCESS.json",
        root / "cache/validation/SUCCESS.json",
        root / "smoke16/eval/SUCCESS.json",
        root / "eval64/SUCCESS.json",
        root / "eval64/per_sample_results.jsonl",
        root / "diagnosis.json",
    ]
    for mode in ("text", "native", "canonical", "zero"):
        required.extend([
            root / f"smoke16/{mode}/TRAIN_SUCCESS.json",
            root / f"formal512/{mode}/TRAIN_SUCCESS.json",
        ])
    for path in required:
        if not path.exists():
            errors.append(f"missing:{path}")
    if not errors:
        evaluation = read_json(root / "eval64/SUCCESS.json")
        records = read_jsonl(root / "eval64/per_sample_results.jsonl")
        if evaluation["samples"] != 64 or len(records) != 64 * 6:
            errors.append("validation_count_mismatch")
        expected = {
            "full_text_representation", "sender_native_kv",
            "learned_canonical_kv", "question_only_zero_memory",
            "canonical_sample_shuffled", "canonical_hard_shuffled",
        }
        if set(evaluation["conditions"]) != expected:
            errors.append("evaluation_conditions_mismatch")
        for mode in ("text", "native", "canonical", "zero"):
            training = read_json(root / f"formal512/{mode}/TRAIN_SUCCESS.json")
            if training["samples"] != 512:
                errors.append(f"formal_sample_count:{mode}")
            if (
                training["receiver_parameters_loaded"]
                or training["sender_parameters_loaded"]
                or training["writer_parameters_loaded"]
            ):
                errors.append(f"frozen_boundary:{mode}")
            if training["probe_metadata"]["layer_head_pooling"]:
                errors.append(f"forbidden_layer_head_pooling:{mode}")
    result = {
        "status": "complete" if not errors else "failed",
        "errors": errors,
        "sentence_id_source": "reconstructed_sidecar_no_KV_regeneration",
        "scope": "diagnostic_information_sufficiency_probe_not_final_reader",
    }
    write_json(root / "audit/SUCCESS.json", result)
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
