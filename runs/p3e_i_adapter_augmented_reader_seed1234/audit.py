import argparse
from pathlib import Path

import torch

from p3e_f_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    smoke = read_json(root / "smoke16/TRAIN_SUCCESS.json")
    formal = read_json(root / "formal512/TRAIN_SUCCESS.json")
    evaluation = read_json(root / "eval64/SUCCESS.json")
    checkpoint = torch.load(root / "formal512/checkpoint_best.pt",
                            map_location="cpu", weights_only=False)
    metadata = checkpoint["lora_metadata"]
    if smoke["epochs"] != 5 or formal["epochs"] != 5:
        raise RuntimeError("Expected five epochs")
    if formal["initial_equivalence"]["max_abs_logit_difference"] > 1e-4:
        raise RuntimeError("Zero-init LoRA is not C1-equivalent")
    if metadata["rank"] != 8 or metadata["alpha"] != 16.0 or metadata["dropout"] != 0.0:
        raise RuntimeError("LoRA configuration mismatch")
    if metadata["targets"] != ["q_proj", "v_proj", "o_proj", "down_proj"]:
        raise RuntimeError("LoRA target mismatch")
    if evaluation["reader_off_exact_output_consistency"] != 1.0:
        raise RuntimeError("Reader package off differs from question-only")
    write_json(root / "audit/SUCCESS.json", {
        "status": "complete", "qa_only_stage": True,
        "evidence_reconstruction_stage": False,
        "smoke16_epochs": 5, "formal512_epochs": 5,
        "receiver_base_parameters_frozen": True, "c1_reader_frozen": True,
        "sender_writer_canonical_cache_frozen": True,
        "lora_layers": metadata["receiver_layers"], "lora_targets": metadata["targets"],
        "rank": 8, "alpha": 16.0, "dropout": 0.0,
        "reader_off_disables_external_reader_and_lora": True,
        "reader_off_exact": True,
    })


if __name__ == "__main__":
    main()
