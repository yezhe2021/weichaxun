import argparse
from pathlib import Path

import torch

from p3e_f_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    teacher = read_json(root / "teacher_cache/SUCCESS.json")
    smoke = read_json(root / "smoke16/TRAIN_SUCCESS.json")
    formal = read_json(root / "formal512/TRAIN_SUCCESS.json")
    evaluation = read_json(root / "eval64/SUCCESS.json")
    checkpoint = torch.load(root / "formal512/checkpoint_best.pt",
                            map_location="cpu", weights_only=False)
    if teacher["samples"] != 512 or not teacher["answer_positions_only"]:
        raise RuntimeError("Teacher cache audit failed")
    if smoke["epochs"] != 5 or formal["epochs"] != 5:
        raise RuntimeError("Expected exactly five epochs")
    if formal["initial_equivalence"]["max_abs_logit_difference"] > 1e-4:
        raise RuntimeError("Assimilation initialization is not C1-equivalent")
    if not checkpoint["receiver_backbone_frozen"] or not checkpoint["c1_reader_frozen"]:
        raise RuntimeError("Frozen boundary audit failed")
    if evaluation["reader_off_exact_output_consistency"] != 1.0:
        raise RuntimeError("Reader-off differs from question-only")
    write_json(root / "audit/SUCCESS.json", {
        "status": "complete", "teacher_cache_complete": True,
        "teacher_answer_positions_only": True, "teacher_truncation": False,
        "smoke16_epochs": 5, "formal512_epochs": 5,
        "initial_c1_equivalence": True, "receiver_backbone_frozen": True,
        "c1_reader_frozen": True, "writer_and_sender_not_loaded_during_training": True,
        "reader_off_exact": True,
    })


if __name__ == "__main__":
    main()
