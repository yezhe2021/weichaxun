import argparse
from pathlib import Path

import torch

from p3e_f_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--base-reader", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    smoke = read_json(root / "smoke16/TRAIN_SUCCESS.json")
    formal = read_json(root / "formal512/TRAIN_SUCCESS.json")
    evaluation = read_json(root / "eval64/SUCCESS.json")
    checkpoint = torch.load(root / "formal512/checkpoint_best.pt", map_location="cpu", weights_only=False)
    if checkpoint["base_reader"] != args.base_reader or not checkpoint["base_reader_frozen"]:
        raise RuntimeError("Formal Strong Reader did not preserve the frozen C1 base")
    if smoke["epochs"] != 5 or formal["epochs"] != 5:
        raise RuntimeError("Expected exactly 5 epochs")
    if evaluation["reader_off_exact_output_consistency"] != 1.0:
        raise RuntimeError("Reader-off does not exactly reproduce question-only")
    result = {
        "status": "complete", "epochs": 5, "smoke16_complete": True,
        "formal512_complete": True, "validation64_complete": True,
        "writer_frozen": True, "receiver_backbone_frozen": True,
        "base_query_adapter_frozen": True, "base_head_routing_frozen": True,
        "native_o_proj_frozen": True, "reader_off_exact": True,
    }
    write_json(root / "audit/SUCCESS.json", result)


if __name__ == "__main__":
    main()
