import argparse
from pathlib import Path

from p3e_f_common import read_json, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    data1024 = read_jsonl(root / "data/train1024.jsonl")
    data2048 = read_jsonl(root / "data/train2048.jsonl")
    if [row["id"] for row in data1024] != [row["id"] for row in data2048[:1024]]:
        raise RuntimeError("Training datasets are not nested")
    train1024 = read_json(root / "train1024/TRAIN_SUCCESS.json")
    train2048 = read_json(root / "train2048/TRAIN_SUCCESS.json")
    if train1024["init_reader_sha256"] != train2048["init_reader_sha256"]:
        raise RuntimeError("Scale runs did not start from the same C1 checkpoint")
    if train1024["validation_used_for_selection"] or train2048["validation_used_for_selection"]:
        raise RuntimeError("Validation was used during checkpoint selection")
    eval1024 = read_json(root / "eval1024/SUCCESS.json")
    eval2048 = read_json(root / "eval2048/SUCCESS.json")
    if eval1024["validation_samples"] != 64 or eval2048["validation_samples"] != 64:
        raise RuntimeError("Fixed validation size is not 64")
    result = {
        "status": "complete", "nested_data": True, "same_initial_checkpoint": True,
        "historical512_retrained": False, "validation_used_once_after_training": True,
        "train_scales": [1024, 2048], "validation_samples": 64,
    }
    write_json(root / "audit/SUCCESS.json", result)


if __name__ == "__main__":
    main()
