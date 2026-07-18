import argparse
import json
from pathlib import Path

import torch

from p2id_common import add_p2i_path, file_sha256, write_json

add_p2i_path()
from p2i_common import LazyPairCache


def main():
    parser = argparse.ArgumentParser(description="Static prerequisite audit for P2-I-D")
    parser.add_argument("--p2i-root", required=True)
    parser.add_argument("--mother-checkpoint", required=True)
    parser.add_argument("--canonical-train-index", required=True)
    parser.add_argument("--canonical-test-index", required=True)
    parser.add_argument("--native-train-index", required=True)
    parser.add_argument("--native-test-index", required=True)
    parser.add_argument("--receiver4-model", required=True)
    parser.add_argument("--receiver35-model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    for path in (
        args.mother_checkpoint,
        args.canonical_train_index,
        args.canonical_test_index,
        args.native_train_index,
        args.native_test_index,
        str(Path(args.receiver4_model) / "config.json"),
        str(Path(args.receiver35_model) / "config.json"),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    canonical_train = LazyPairCache(args.canonical_train_index, capacity=1)
    canonical_test = LazyPairCache(args.canonical_test_index, capacity=1)
    native_train = LazyPairCache(args.native_train_index, capacity=1)
    native_test = LazyPairCache(args.native_test_index, capacity=1)
    if len(canonical_train) != 512 or len(native_train) != 512:
        raise ValueError("P2-I-D requires exactly 512 cached training pairs")
    if len(canonical_test) != 64 or len(native_test) != 64:
        raise ValueError("P2-I-D requires exactly 64 cached test pairs")
    for left, right, name in (
        (canonical_train, native_train, "train"),
        (canonical_test, native_test, "test"),
    ):
        if [entry["pair_id"] for entry in left.entries] != [entry["pair_id"] for entry in right.entries]:
            raise ValueError(f"Canonical and Native {name} caches are not aligned")
    checkpoint = torch.load(args.mother_checkpoint, map_location="cpu", weights_only=False)
    writer_hash = checkpoint.get("writer_sha256")
    if not writer_hash or canonical_train.index.get("writer_sha256") != writer_hash:
        raise ValueError("Canonical train cache does not match the mother Writer hash")
    if canonical_test.index.get("writer_sha256") != writer_hash:
        raise ValueError("Canonical test cache does not match the mother Writer hash")
    report = {
        "status": "complete",
        "split": {"train": 448, "validation": 64, "test": 64},
        "canonical_train_pairs": len(canonical_train),
        "canonical_test_pairs": len(canonical_test),
        "mother_checkpoint_sha256": file_sha256(args.mother_checkpoint),
        "canonical_writer_sha256": writer_hash,
        "p2i_dependency": str(Path(args.p2i_root).resolve()),
    }
    write_json(args.out, report)


if __name__ == "__main__":
    main()
