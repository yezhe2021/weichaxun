import argparse
import json
from pathlib import Path

import torch

from p2iw_common import PairCache, file_sha256, label_vocabulary, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-train", required=True)
    parser.add_argument("--native-test", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(Path(args.model) / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    train, test = PairCache(args.native_train, 1), PairCache(args.native_test, 1)
    if len(train) != 512 or len(test) != 64:
        raise ValueError(f"Expected 512/64 pairs, found {len(train)}/{len(test)}")
    labels = label_vocabulary(train, test)
    train_ids = {entry["pair_id"] for entry in train.entries}
    test_ids = {entry["pair_id"] for entry in test.entries}
    if train_ids & test_ids:
        raise ValueError("Train and test pair IDs overlap")
    sample = train.load(0)
    shape = tuple(sample["base"]["memory"]["keys"][-1].shape)
    if shape[0] != 8 or shape[-1] != 128 or shape != tuple(sample["base"]["memory"]["values"][-1].shape):
        raise ValueError(f"Unexpected final-layer Native KV shape: {shape}")
    for variant in ("base", "counterfactual"):
        row = sample[variant]
        local_shape = tuple(row["memory"]["keys"][-1].shape)
        if local_shape[1] != len(row["evidence_token_ids"]):
            raise ValueError("Native token axis and evidence token IDs disagree")
        if not row["memory"]["answer_token_mask"].any():
            raise ValueError("Answer token mask is empty")
    result = {
        "status": "complete", "train_pairs": len(train), "test_pairs": len(test),
        "labels": len(labels), "split_overlap": 0, "sample_final_kv_shape": shape,
        "model_layers": config.get("num_hidden_layers"), "model_hidden_size": config.get("hidden_size"),
        "model_kv_heads": config.get("num_key_value_heads"), "model_head_dim": config.get("head_dim"),
        "native_train_sha256": file_sha256(args.native_train),
        "native_test_sha256": file_sha256(args.native_test),
    }
    write_json(args.out, result)


if __name__ == "__main__":
    main()
