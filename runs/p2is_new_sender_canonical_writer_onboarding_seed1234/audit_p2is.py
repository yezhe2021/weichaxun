import argparse
import json
from pathlib import Path

import torch

from p2is_common import PairCache, file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--p2iw-writer", required=True); parser.add_argument("--reader4", required=True)
    parser.add_argument("--reader35", required=True); parser.add_argument("--old-train", required=True); parser.add_argument("--old-test", required=True)
    parser.add_argument("--q8-train", required=True); parser.add_argument("--q8-test", required=True); parser.add_argument("--model4", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); old_train, old_test = PairCache(args.old_train, 1), PairCache(args.old_test, 1)
    q8_train, q8_test = PairCache(args.q8_train, 1), PairCache(args.q8_test, 1)
    if (len(old_train), len(old_test), len(q8_train), len(q8_test)) != (512, 64, 512, 64): raise ValueError("P2-I-S requires exact 512/64 aligned source caches")
    if [x["pair_id"] for x in old_train.entries] != [x["pair_id"] for x in q8_train.entries]: raise RuntimeError("Old Canonical and Q8 train caches are not aligned")
    for reader_path in (args.reader4, args.reader35):
        reader = torch.load(reader_path, map_location="cpu", weights_only=False)
        if reader["writer_checkpoint_sha256"] != old_train.index["writer_checkpoint_sha256"]: raise RuntimeError("Frozen Reader is not bound to the old public Writer")
    with open(Path(args.model4) / "config.json", encoding="utf-8") as handle: model = json.load(handle)
    if int(model["num_key_value_heads"]) != 8 or int(model["head_dim"]) != 128: raise RuntimeError("Qwen3-4B Sender geometry changed")
    write_json(args.out, {
        "status": "complete", "train_pairs": 512, "test_pairs": 64, "strict_token_alignment_required": True,
        "old_writer_checkpoint_sha256": old_train.index["writer_checkpoint_sha256"],
        "p2iw_writer_file_sha256": file_sha256(args.p2iw_writer), "both_readers_frozen_interface_verified": True,
        "qwen3_4b_sender": {"kv_heads": 8, "head_dim": 128, "flat_dim": 1024},
    })


if __name__ == "__main__": main()
