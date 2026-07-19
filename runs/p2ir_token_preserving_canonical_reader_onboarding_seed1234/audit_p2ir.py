import argparse
import json
from pathlib import Path

from p2ir_common import PairCache, file_sha256, write_json


def config(path):
    with open(Path(path) / "config.json", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2iw-root", required=True); parser.add_argument("--writer-checkpoint", required=True)
    parser.add_argument("--token-train", required=True); parser.add_argument("--token-test", required=True)
    parser.add_argument("--receiver4", required=True); parser.add_argument("--receiver35", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    p2iw_success = Path(args.p2iw_root) / "SUCCESS.json"
    if not p2iw_success.is_file():
        raise FileNotFoundError("P2-I-W final summary is missing")
    train, test = PairCache(args.token_train, 1), PairCache(args.token_test, 1)
    if len(train) != 512 or len(test) != 64:
        raise ValueError("P2-I-R requires exact 512/64 P2-I-W token caches")
    if {entry["pair_id"] for entry in train.entries} & {entry["pair_id"] for entry in test.entries}:
        raise ValueError("Train/test pair IDs overlap")
    sample = train.load(0)
    for variant in ("base", "counterfactual"):
        row = sample[variant]
        if row["key_flat"].shape[-1] != 1024 or row["value_flat"].shape != row["key_flat"].shape:
            raise ValueError("Unexpected P2-I-W flattened Native KV geometry")
    c4, c35 = config(args.receiver4), config(args.receiver35)
    write_json(args.out, {
        "status": "complete", "train_pairs": 512, "test_pairs": 64,
        "writer_checkpoint_sha256": file_sha256(args.writer_checkpoint),
        "qwen3_4b": {"layers": c4.get("num_hidden_layers"), "hidden_size": c4.get("hidden_size")},
        "qwen3_5_4b": {
            "layers": c35.get("num_hidden_layers", c35.get("text_config", {}).get("num_hidden_layers")),
            "hidden_size": c35.get("hidden_size", c35.get("text_config", {}).get("hidden_size")),
            "layer_types": c35.get("layer_types", c35.get("text_config", {}).get("layer_types")),
        },
    })


if __name__ == "__main__":
    main()
