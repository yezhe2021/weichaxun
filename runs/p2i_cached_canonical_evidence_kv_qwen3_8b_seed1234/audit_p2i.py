import argparse
import json
from pathlib import Path

import torch

from canonical_modules import CanonicalEvidenceWriter, permute_slots
from p2i_common import LazyPairCache, file_sha256, load_jsonl


def model_geometry(path):
    with open(Path(path) / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    text = config.get("text_config", config)
    types = text.get("layer_types", [])
    return {
        "model_type": config.get("model_type"),
        "layers": int(text["num_hidden_layers"]),
        "hidden_size": int(text["hidden_size"]),
        "query_heads": int(text["num_attention_heads"]),
        "kv_heads": int(text["num_key_value_heads"]),
        "head_dim": int(text["head_dim"]),
        "full_attention_layers": [index for index, value in enumerate(types) if value == "full_attention"]
        if types else list(range(int(text["num_hidden_layers"]))),
    }


def smoke_interface():
    writer = CanonicalEvidenceWriter(3, 2, 8, slots=16, canonical_dim=32, atom_dim=8)
    memory = {
        "keys": [torch.randn(2, 5, 8) for _ in range(3)],
        "values": [torch.randn(2, 5, 8) for _ in range(3)],
        "answer_token_mask": torch.tensor([False, False, True, False, False]),
    }
    output = writer(memory, return_diagnostics=True)
    assert output["keys"].shape == (16, 32)
    assert output["values"].shape == (16, 32)
    assert output["answer_slot_mass"].shape == (16,)
    loss = output["keys"].square().mean() + output["values"].square().mean()
    loss.backward()
    if not any(parameter.grad is not None for parameter in writer.parameters()):
        raise RuntimeError("Writer smoke test produced no gradients")
    permutation = torch.randperm(16)
    moved = permute_slots(output, permutation)
    query = torch.randn(4, 32)
    left = (query @ output["keys"].T / 32**0.5).softmax(-1) @ output["values"]
    right = (query @ moved["keys"].T / 32**0.5).softmax(-1) @ moved["values"]
    difference = float((left - right).abs().max())
    if difference > 1e-5:
        raise RuntimeError(f"Slot permutation invariant failed: {difference}")
    return difference


def main():
    parser = argparse.ArgumentParser(description="Static audit for P2-I cached canonical interface")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver4-model", required=True)
    parser.add_argument("--receiver35-model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--sender-train-index", required=True)
    parser.add_argument("--sender-test-index", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sender = model_geometry(args.sender_model)
    receiver4 = model_geometry(args.receiver4_model)
    receiver35 = model_geometry(args.receiver35_model)
    expected_sender = {"layers": 36, "kv_heads": 8, "head_dim": 128}
    if any(sender[key] != value for key, value in expected_sender.items()):
        raise ValueError(f"Unexpected Qwen3-8B sender geometry: {sender}")
    if len(receiver4["full_attention_layers"]) != 36:
        raise ValueError("Qwen3-4B must expose 36 full-attention layers")
    if receiver35["full_attention_layers"] != [3, 7, 11, 15, 19, 23, 27, 31]:
        raise ValueError(f"Unexpected Qwen3.5 full-attention layout: {receiver35}")
    train_rows = load_jsonl(args.train_data)
    test_rows = load_jsonl(args.test_data)
    if len(train_rows) < 1024 or len(test_rows) < 128:
        raise ValueError("P2-I requires at least 512 train and 64 test counterfactual pairs")
    train_cache = LazyPairCache(args.sender_train_index)
    test_cache = LazyPairCache(args.sender_test_index)
    if len(train_cache) < 512 or len(test_cache) < 64:
        raise ValueError("Sender Native-KV cache is incomplete")

    report = {
        "status": "complete",
        "interface": {"slots": 256, "canonical_dim": 256, "receiver_axes": False},
        "sender": sender,
        "receiver_qwen3_4b": receiver4,
        "receiver_qwen3_5_4b": receiver35,
        "data": {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_pairs_used": 448,
            "validation_pairs_reserved": 64,
            "test_pairs": 64,
            "train_sha256": file_sha256(args.train_data),
            "test_sha256": file_sha256(args.test_data),
        },
        "sender_cache": {"train_pairs": len(train_cache), "test_pairs": len(test_cache)},
        "slot_permutation_smoke_max_difference": smoke_interface(),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
