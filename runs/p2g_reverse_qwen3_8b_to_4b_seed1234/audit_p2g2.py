import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig


def geometry(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    return {
        "model_type": config.model_type,
        "layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "query_heads": int(config.num_attention_heads),
        "kv_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "query_width": int(config.num_attention_heads) * int(config.head_dim),
    }


def load_index(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_pair(index_path, index, position):
    root = Path(index_path).parent
    payload = torch.load(
        root / index["pair_files"][position]["file"], map_location="cpu", weights_only=False
    )
    return {example["variant"]: example for example in payload["examples"]}


def main():
    parser = argparse.ArgumentParser(description="Audit P2-G2 8B-to-4B reverse KV communication")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--sender-train-index", required=True)
    parser.add_argument("--sender-test-index", required=True)
    parser.add_argument("--teacher-train-index", required=True)
    parser.add_argument("--teacher-test-index", required=True)
    parser.add_argument("--native-gate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sender_geometry = geometry(args.sender_model)
    receiver_geometry = geometry(args.receiver_model)
    expected_sender = {
        "model_type": "qwen3",
        "layers": 36,
        "hidden_size": 4096,
        "query_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "query_width": 4096,
    }
    expected_receiver = {
        "model_type": "qwen3",
        "layers": 36,
        "hidden_size": 2560,
        "query_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "query_width": 4096,
    }
    if sender_geometry != expected_sender:
        raise RuntimeError(f"Unexpected sender geometry: {sender_geometry}")
    if receiver_geometry != expected_receiver:
        raise RuntimeError(f"Unexpected receiver geometry: {receiver_geometry}")

    checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=False)
    reader_args = checkpoint["args"]
    if reader_args.get("model") != args.receiver_model:
        raise RuntimeError("Reader checkpoint is not the Qwen3-4B Native Reader")
    if reader_args.get("config_name") != "query_only" or int(reader_args.get("query_rank", -1)) != 32:
        raise RuntimeError("Reader checkpoint is not rank-32 Query-only")
    query_down = checkpoint["adapter"]["query_down.0.weight"]
    query_up = checkpoint["adapter"]["query_up.0.weight"]
    if tuple(query_down.shape) != (32, 4096) or tuple(query_up.shape) != (4096, 32):
        raise RuntimeError(f"Unexpected Query adapter shapes: {query_down.shape}, {query_up.shape}")

    split_reports = {}
    for split, sender_path, teacher_path, expected_pairs in (
        ("train", args.sender_train_index, args.teacher_train_index, 512),
        ("test", args.sender_test_index, args.teacher_test_index, 64),
    ):
        sender_index = load_index(sender_path)
        teacher_index = load_index(teacher_path)
        if sender_index.get("model") != args.sender_model:
            raise RuntimeError(f"{split} sender cache model mismatch")
        if teacher_index.get("model") != args.receiver_model:
            raise RuntimeError(f"{split} teacher cache model mismatch")
        sender_entries = sender_index["pair_files"]
        teacher_entries = teacher_index["pair_files"]
        if len(sender_entries) != expected_pairs or len(teacher_entries) != expected_pairs:
            raise RuntimeError(
                f"{split} pair count mismatch: sender={len(sender_entries)} teacher={len(teacher_entries)}"
            )
        sender_ids = [entry["pair_id"] for entry in sender_entries]
        teacher_ids = [entry["pair_id"] for entry in teacher_entries]
        if sender_ids != teacher_ids:
            raise RuntimeError(f"{split} cache pair IDs are not aligned")
        sampled = sorted({0, expected_pairs // 2, expected_pairs - 1})
        for position in sampled:
            sender_pair = load_pair(sender_path, sender_index, position)
            teacher_pair = load_pair(teacher_path, teacher_index, position)
            for variant in ("base", "counterfactual"):
                if sender_pair[variant]["evidence_token_ids"] != teacher_pair[variant]["evidence_token_ids"]:
                    raise RuntimeError(f"{split} token mismatch at pair {position} variant {variant}")
        split_reports[split] = {
            "pairs": expected_pairs,
            "pair_ids_aligned": True,
            "sampled_token_identity_positions": sampled,
        }

    with open(args.native_gate, encoding="utf-8") as handle:
        native_gate = json.load(handle)
    report = {
        "status": "passed",
        "sender_model": args.sender_model,
        "receiver_model": args.receiver_model,
        "sender_geometry": sender_geometry,
        "receiver_geometry": receiver_geometry,
        "reader": {
            "checkpoint": args.reader_checkpoint,
            "frozen": True,
            "query_rank": 32,
            "native_gate_status": native_gate["status"],
            "native_paired_consistency": native_gate["native_kv"]["paired_consistency"],
            "native_gate_override": native_gate["status"] != "passed",
        },
        "splits": split_reports,
        "claim_boundary": "Writer output remains specific to the frozen Qwen3-4B Reader interface.",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
