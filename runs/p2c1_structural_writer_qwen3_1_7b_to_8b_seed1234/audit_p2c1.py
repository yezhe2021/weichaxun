import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from p2a_common import load_jsonl, sender_text


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def model_geometry(model_path):
    with open(Path(model_path) / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    return {
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "layers": config.get("num_hidden_layers"),
        "query_heads": config.get("num_attention_heads"),
        "kv_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "vocab_size": config.get("vocab_size"),
        "rope_theta": config.get("rope_theta"),
        "rope_scaling": config.get("rope_scaling"),
        "config_sha256": file_hash(Path(model_path) / "config.json"),
    }


def tokenizer_metadata(tokenizer):
    special = {
        name: getattr(tokenizer, name, None)
        for name in (
            "bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
            "additional_special_tokens_ids",
        )
    }
    return {
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "vocab_sha256": stable_hash(tokenizer.get_vocab()),
        "special_token_ids": special,
        "chat_template_sha256": sha256_bytes((tokenizer.chat_template or "").encode("utf-8")),
        "is_fast": bool(tokenizer.is_fast),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Qwen3-1.7B to Qwen3-8B P2-C1 compatibility")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    sender_geometry = model_geometry(args.sender_model)
    receiver_geometry = model_geometry(args.receiver_model)
    sender_tokenizer = AutoTokenizer.from_pretrained(
        args.sender_model, trust_remote_code=True, local_files_only=True
    )
    receiver_tokenizer = AutoTokenizer.from_pretrained(
        args.receiver_model, trust_remote_code=True, local_files_only=True
    )
    rows = load_jsonl(args.data, args.max_samples)
    exact_tokens = 0
    exact_offsets = 0
    aligned_token_positions = 0
    total_token_positions = 0
    for row in rows:
        text, _ = sender_text(row)
        sender = sender_tokenizer(
            text, add_special_tokens=True, return_offsets_mapping=True
        )
        receiver = receiver_tokenizer(
            text, add_special_tokens=True, return_offsets_mapping=True
        )
        sender_ids = sender.input_ids
        receiver_ids = receiver.input_ids
        sender_offsets = [tuple(value) for value in sender.offset_mapping]
        receiver_offsets = [tuple(value) for value in receiver.offset_mapping]
        exact_tokens += int(sender_ids == receiver_ids)
        exact_offsets += int(sender_offsets == receiver_offsets)
        common = min(len(sender_ids), len(receiver_ids))
        aligned_token_positions += sum(
            sender_ids[index] == receiver_ids[index]
            and sender_offsets[index] == receiver_offsets[index]
            for index in range(common)
        )
        total_token_positions += max(len(sender_ids), len(receiver_ids))

    report = {
        "status": "complete",
        "args": vars(args),
        "sender_geometry": sender_geometry,
        "receiver_geometry": receiver_geometry,
        "sender_tokenizer": tokenizer_metadata(sender_tokenizer),
        "receiver_tokenizer": tokenizer_metadata(receiver_tokenizer),
        "alignment": {
            "samples": len(rows),
            "exact_token_sequence_rate": exact_tokens / max(1, len(rows)),
            "exact_offset_sequence_rate": exact_offsets / max(1, len(rows)),
            "token_and_offset_position_rate": aligned_token_positions / max(1, total_token_positions),
        },
        "raw_kv_shape_compatible_by_config": (
            sender_geometry["layers"] == receiver_geometry["layers"]
            and sender_geometry["kv_heads"] == receiver_geometry["kv_heads"]
            and sender_geometry["head_dim"] == receiver_geometry["head_dim"]
        ),
        "reader_checkpoint_sha256": file_hash(args.reader_checkpoint),
        "coordinate_contract": "pre_rope_k_native_v_evidence_tokens_only",
        "claim_scope": (
            "Qwen3-1.7B sender-specific Writer into a frozen Qwen3-8B Reader-compatible "
            "per-layer KV interface; not receiver-independent Canonical Evidence-KV."
        ),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
