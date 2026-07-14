import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from p2a_common import load_jsonl, sender_text


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_geometry(model_path):
    with open(Path(model_path) / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    return {
        "hidden_size": config.get("hidden_size"),
        "layers": config.get("num_hidden_layers"),
        "attention_heads": config.get("num_attention_heads"),
        "kv_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "vocab_size": config.get("vocab_size"),
        "rope_theta": config.get("rope_theta"),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit P2-B sender/receiver geometry and token alignment")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()

    sender_geometry = model_geometry(args.sender_model)
    receiver_geometry = model_geometry(args.receiver_model)
    sender_tokenizer = AutoTokenizer.from_pretrained(
        args.sender_model, trust_remote_code=True, local_files_only=True
    )
    receiver_tokenizer = AutoTokenizer.from_pretrained(
        args.receiver_model, trust_remote_code=True, local_files_only=True
    )
    rows = load_jsonl(args.data, args.samples)
    aligned = 0
    for row in rows:
        text, _ = sender_text(row)
        sender_ids = sender_tokenizer(text, add_special_tokens=True).input_ids
        receiver_ids = receiver_tokenizer(text, add_special_tokens=True).input_ids
        aligned += int(sender_ids == receiver_ids)

    report = {
        "status": "complete",
        "args": vars(args),
        "sender_geometry": sender_geometry,
        "receiver_geometry": receiver_geometry,
        "raw_kv_shape_compatible": (
            sender_geometry["layers"] == receiver_geometry["layers"]
            and sender_geometry["kv_heads"] == receiver_geometry["kv_heads"]
            and sender_geometry["head_dim"] == receiver_geometry["head_dim"]
        ),
        "sample_token_alignment_rate": aligned / max(1, len(rows)),
        "reader_checkpoint_sha256": file_hash(args.reader_checkpoint),
        "claim_scope": (
            "Frozen Qwen3-8B Reader-compatible per-layer KV sender integration; "
            "not receiver-independent Canonical Evidence-KV."
        ),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
