import argparse
import json
from pathlib import Path

import torch


def load_index(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def first_example(index_path, index):
    root = Path(index_path).parent
    payload = torch.load(root / index["pair_files"][0]["file"], map_location="cpu", weights_only=False)
    return payload["examples"][0]


def describe(example):
    return {
        "layers": len(example["memory"]["keys"]),
        "key_shape_first_layer": list(example["memory"]["keys"][0].shape),
        "value_shape_first_layer": list(example["memory"]["values"][0].shape),
        "evidence_tokens": example["evidence_tokens"],
        "answer_mask_tokens": int(example["memory"]["answer_token_mask"].sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify actual P2-C1 cached KV tensor geometry")
    parser.add_argument("--sender-index", required=True)
    parser.add_argument("--teacher-index", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sender_index = load_index(args.sender_index)
    teacher_index = load_index(args.teacher_index)
    sender_example = first_example(args.sender_index, sender_index)
    teacher_example = first_example(args.teacher_index, teacher_index)
    if sender_example["pair_id"] != teacher_example["pair_id"]:
        raise ValueError("Sender and teacher first cached pairs do not match")
    report = {
        "status": "complete",
        "sender_index_geometry": {
            key: sender_index[key] for key in ("layers", "query_heads", "kv_heads", "head_dim")
        },
        "teacher_index_geometry": {
            key: teacher_index[key] for key in ("layers", "query_heads", "kv_heads", "head_dim")
        },
        "sender_actual_tensor": describe(sender_example),
        "teacher_actual_tensor": describe(teacher_example),
        "evidence_token_ids_equal": (
            sender_example.get("evidence_token_ids") == teacher_example.get("evidence_token_ids")
        ),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
