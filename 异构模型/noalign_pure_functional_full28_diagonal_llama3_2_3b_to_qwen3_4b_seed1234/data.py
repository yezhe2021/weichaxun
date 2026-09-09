from pathlib import Path

import torch

from common import read_json, run_root, save_json, log


def source_cfg(cfg):
    return read_json(Path(cfg["source_root"]) / "run_config.json")


def pair_source_cfg(cfg):
    return read_json(Path(cfg["pair_source_root"]) / "run_config.json")


def rows(cfg, split):
    payload = read_json(Path(cfg["source_root"]) / "manifests" / f"{split}.json")
    count = cfg[f"{split}_samples"]
    if payload["signature"] != source_cfg(cfg)["signature"] or len(payload["rows"]) < count:
        raise RuntimeError(f"Invalid source manifest: {split}")
    return payload["rows"][:count]


def pair_path(cfg, split, row):
    return Path(cfg["pair_source_root"]) / "pair_cache" / split / f"{row['id']}.pt"


def load_noalign(cfg, split, row):
    """Expose no aligned content target KV: only source content, native token0, and full-Qwen logits."""
    raw = torch.load(pair_path(cfg, split, row), map_location="cpu", weights_only=True)
    if raw["signature"] != pair_source_cfg(cfg)["signature"]:
        raise RuntimeError("Reused pair cache signature mismatch")
    metadata = raw["metadata"]
    source_indices = metadata["source_indices"][1:]
    if len(source_indices) != 32 or any(a > b for a, b in zip(source_indices, source_indices[1:])):
        raise RuntimeError("Source memories are not in stable source-text order")
    # Deliberately never expose target_k/target_v positions 1..32.
    return {
        "id": metadata["id"], "gold_index": metadata["gold_index"],
        "source_indices": source_indices,
        "source_k": raw["source_k"][:, 1:].contiguous(),
        "source_v": raw["source_v"][:, 1:].contiguous(),
        "qwen_native_token0_k": raw["target_k"][:, :1].contiguous(),
        "qwen_native_token0_v": raw["target_v"][:, :1].contiguous(),
        "qwen_suffix": metadata["qwen_suffix"],
        "qwen_choice_ids": metadata["qwen_choice_ids"],
        "teacher_full_logits": raw["qwen_full_native_logits"],
    }


def protocol_audit(cfg):
    report = {
        "experiment": "NoAlign RandomInit Full28 Diagonal Pure Functional",
        "cross_tokenizer_alignment_used_by_training": False,
        "qwen_content_target_kv_exposed": False,
        "stage_a": False,
        "gold_supervision": False,
        "teacher": "Qwen full Context+Query final logits",
        "student_memory": "Qwen Native token0 + 32 ordered functional memories",
        "canonical_positions": list(range(33)),
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        selected = rows(cfg, split)
        duplicate_counts = []
        for row in selected:
            item = load_noalign(cfg, split, row)
            if tuple(item["source_k"].shape) != (28, 32, 8, 128):
                raise RuntimeError("Bad Llama source KV shape")
            if tuple(item["qwen_native_token0_k"].shape) != (36, 1, 8, 128):
                raise RuntimeError("Bad Qwen token0 shape")
            duplicate_counts.append(32 - len(set(item["source_indices"])))
        report["splits"][split] = {
            "count": len(selected), "ordered": True,
            "samples_with_duplicate_source_tokens": sum(x > 0 for x in duplicate_counts),
            "max_duplicate_entries": max(duplicate_counts),
        }
        log(f"NoAlign audit {split}: {len(selected)} samples")
    save_json(run_root(cfg) / "audit" / "protocol_audit.json", report)
    return report
