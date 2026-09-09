from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common import fixed_samples, load_json, load_tokenizer, progress, read_jsonl, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    tok = load_tokenizer(cfg)
    rows = fixed_samples(cfg, tok)
    reference_root = Path(cfg["reference_4b_dir"]) / "artifacts"
    reference_manifest = load_json(reference_root / "dataset_manifest.json")
    reference_rows = read_jsonl(reference_root / "rendered_samples.jsonl")
    same_ids = [row["id"] for row in rows] == reference_manifest["ids"]
    same_tokens = all(
        row["id"] == reference["id"] and row["input_ids"] == reference["input_ids"]
        for row, reference in zip(rows, reference_rows)
    ) and len(rows) == len(reference_rows)
    if not same_ids or not same_tokens:
        raise RuntimeError("Qwen3-1.7B audit does not match the frozen Qwen3-4B samples/tokens")
    root = Path(cfg["work_dir"]) / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    rendered = root / "rendered_samples.jsonl"
    with rendered.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audits = [{
        "sample_id": row["id"], "type": row["type"],
        "split_equal": row["split_equal"],
        "full_tokens": len(row["input_ids"]),
        "context_tokens": len(row["context_input_ids"]),
        "question_suffix_tokens": len(row["question_input_ids"]),
        "context_end_index": row["context_end_index"],
        "question_char_index": row["question_char_index"],
        "boundary_token_offset": row["boundary_token_offset"],
        "boundary_crosses_token": row["boundary_token_offset"][0] < row["question_char_index"],
    } for row in rows]
    with (root / "token_split_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(rendered.read_bytes()).hexdigest()
    save_json(root / "dataset_manifest.json", {
        "seed": cfg["seed"], "dataset": cfg["hotpot_dev"],
        "selection": "seeded balanced 32 bridge / 32 comparison",
        "count": len(rows), "ids": [row["id"] for row in rows],
        "rendered_samples_sha256": digest,
        "reference_4b_manifest": str(reference_root / "dataset_manifest.json"),
        "same_sample_ids_as_4b": same_ids,
        "same_full_input_ids_as_4b": same_tokens,
        "system_message": __import__("common").SYSTEM,
        "chat_template": "model-native; enable_thinking=False; add_generation_prompt=True",
        "decoding": {"do_sample": False, "temperature": 0, "max_new_tokens": cfg["max_new_tokens"]},
    })
    if not all(row["split_equal"] for row in rows):
        raise RuntimeError("token split concatenation audit failed")
    progress(f"rendered and froze {len(rows)} full-context samples")


if __name__ == "__main__":
    main()
