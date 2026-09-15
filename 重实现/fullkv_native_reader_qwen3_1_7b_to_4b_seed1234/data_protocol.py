from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from common import SYSTEM, load_json, load_tokenizer, progress, save_json, sha256_ints, write_jsonl


def context_text(row: dict[str, Any]) -> str:
    blocks = []
    for index, (title, sentences) in enumerate(row["context"], 1):
        blocks.append(f"[{index}] {title}\n{''.join(sentences)}")
    return "\n\n".join(blocks)


def serialize(row: dict[str, Any]) -> tuple[str, str, str]:
    prefix = f"{SYSTEM}\n\nContext:\n{context_text(row)}\n\n"
    suffix = f"Question: {row['question']}\n\nAnswer:"
    question_only = f"{SYSTEM}\n\nQuestion: {row['question']}\n\nAnswer:"
    return prefix, suffix, question_only


def encode_pair(tokenizer17, tokenizer4, row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    prefix, suffix, question_only = serialize(row)
    encoded: dict[str, list[int]] = {}
    for label, text in (("prefix", prefix), ("suffix", suffix), ("question_only", question_only)):
        ids17 = list(tokenizer17(text, add_special_tokens=False).input_ids)
        ids4 = list(tokenizer4(text, add_special_tokens=False).input_ids)
        if ids17 != ids4:
            raise RuntimeError(f"tokenizer mismatch for sample={row['_id']} field={label}")
        encoded[label] = ids4

    combined17 = list(tokenizer17(prefix + suffix, add_special_tokens=False).input_ids)
    combined4 = list(tokenizer4(prefix + suffix, add_special_tokens=False).input_ids)
    composed = encoded["prefix"] + encoded["suffix"]
    if combined17 != combined4 or combined4 != composed:
        raise RuntimeError(
            f"non-compositional prefix/suffix tokenization for sample={row['_id']}; "
            "the cache and full-text conditions would not be token-identical"
        )

    answer17 = list(tokenizer17(row["answer"], add_special_tokens=False).input_ids)
    answer4 = list(tokenizer4(row["answer"], add_special_tokens=False).input_ids)
    if answer17 != answer4:
        raise RuntimeError(f"answer tokenizer mismatch for sample={row['_id']}")
    answer = answer4[: cfg["max_answer_tokens"]]
    if not answer:
        answer = [tokenizer4.eos_token_id]
    if answer[-1] != tokenizer4.eos_token_id:
        answer.append(tokenizer4.eos_token_id)

    return {
        "id": row["_id"],
        "type": row.get("type", "unknown"),
        "question": row["question"],
        "answer": row["answer"],
        "titles": [title for title, _ in row["context"]],
        "prefix_text": prefix,
        "suffix_text": suffix,
        "prefix_token_ids": encoded["prefix"],
        "question_suffix_ids": encoded["suffix"],
        "question_only_ids": encoded["question_only"],
        "full_prompt_ids": composed,
        "answer_token_ids": answer,
        "context_length": len(encoded["prefix"]),
        "prefix_token_sha256": sha256_ints(encoded["prefix"]),
        "tokenization_equal_1_7b_4b": True,
        "prefix_suffix_composition_equal": True,
    }


def balanced_candidates(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups = {"bridge": [], "comparison": []}
    for row in rows:
        if row.get("type") in groups:
            groups[row["type"]].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    output = []
    while groups["bridge"] or groups["comparison"]:
        for kind in ("bridge", "comparison"):
            if groups[kind]:
                output.append(groups[kind].pop())
    return output


def select_rows(raw_rows, tokenizer17, tokenizer4, count: int, cfg: dict[str, Any], seed: int):
    selected, rejected = [], {"too_long": 0}
    for row in balanced_candidates(raw_rows, seed):
        # Tokenizer disagreement is a protocol failure, not a sample-filtering rule.
        encoded = encode_pair(tokenizer17, tokenizer4, row, cfg)
        if encoded["context_length"] > cfg["max_prefix_tokens"] or len(encoded["question_suffix_ids"]) > cfg["max_suffix_tokens"]:
            rejected["too_long"] += 1
            continue
        selected.append(encoded)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only selected {len(selected)}/{count} full, untruncated samples; rejected={rejected}")
    return selected, rejected


def prepare(cfg: dict[str, Any]) -> None:
    tokenizer17 = load_tokenizer(cfg["model_1_7b"])
    tokenizer4 = load_tokenizer(cfg["model_4b"])
    train_raw = load_json(cfg["hotpot_train"])
    test_raw = load_json(cfg["hotpot_dev"])
    train_count = cfg["splits"]["train"] + cfg["splits"]["validation"]
    train_and_validation, rejected_train = select_rows(
        train_raw, tokenizer17, tokenizer4, train_count, cfg, cfg["seed"]
    )
    test, rejected_test = select_rows(
        test_raw, tokenizer17, tokenizer4, cfg["splits"]["test"], cfg, cfg["seed"] + 1
    )
    split = cfg["splits"]["train"]
    manifests = {
        "train": train_and_validation[:split],
        "validation": train_and_validation[split:],
        "test": test,
    }
    root = Path(cfg["work_dir"]) / "artifacts" / "manifests"
    for name, rows in manifests.items():
        write_jsonl(root / f"{name}.jsonl", rows)
    save_json(root / "summary.json", {
        "counts": {name: len(rows) for name, rows in manifests.items()},
        "rejected_train_pool": rejected_train,
        "rejected_test_pool": rejected_test,
        "uses_supporting_facts": False,
        "uses_selected_tokens": False,
        "all_context_documents_included": True,
        "truncation_used": False,
        "tokenizers_equal_on_all_serialized_fields": True,
    })
    progress("prepared full-context manifests")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare(load_json(args.config))


if __name__ == "__main__":
    main()
