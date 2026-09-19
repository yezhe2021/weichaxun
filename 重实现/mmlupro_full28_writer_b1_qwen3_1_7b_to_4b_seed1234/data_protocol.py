from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from common import LABELS, choice_instruction, load_json, load_tokenizer, progress, read_jsonl, save_json, sha256_ints, stable_rank, write_jsonl


def options_block(options: list[str]) -> str:
    if not 2 <= len(options) <= len(LABELS):
        raise RuntimeError(f"MMLU-Pro sample has unsupported option count={len(options)}")
    return "Candidate answers:\n" + "\n".join(f"{label}. {text}" for label, text in zip(LABELS, options))


def serialize(row: dict[str, Any]) -> tuple[str, str, str, str]:
    options = options_block(row["options"])
    instruction = choice_instruction(LABELS[len(row["options"]) - 1])
    prefix = options + "\n\n"
    suffix = f"Question:\n{row['question']}\n\n{instruction}\nAnswer:"
    standard = f"Question:\n{row['question']}\n\n{options}\n\n{instruction}\nAnswer:"
    question_only = f"Question:\n{row['question']}\n\n{instruction}\nAnswer:"
    return prefix, suffix, standard, question_only


def label_token_ids(tokenizer, labels: list[str]) -> list[int]:
    ids = []
    for label in labels:
        encoded = list(tokenizer(" " + label, add_special_tokens=False).input_ids)
        if len(encoded) != 1:
            raise RuntimeError(f"label {label!r} is not one token after 'Answer:': ids={encoded}")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"A-J label token IDs are not unique: {ids}")
    return ids


def encode_sample(tokenizer17, tokenizer4, row: dict[str, Any], cfg: dict[str, Any], label_ids: list[int]) -> dict[str, Any]:
    prefix, suffix, standard, question_only = serialize(row)
    texts = {"prefix": prefix, "suffix": suffix, "standard": standard, "question_only": question_only}
    encoded: dict[str, list[int]] = {}
    for name, text in texts.items():
        ids17 = list(tokenizer17(text, add_special_tokens=False).input_ids)
        ids4 = list(tokenizer4(text, add_special_tokens=False).input_ids)
        if ids17 != ids4:
            raise RuntimeError(f"tokenizer mismatch sample={row['question_id']} field={name}")
        encoded[name] = ids4
    full17 = list(tokenizer17(prefix + suffix, add_special_tokens=False).input_ids)
    full4 = list(tokenizer4(prefix + suffix, add_special_tokens=False).input_ids)
    composed = encoded["prefix"] + encoded["suffix"]
    if full17 != full4 or full4 != composed:
        raise RuntimeError(f"non-compositional Options/Question boundary sample={row['question_id']}")
    answer_index = int(row["answer_index"])
    if not 0 <= answer_index < len(row["options"]):
        raise RuntimeError(f"invalid answer_index={answer_index} sample={row['question_id']}")
    return {
        "id": str(row["question_id"]),
        "question_id": int(row["question_id"]),
        "category": row["category"],
        "source": row["src"],
        "question": row["question"],
        "options": row["options"],
        "num_options": len(row["options"]),
        "valid_labels": list(LABELS[:len(row["options"])]),
        "gold_index": answer_index,
        "gold_label": LABELS[answer_index],
        "prefix_text": prefix,
        "suffix_text": suffix,
        "standard_prompt_text": standard,
        "question_only_text": question_only,
        "prefix_token_ids": encoded["prefix"],
        "question_suffix_ids": encoded["suffix"],
        "standard_prompt_ids": encoded["standard"],
        "question_only_ids": encoded["question_only"],
        "full_prompt_ids": composed,
        "label_token_ids": label_ids[:len(row["options"])],
        "context_length": len(encoded["prefix"]),
        "suffix_length": len(encoded["suffix"]),
        "prefix_token_sha256": sha256_ints(encoded["prefix"]),
        "original_option_order": True,
        "cot_content_used": False,
        "gold_used_for_training": False,
        "tokenization_equal_1_7b_4b": True,
        "prefix_suffix_composition_equal": True,
    }


def read_rows(path: str) -> list[dict[str, Any]]:
    columns = ["question_id", "question", "options", "answer", "answer_index", "category", "src"]
    return pq.read_table(path, columns=columns).to_pylist()


def balanced_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    for category, values in groups.items():
        values.sort(key=lambda row: stable_rank(f"{category}:{row['question_id']}", seed))
    ordered, categories = [], sorted(groups)
    while any(groups.values()):
        for category in categories:
            if groups[category]:
                ordered.append(groups[category].pop())
    return ordered


def filter_and_encode(rows, tokenizer17, tokenizer4, cfg, seed, label_ids):
    accepted, rejected = [], {"prefix_too_long": 0, "suffix_too_long": 0}
    for row in balanced_order(rows, seed):
        sample = encode_sample(tokenizer17, tokenizer4, row, cfg, label_ids)
        if sample["context_length"] > cfg["max_prefix_tokens"]:
            rejected["prefix_too_long"] += 1
            continue
        if sample["suffix_length"] > cfg["max_suffix_tokens"]:
            rejected["suffix_too_long"] += 1
            continue
        accepted.append(sample)
    return accepted, rejected


def split_samples(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data_cfg, counts = cfg["data"], cfg["data"]["splits"]
    strategy = data_cfg["split_strategy"]
    if strategy == "stratified_random":
        required = counts["train"] + counts["validation"] + counts["test"]
        if len(samples) < required:
            raise RuntimeError(f"only {len(samples)} eligible samples for {required} requested")
        a, b = counts["train"], counts["train"] + counts["validation"]
        return {"train": samples[:a], "validation": samples[a:b], "test": samples[b:required]}
    if strategy == "anchored_scale_up":
        anchor_root = Path(data_cfg["anchor_manifest_dir"])
        anchored = {
            split: read_jsonl(anchor_root / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        }
        by_id = {sample["id"]: sample for sample in samples}

        def resolve(rows):
            output = []
            for row in rows:
                if row["id"] not in by_id:
                    raise RuntimeError(f"anchored sample is no longer eligible: {row['id']}")
                current = by_id[row["id"]]
                if current["prefix_token_sha256"] != row["prefix_token_sha256"]:
                    raise RuntimeError(f"anchored prefix changed: {row['id']}")
                output.append(current)
            return output

        if counts["test"] != len(anchored["test"]):
            raise RuntimeError("anchored_scale_up requires the exact previous test size")
        test = resolve(anchored["test"])
        validation = resolve(anchored["validation"])
        if len(validation) > counts["validation"]:
            raise RuntimeError("requested validation is smaller than the anchored validation")
        all_anchor_ids = {
            row["id"] for rows in anchored.values() for row in rows
        }
        validation_ids = {sample["id"] for sample in validation}
        for sample in samples:
            if len(validation) >= counts["validation"]:
                break
            if sample["id"] not in all_anchor_ids and sample["id"] not in validation_ids:
                validation.append(sample)
                validation_ids.add(sample["id"])
        train = resolve(anchored["train"])
        used_ids = validation_ids | {sample["id"] for sample in test}
        train_ids = {sample["id"] for sample in train}
        for sample in samples:
            if len(train) >= counts["train"]:
                break
            if sample["id"] not in used_ids and sample["id"] not in train_ids:
                train.append(sample)
                train_ids.add(sample["id"])
        output = {"train": train, "validation": validation, "test": test}
        if any(len(output[split]) != counts[split] for split in output):
            raise RuntimeError(f"could not satisfy anchored scale-up counts: { {key: len(value) for key, value in output.items()} }")
        return output
    if strategy != "heldout_category":
        raise ValueError(f"unknown split_strategy={strategy}")
    category_sets = {split: set(data_cfg[f"{split}_categories"]) for split in ("train", "validation", "test")}
    union = set().union(*category_sets.values())
    if sum(len(values) for values in category_sets.values()) != len(union):
        raise RuntimeError("held-out category lists overlap")
    output = {}
    for split, categories in category_sets.items():
        pool = [sample for sample in samples if sample["category"] in categories]
        if len(pool) < counts[split]:
            raise RuntimeError(f"{split} category pool has {len(pool)} samples; need {counts[split]}")
        output[split] = pool[:counts[split]]
    return output


def prepare(cfg: dict[str, Any]) -> None:
    tokenizer17 = load_tokenizer(cfg["model_1_7b"])
    tokenizer4 = load_tokenizer(cfg["model_4b"])
    if tuple(cfg["labels"]) != LABELS:
        raise RuntimeError("config labels must be exactly A-J")
    label_ids17 = label_token_ids(tokenizer17, list(cfg["labels"]))
    label_ids4 = label_token_ids(tokenizer4, list(cfg["labels"]))
    if label_ids17 != label_ids4:
        raise RuntimeError(f"1.7B/4B A-J label token IDs differ: {label_ids17} vs {label_ids4}")
    samples, rejected = filter_and_encode(
        read_rows(cfg["mmlupro_test"]), tokenizer17, tokenizer4, cfg, cfg["seed"], label_ids4
    )
    manifests = split_samples(samples, cfg)
    ids = [sample["id"] for rows in manifests.values() for sample in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("derived train/validation/test splits overlap")
    root = Path(cfg["work_dir"]) / "artifacts" / "manifests"
    for split, rows in manifests.items():
        write_jsonl(root / f"{split}.jsonl", rows)
    save_json(root / "summary.json", {
        "source_file": cfg["mmlupro_test"],
        "official_test_repartitioned": True,
        "leaderboard_comparison_valid": False,
        "split_strategy": cfg["data"]["split_strategy"],
        "anchor_manifest_dir": cfg["data"].get("anchor_manifest_dir"),
        "test_set_anchored_to_previous_experiment": cfg["data"]["split_strategy"] == "anchored_scale_up",
        "counts": {split: len(rows) for split, rows in manifests.items()},
        "categories": {split: sorted({row["category"] for row in rows}) for split, rows in manifests.items()},
        "rejected": rejected,
        "original_option_order": True,
        "option_permutation_used": False,
        "cot_content_used": False,
        "gold_fields_used_only_for_evaluation": True,
        "tokenizers_equal": True,
        "prefix_suffix_composition_equal": True,
    })
    progress("prepared MMLU-Pro Options-first manifests")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare(load_json(args.config))


if __name__ == "__main__":
    main()
