import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from canonical_modules import (
    CanonicalExternalReader,
    drop_half_slots,
    full_attention_layers,
    mismatched_slots,
    permute_slots,
    zero_slots,
)
from p2i_common import (
    LazyPairCache,
    canonical_to,
    extract_answer,
    full_text_prefixed_prompt,
    generate,
    load_receiver,
    normalize_answer,
    parse_dtype,
    resolve_device,
    student_prefixed_prompt,
    write_jsonl,
)


def fixed_negative(cache, index):
    current = cache.entries[index]
    current_answers = {current["base_answer"], current["counterfactual_answer"]}
    for offset in range(1, len(cache)):
        candidate = (index + offset) % len(cache)
        other = cache.entries[candidate]
        if current_answers.isdisjoint({other["base_answer"], other["counterfactual_answer"]}):
            return candidate
    raise RuntimeError(f"No compatible negative for test pair {index}")


@torch.inference_mode()
def first_step_logits(model, tokenizer, reader, row, memory, device):
    ids = tokenizer(
        student_prefixed_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    with reader.inject(model, memory):
        output = model(input_ids=ids, use_cache=False, return_dict=True)
    return output.logits[:, -1].float()


def summarize(records):
    conditions = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        conditions.append(
            {
                "condition": condition,
                "n": len(selected),
                "target_em": float(np.mean([row["target_em"] for row in selected])),
                "source_memory_answer_rate": float(
                    np.mean([row["source_memory_answer_hit"] for row in selected])
                ),
                "insufficient_rate": float(
                    np.mean([row["prediction"] == "INSUFFICIENT" for row in selected])
                ),
                "eos_rate": float(np.mean([row["eos_reached"] for row in selected])),
                "mean_tokens": float(np.mean([row["generated_tokens"] for row in selected])),
            }
        )
    by_condition = defaultdict(dict)
    for row in records:
        by_condition[row["condition"]][row["pair_id"]] = row

    def paired(left, right):
        ids = by_condition[left].keys() & by_condition[right].keys()
        return float(np.mean([
            by_condition[left][pair_id]["target_em"] * by_condition[right][pair_id]["target_em"]
            for pair_id in ids
        ]))

    def switch(left, right):
        ids = by_condition[left].keys() & by_condition[right].keys()
        return float(np.mean([
            by_condition[left][pair_id]["prediction"] != by_condition[right][pair_id]["prediction"]
            for pair_id in ids
        ]))

    return {
        "conditions": conditions,
        "canonical_paired_consistency": paired("correct_slots", "counterfactual_slots"),
        "canonical_prediction_switch": switch("correct_slots", "counterfactual_slots"),
        "full_text_paired_consistency": paired("full_text_base", "full_text_counterfactual"),
    }


def main():
    parser = argparse.ArgumentParser(description="Free-running evaluation for one P2-I Receiver")
    parser.add_argument("--receiver-name", choices=("qwen3_4b", "qwen3_5_4b"), required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--canonical-index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mismatch-index")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    cache = LazyPairCache(args.canonical_index, capacity=2)
    mismatch = LazyPairCache(args.mismatch_index, capacity=1) if args.mismatch_index else None
    if mismatch is not None and [x["pair_id"] for x in mismatch.entries] != [x["pair_id"] for x in cache.entries]:
        raise ValueError("True mismatch cache is not pair-aligned")
    count = min(len(cache), args.max_pairs) if args.max_pairs > 0 else len(cache)
    allowed = sorted({answer for entry in cache.entries for answer in (
        entry["base_answer"], entry["counterfactual_answer"]
    )})

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.receiver_name not in checkpoint.get("readers", {}):
        raise ValueError(f"Checkpoint has no Reader state for {args.receiver_name}")
    canonical_meta = cache.index
    if canonical_meta.get("writer_sha256") != checkpoint.get("writer_sha256"):
        raise ValueError("Canonical cache and checkpoint Writer hashes differ")
    model, tokenizer = load_receiver(args.receiver_model, device, dtype)
    metadata = checkpoint["reader_metadata"][args.receiver_name]
    reader = CanonicalExternalReader(
        model,
        canonical_dim=metadata["canonical_dim"],
        adapter_rank=metadata["adapter_rank"],
        max_gate=metadata["max_gate"],
        gate_init=0.0,
        active_layers=metadata["active_layers"],
    ).to(device).eval()
    reader.load_state_dict(checkpoint["readers"][args.receiver_name])
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    if metadata["active_layers"] != full_attention_layers(model):
        raise ValueError("Reader active layers no longer match receiver full-attention layers")

    records = []
    layer_records = []
    permutation_differences = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for index in tqdm(range(count), desc=f"eval_{args.receiver_name}"):
        pair = cache.load(index)
        other_pair = cache.load(fixed_negative(cache, index))
        mismatch_pair = mismatch.load(index) if mismatch is not None else None
        base = pair["base"]
        cf = pair["counterfactual"]
        other = other_pair["base"]
        base_memory = canonical_to(base["memory"], device, dtype)
        cf_memory = canonical_to(cf["memory"], device, dtype)
        other_memory = canonical_to(other["memory"], device, dtype)
        permutation = torch.randperm(base_memory["keys"].shape[0], generator=generator, device=device)
        permuted = permute_slots(base_memory, permutation)
        permutation_differences.append(float(
            (first_step_logits(model, tokenizer, reader, base, base_memory, device)
             - first_step_logits(model, tokenizer, reader, base, permuted, device)).abs().max().cpu()
        ))
        conditions = [
            ("full_text_base", full_text_prefixed_prompt(tokenizer, base), None, base["answer"], base["answer"], False),
            ("full_text_counterfactual", full_text_prefixed_prompt(tokenizer, cf), None, cf["answer"], cf["answer"], False),
            ("correct_slots", student_prefixed_prompt(tokenizer, base), base_memory, base["answer"], base["answer"], True),
            ("counterfactual_slots", student_prefixed_prompt(tokenizer, base), cf_memory, cf["answer"], cf["answer"], True),
            ("shuffled_complete_memory", student_prefixed_prompt(tokenizer, base), other_memory, base["answer"], other["answer"], True),
            ("kv_mismatched_slots", student_prefixed_prompt(tokenizer, base), mismatched_slots(base_memory, other_memory), base["answer"], other["answer"], True),
            ("zero_slots", student_prefixed_prompt(tokenizer, base), zero_slots(base_memory), base["answer"], "", True),
            ("reader_off", student_prefixed_prompt(tokenizer, base), base_memory, base["answer"], base["answer"], False),
            ("slot_permutation", student_prefixed_prompt(tokenizer, base), permuted, base["answer"], base["answer"], True),
            ("drop_half_slots", student_prefixed_prompt(tokenizer, base), drop_half_slots(base_memory), base["answer"], base["answer"], True),
        ]
        if mismatch_pair is not None:
            true_mismatch = canonical_to(mismatch_pair["base"]["memory"], device, dtype)
            conditions.append((
                "true_mismatched_a_b", student_prefixed_prompt(tokenizer, base), true_mismatch,
                base["answer"], mismatch_pair["base"]["answer"], True,
            ))

        for condition, prompt, memory, target, source_answer, enabled in conditions:
            result = generate(
                model, tokenizer, reader, prompt, memory, args.max_new_tokens, device, enabled
            )
            prediction, method = extract_answer(result["text"], allowed)
            record = {
                "pair_id": base["pair_id"],
                "receiver": args.receiver_name,
                "condition": condition,
                "target": target,
                "source_memory_answer": source_answer,
                "prediction": prediction,
                "generated_text": result["text"],
                "generated_token_ids": result["token_ids"],
                "generated_tokens": len(result["token_ids"]),
                "eos_reached": result["eos_reached"],
                "extraction_method": method,
                "target_em": float(normalize_answer(prediction) == normalize_answer(target)),
                "source_memory_answer_hit": float(
                    bool(source_answer) and normalize_answer(prediction) == normalize_answer(source_answer)
                ),
            }
            records.append(record)
            for layer in result["diagnostics"]:
                layer_records.append({"pair_id": base["pair_id"], "condition": condition, **layer})

    metrics = summarize(records)
    summary = {
        "status": "complete",
        "receiver": args.receiver_name,
        "args": vars(args),
        **metrics,
        "slot_permutation_max_logit_difference": max(permutation_differences),
        "slot_permutation_mean_logit_difference": float(np.mean(permutation_differences)),
        "writer_sha256": checkpoint["writer_sha256"],
        "reader_active_layers": metadata["active_layers"],
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_jsonl(output / "per_layer_reader_diagnostics.jsonl", layer_records)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
