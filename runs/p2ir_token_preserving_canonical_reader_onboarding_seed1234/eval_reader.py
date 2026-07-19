import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from p2ir_common import (
    PairCache, canonical_to, extract_answer, fixed_negative, generate, load_receiver,
    normalize_answer, parse_dtype, resolve_device, write_json, write_jsonl,
)
from p2ir_reader import TokenCanonicalReader, full_attention_layers


def zero_memory(memory):
    return {**memory, "keys": torch.zeros_like(memory["keys"]), "values": torch.zeros_like(memory["values"]), "answer_token_mask": torch.zeros_like(memory["answer_token_mask"])}


def permute_memory(memory, order):
    return {name: value.index_select(0, order) if value.ndim >= 1 and value.shape[0] == len(order) else value for name, value in memory.items()}


def resize_rows(value, target):
    if value.shape[0] == target:
        return value
    index = torch.linspace(0, value.shape[0] - 1, target, device=value.device).round().long()
    return value.index_select(0, index)


def mismatched_memory(current, other):
    target = current["keys"].shape[0]
    return {
        "keys": current["keys"], "values": resize_rows(other["values"], target),
        "mask": current["mask"], "answer_token_mask": resize_rows(other["answer_token_mask"].float()[:, None], target)[:, 0].bool(),
    }


def summarize(records):
    conditions = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        conditions.append({
            "condition": condition, "n": len(selected),
            "target_em": float(np.mean([row["target_em"] for row in selected])),
            "source_memory_accuracy": float(np.mean([row["source_memory_correct"] for row in selected])),
            "eos_rate": float(np.mean([row["eos_reached"] for row in selected])),
            "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in selected])),
        })
    correct = defaultdict(dict)
    for row in records:
        if row["condition"] == "correct":
            correct[row["pair_id"]][row["variant"]] = row
    complete = [pair for pair in correct.values() if {"base", "counterfactual"}.issubset(pair)]
    paired = float(np.mean([pair["base"]["target_em"] * pair["counterfactual"]["target_em"] for pair in complete]))
    switch = float(np.mean([pair["base"]["prediction"] != pair["counterfactual"]["prediction"] for pair in complete]))
    base_em = float(np.mean([pair["base"]["target_em"] for pair in complete]))
    cf_em = float(np.mean([pair["counterfactual"]["target_em"] for pair in complete]))
    return {"base_em": base_em, "counterfactual_em": cf_em, "paired_consistency": paired, "answer_switch_rate": switch, "conditions": conditions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-name", choices=("qwen3_4b", "qwen3_5_4b"), required=True)
    parser.add_argument("--receiver-model", required=True); parser.add_argument("--canonical-index", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=64); parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda"); parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    device = resolve_device(args.device); dtype = parse_dtype(args.dtype, device)
    cache = PairCache(args.canonical_index, capacity=3); count = min(len(cache), args.max_pairs)
    labels = sorted({answer for entry in cache.entries for answer in (entry["base_answer"], entry["counterfactual_answer"])})
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["writer_checkpoint_sha256"] != cache.index["writer_checkpoint_sha256"] or checkpoint["writer_state_sha256"] != cache.index["writer_state_sha256"]:
        raise RuntimeError("Reader checkpoint and Canonical cache use different frozen Writers")
    model, tokenizer = load_receiver(args.receiver_model, device, dtype)
    metadata = checkpoint["reader_metadata"]
    reader = TokenCanonicalReader(
        model, canonical_dim=metadata["canonical_dim"], rank=metadata["rank"], max_gate=metadata["max_gate"],
        gate_init=0.0, active_layers=metadata["active_layers"],
    ).to(device).eval()
    reader.load_state_dict(checkpoint["reader"])
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    if metadata["active_layers"] != full_attention_layers(model):
        raise RuntimeError("Reader active layers do not match this Receiver")
    records, layer_records = [], []
    generator = torch.Generator(device=device).manual_seed(args.seed + 77)
    for index in tqdm(range(count), desc=f"eval_{args.receiver_name}"):
        pair = cache.load(index); other_pair = cache.load(fixed_negative(cache, index))
        for variant in ("base", "counterfactual"):
            row = pair[variant]; opposite = pair["counterfactual" if variant == "base" else "base"]
            other = other_pair[variant]
            correct = canonical_to(row["memory"], device); swapped = canonical_to(opposite["memory"], device)
            shuffled = canonical_to(other["memory"], device)
            order = torch.randperm(correct["keys"].shape[0], generator=generator, device=device)
            conditions = [
                ("correct", correct, row["answer"], row["answer"], True),
                ("base_cf_memory_swap", swapped, row["answer"], opposite["answer"], True),
                ("cross_sample_shuffled", shuffled, row["answer"], other["answer"], True),
                ("k_current_v_other", mismatched_memory(correct, shuffled), row["answer"], other["answer"], True),
                ("zero", zero_memory(correct), row["answer"], "", True),
                ("reader_off", correct, row["answer"], row["answer"], False),
                ("token_permutation", permute_memory(correct, order), row["answer"], row["answer"], True),
            ]
            for condition, memory, target, source_answer, enabled in conditions:
                result = generate(model, tokenizer, reader, row, memory, args.max_new_tokens, device, enabled)
                prediction, method = extract_answer(result["text"], labels)
                record = {
                    "pair_id": row["pair_id"], "variant": variant, "condition": condition,
                    "target": target, "source_memory_answer": source_answer, "prediction": prediction,
                    "generated_text": result["text"], "token_ids": result["token_ids"],
                    "generated_tokens": len(result["token_ids"]), "eos_reached": result["eos_reached"],
                    "extraction_method": method,
                    "target_em": float(normalize_answer(prediction) == normalize_answer(target)),
                    "source_memory_correct": float(bool(source_answer) and normalize_answer(prediction) == normalize_answer(source_answer)),
                }
                records.append(record)
                for layer, values in result["diagnostics"].items():
                    layer_records.append({"pair_id": row["pair_id"], "variant": variant, "condition": condition, "layer": int(layer), **values})
    metrics = summarize(records)
    condition_map = {row["condition"]: row for row in metrics["conditions"]}
    threshold = 0.85 if args.receiver_name == "qwen3_4b" else 0.70
    metrics["success_threshold"] = threshold
    metrics["threshold_passed"] = bool(
        metrics["paired_consistency"] >= threshold
        and condition_map["zero"]["target_em"] <= 0.10
        and condition_map["reader_off"]["target_em"] <= 0.10
        and condition_map["base_cf_memory_swap"]["source_memory_accuracy"] > condition_map["base_cf_memory_swap"]["target_em"]
    )
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records); write_jsonl(output / "per_layer_diagnostics.jsonl", layer_records)
    write_json(output / "SUCCESS.json", {
        "status": "complete", "receiver": args.receiver_name, "pairs": count, **metrics,
        "writer_checkpoint_sha256": cache.index["writer_checkpoint_sha256"],
        "writer_state_sha256": cache.index["writer_state_sha256"], "reader_active_layers": metadata["active_layers"],
    })


if __name__ == "__main__":
    main()
