import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2a_common import (
    NativeKVExternalReader,
    extract_answer,
    full_text_prefixed_prompt,
    memory_to,
    mismatched_memory,
    normalize_answer,
    parse_dtype,
    resolve_device,
    student_prefixed_prompt,
    summarize_diagnostics,
    write_jsonl,
    zero_memory,
)


class LazyPairCache:
    def __init__(self, index_path, capacity=2):
        with open(index_path, encoding="utf-8") as handle:
            self.index = json.load(handle)
        if self.index.get("format_version") != 3:
            raise ValueError("P2-G1 evaluation requires format_version=3 pair cache")
        self.root = Path(index_path).parent
        self.entries = self.index["pair_files"]
        self.capacity = capacity
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index in self.loaded:
            self.loaded.move_to_end(index)
            return self.loaded[index]
        payload = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
        pair = {example["variant"]: example for example in payload["examples"]}
        self.loaded[index] = pair
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
        return pair


def compatible_negative(cache, index, candidate):
    if index == candidate:
        return False
    left = cache.entries[index]
    right = cache.entries[candidate]
    return {
        left["base_answer"], left["counterfactual_answer"]
    }.isdisjoint({right["base_answer"], right["counterfactual_answer"]})


def fixed_negative(cache, index):
    for offset in range(1, len(cache)):
        candidate = (index + offset) % len(cache)
        if compatible_negative(cache, index, candidate):
            return candidate
    raise RuntimeError(f"No compatible evaluation negative for pair {index}")


@torch.inference_mode()
def generate(receiver, tokenizer, adapter, prompt, memory, max_new_tokens, device, enable_reader):
    current = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    past = None
    token_ids = []
    diagnostics = {}
    eos_ids = tokenizer.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    eos_reached = False
    for _ in range(max_new_tokens):
        if enable_reader:
            with adapter.inject(receiver, memory, diagnostics):
                output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        else:
            output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        token = int(output.logits[:, -1, :].argmax(dim=-1).item())
        token_ids.append(token)
        past = output.past_key_values
        if token in eos_ids:
            eos_reached = True
            break
        current = torch.tensor([[token]], dtype=torch.long, device=device)
    return {
        "token_ids": token_ids,
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "eos_reached": eos_reached,
        "diagnostics": summarize_diagnostics(diagnostics),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_summary(records):
    output = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        output.append(
            {
                "condition": condition,
                "metric_role": selected[0]["metric_role"],
                "n": len(selected),
                "target_em": float(np.mean([row["target_em"] for row in selected])),
                "insufficient_rate": float(np.mean([row["prediction"] == "INSUFFICIENT" for row in selected])),
                "answer_found_rate": float(np.mean([row["answer_found"] for row in selected])),
                "eos_rate": float(np.mean([row["eos_reached"] for row in selected])),
                "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in selected])),
            }
        )
    return output


def paired_consistency(records, left_condition, right_condition):
    left = {row["pair_id"]: row for row in records if row["condition"] == left_condition}
    right = {row["pair_id"]: row for row in records if row["condition"] == right_condition}
    common = left.keys() & right.keys()
    return float(np.mean([left[pair_id]["target_em"] * right[pair_id]["target_em"] for pair_id in common]))


def prediction_switch_rate(records, left_condition, right_condition):
    left = {row["pair_id"]: row for row in records if row["condition"] == left_condition}
    right = {row["pair_id"]: row for row in records if row["condition"] == right_condition}
    common = left.keys() & right.keys()
    return float(np.mean([left[pair_id]["prediction"] != right[pair_id]["prediction"] for pair_id in common]))


def main():
    parser = argparse.ArgumentParser(description="P2-G1 Qwen3-4B Native Query Reader free-running evaluation")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-4B")
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    if train_args.get("config_name") != "query_only" or int(train_args.get("query_rank", -1)) != 32:
        raise ValueError("P2-G1 evaluation requires the rank-32 Query-only Reader checkpoint")
    cache = LazyPairCache(args.test_index)
    if cache.index.get("model") != args.model:
        raise ValueError(f"Cache model {cache.index.get('model')!r} does not match receiver {args.model!r}")
    pair_count = min(len(cache), args.max_pairs) if args.max_pairs > 0 else len(cache)
    allowed_answers = sorted(
        {
            answer
            for entry in cache.entries
            for answer in (entry["base_answer"], entry["counterfactual_answer"])
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    adapter = NativeKVExternalReader(
        receiver,
        max_gate=float(train_args["max_gate"]),
        gate_init=float(train_args["gate_init"]),
        query_rank=int(train_args["query_rank"]),
        output_rank=int(train_args["output_rank"]),
    ).to(device).eval()
    adapter.load_state_dict(checkpoint["adapter"])

    records = []
    layer_rows = []
    for pair_index in tqdm(range(pair_count), desc=f"p2a2_eval_{train_args['config_name']}"):
        pair = cache.load(pair_index)
        negative_pair = cache.load(fixed_negative(cache, pair_index))
        base = pair["base"]
        counterfactual = pair["counterfactual"]
        other = negative_pair["base"]
        base_memory = memory_to(base["memory"], device, dtype)
        cf_memory = memory_to(counterfactual["memory"], device, dtype)
        other_memory = memory_to(other["memory"], device, dtype)
        conditions = [
            (
                "full_text_prefilled_final_base",
                full_text_prefixed_prompt(tokenizer, base),
                None,
                base["answer"],
                False,
                "positive_accuracy",
            ),
            (
                "full_text_prefilled_final_counterfactual",
                full_text_prefixed_prompt(tokenizer, counterfactual),
                None,
                counterfactual["answer"],
                False,
                "positive_accuracy",
            ),
            (
                "correct_kv",
                student_prefixed_prompt(tokenizer, base),
                base_memory,
                base["answer"],
                True,
                "positive_accuracy",
            ),
            (
                "counterfactual_kv",
                student_prefixed_prompt(tokenizer, base),
                cf_memory,
                counterfactual["answer"],
                True,
                "positive_accuracy",
            ),
            (
                "shuffled_kv",
                student_prefixed_prompt(tokenizer, base),
                other_memory,
                base["answer"],
                True,
                "original_answer_leakage",
            ),
            (
                "mismatched_kv",
                student_prefixed_prompt(tokenizer, base),
                mismatched_memory(base_memory, other_memory),
                base["answer"],
                True,
                "original_answer_leakage",
            ),
            (
                "zero_kv",
                student_prefixed_prompt(tokenizer, base),
                zero_memory(base_memory),
                base["answer"],
                True,
                "original_answer_leakage",
            ),
            (
                "reader_off",
                student_prefixed_prompt(tokenizer, base),
                base_memory,
                base["answer"],
                False,
                "original_answer_leakage",
            ),
        ]

        for condition, prompt, memory, target, enabled, metric_role in conditions:
            result = generate(
                receiver, tokenizer, adapter, prompt, memory, args.max_new_tokens, device, enabled
            )
            prediction, extraction_method = extract_answer(result["text"], allowed_answers)
            target_em = float(normalize_answer(prediction) == normalize_answer(target))
            records.append(
                {
                    "pair_id": base["pair_id"],
                    "config_name": train_args["config_name"],
                    "condition": condition,
                    "metric_role": metric_role,
                    "target": target,
                    "base_answer": base["answer"],
                    "counterfactual_answer": counterfactual["answer"],
                    "negative_pair_id": other["pair_id"],
                    "prediction": prediction,
                    "generated_text": result["text"],
                    "generated_token_ids": result["token_ids"],
                    "generated_tokens": len(result["token_ids"]),
                    "eos_reached": result["eos_reached"],
                    "answer_found": bool(prediction),
                    "extraction_method": extraction_method,
                    "target_em": target_em,
                }
            )
            for layer in result["diagnostics"]:
                layer_rows.append(
                    {
                        "pair_id": base["pair_id"],
                        "config_name": train_args["config_name"],
                        "condition": condition,
                        **layer,
                    }
                )

    conditions = condition_summary(records)
    summary = {
        "status": "complete",
        "config_name": train_args["config_name"],
        "query_rank": int(train_args["query_rank"]),
        "output_rank": int(train_args["output_rank"]),
        "args": vars(args),
        "conditions": conditions,
        "kv_paired_consistency": paired_consistency(records, "correct_kv", "counterfactual_kv"),
        "kv_prediction_switch_rate": prediction_switch_rate(records, "correct_kv", "counterfactual_kv"),
        "full_text_prefilled_final_paired_consistency": paired_consistency(
            records,
            "full_text_prefilled_final_base",
            "full_text_prefilled_final_counterfactual",
        ),
        "gates": adapter.gates().detach().float().cpu().tolist(),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_jsonl(output / "per_layer_reader_diagnostics.jsonl", layer_rows)
    write_csv(output / "condition_summary.csv", conditions)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
