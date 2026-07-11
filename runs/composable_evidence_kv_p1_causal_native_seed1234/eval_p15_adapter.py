import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from causal_common import inject_state, iter_cache, parse_dtype, resolve_device, write_csv, write_jsonl
from p15_common import GeneralEvidenceAdapter, extract_answer, normalize_answer, render_student_prompt


def condition_memories(example, condition, device, other=None):
    own_a = example["memory_a"].to(device).unsqueeze(0)
    own_b = example["memory_b"].to(device).unsqueeze(0)
    if condition == "correct":
        return own_a, own_b
    if condition == "zero":
        return None, None
    if condition == "shuffled":
        return own_a, other["memory_b"].to(device).unsqueeze(0)
    if condition == "mismatched":
        return other["memory_a"].to(device).unsqueeze(0), other["memory_b"].to(device).unsqueeze(0)
    if condition == "corrupted":
        generator = torch.Generator(device=device).manual_seed(sum(map(ord, example["id"])))
        noise_a = torch.randn(own_a.shape, generator=generator, device=device, dtype=own_a.dtype)
        noise_b = torch.randn(own_b.shape, generator=generator, device=device, dtype=own_b.dtype)
        return noise_a * own_a.float().std().clamp_min(1e-3), noise_b * own_b.float().std().clamp_min(1e-3)
    raise ValueError(condition)


def compute_state(adapter, example, condition, device, other=None):
    question = example["question_state"].to(device=device, dtype=torch.float32).unsqueeze(0)
    memory_a, memory_b = condition_memories(example, condition, device, other)
    return adapter.reader(question, memory_a, memory_b)


@torch.inference_mode()
def generate(receiver, tokenizer, adapter, row, state, device, max_new_tokens, enable_adapter):
    prompt = render_student_prompt(tokenizer, row)
    current = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    past = None
    generated = []
    diagnostics = {}
    eos_ids = tokenizer.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    ended = False
    for _ in range(max_new_tokens):
        if enable_adapter:
            def selector(hidden):
                return hidden.shape[1] - 1, hidden.shape[1]

            with inject_state(receiver, adapter, state, selector, diagnostics):
                output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        else:
            output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        token = int(output.logits[:, -1, :].argmax(dim=-1).item())
        generated.append(token)
        past = output.past_key_values
        if token in eos_ids:
            ended = True
            break
        current = torch.tensor([[token]], dtype=torch.long, device=device)
    return generated, tokenizer.decode(generated, skip_special_tokens=True), ended, diagnostics


def token_f1(prediction, target):
    pred = re.findall(r"[A-Z0-9_]+", normalize_answer(prediction))
    gold = re.findall(r"[A-Z0-9_]+", normalize_answer(target))
    if not pred or not gold:
        return float(pred == gold)
    common = sum(min(pred.count(token), gold.count(token)) for token in set(pred))
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(gold)
    return 2 * precision * recall / (precision + recall)


def summarize(records):
    rows = []
    for condition in sorted({record["condition"] for record in records}):
        selected = [record for record in records if record["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "n": len(selected),
                "exact_match": float(np.mean([row["exact_match"] for row in selected])),
                "token_f1": float(np.mean([row["token_f1"] for row in selected])),
                "answer_found_rate": float(np.mean([row["answer_found"] for row in selected])),
                "eos_rate": float(np.mean([row["eos_reached"] for row in selected])),
                "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in selected])),
                "own_answer_rate": float(np.mean([row["prediction"] == row["own_answer"] for row in selected])),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Free-running P1.5 dependence and intervention evaluation")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    adapter = GeneralEvidenceAdapter(
        memory_dim=int(checkpoint["memory_dim"]),
        receiver_dim=int(checkpoint["receiver_hidden_size"]),
        state_dim=int(train_args["state_dim"]),
        reader_heads=int(train_args["reader_heads"]),
        reader_rounds=int(train_args["reader_rounds"]),
        writer_layers=train_args["writer_layers"],
        writer_bottleneck=int(train_args["writer_bottleneck"]),
        max_gate=float(train_args["max_gate"]),
    ).to(device).eval()
    adapter.load_state_dict(checkpoint["adapter"])

    all_examples = list(iter_cache(args.test_index))
    grouped = defaultdict(dict)
    for example in all_examples:
        grouped[example["pair_id"]][example["variant"]] = example
    pairs = [pair for pair in grouped.values() if {"base", "counterfactual"}.issubset(pair)]
    pairs = pairs[: args.max_pairs if args.max_pairs > 0 else None]
    examples = [item for pair in pairs for item in (pair["base"], pair["counterfactual"])]

    correct_states = {}
    with torch.inference_mode():
        for example in examples:
            correct_states[example["id"]] = compute_state(adapter, example, "correct", device)

    conditions = (
        "question_only", "correct", "zero", "shuffled", "mismatched", "corrupted", "state_ablation", "state_swap"
    )
    records = []
    for index, example in enumerate(tqdm(examples, desc="p15_free_running")):
        other = examples[(index + 2) % len(examples)]
        counterpart = grouped[example["pair_id"]]["counterfactual" if example["variant"] == "base" else "base"]
        for condition in conditions:
            enable_adapter = condition != "question_only"
            if condition == "question_only":
                state = torch.zeros(1, int(train_args["state_dim"]), device=device)
                target = example["answer"]
            elif condition == "state_ablation":
                state = torch.zeros_like(correct_states[example["id"]])
                target = example["answer"]
            elif condition == "state_swap":
                state = correct_states[counterpart["id"]]
                target = counterpart["answer"]
            else:
                state = compute_state(adapter, example, condition, device, other)
                target = example["answer"] if condition == "correct" else "INSUFFICIENT"

            token_ids, text, eos_reached, diagnostics = generate(
                receiver, tokenizer, adapter, example, state, device, args.max_new_tokens, enable_adapter
            )
            allowed = list(dict.fromkeys([*example["candidate_answers"], *counterpart["candidate_answers"]]))
            prediction, extraction_method = extract_answer(text, allowed)
            records.append(
                {
                    "id": example["id"],
                    "pair_id": example["pair_id"],
                    "variant": example["variant"],
                    "schema": example["schema"],
                    "condition": condition,
                    "target": target,
                    "own_answer": example["answer"],
                    "counterfactual_answer": counterpart["answer"],
                    "prediction": prediction,
                    "generated_text": text,
                    "generated_token_ids": token_ids,
                    "generated_tokens": len(token_ids),
                    "eos_reached": eos_reached,
                    "answer_found": bool(prediction),
                    "extraction_method": extraction_method,
                    "exact_match": float(normalize_answer(prediction) == normalize_answer(target)),
                    "token_f1": token_f1(prediction, target),
                    "gate_mean": float(np.mean(list(diagnostics.values()))) if diagnostics else 0.0,
                }
            )

    condition_summary = summarize(records)
    summary_by_name = {row["condition"]: row for row in condition_summary}
    correct_by_pair = defaultdict(dict)
    for row in records:
        if row["condition"] == "correct":
            correct_by_pair[row["pair_id"]][row["variant"]] = row
    complete = [pair for pair in correct_by_pair.values() if {"base", "counterfactual"}.issubset(pair)]
    summary = {
        "status": "complete",
        "args": vars(args),
        "conditions": condition_summary,
        "counterfactual_consistency": float(
            np.mean([pair["base"]["exact_match"] * pair["counterfactual"]["exact_match"] for pair in complete])
        ) if complete else 0.0,
        "state_ablation_em_drop": summary_by_name["correct"]["exact_match"] - summary_by_name["state_ablation"]["exact_match"],
        "state_swap_answer_follow_rate": summary_by_name["state_swap"]["exact_match"],
    }

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_csv(output / "condition_summary.csv", condition_summary)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
