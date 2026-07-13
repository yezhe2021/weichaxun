import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2a_common import (
    extract_answer,
    full_text_prompt,
    load_jsonl,
    normalize_answer,
    parse_dtype,
    resolve_device,
    student_prompt,
    write_jsonl,
)


@torch.inference_mode()
def generate(model, tokenizer, prompt, device, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True), int(generated.numel())


def main():
    parser = argparse.ArgumentParser(description="Check question-only leakage and the P2-A full-text upper bound")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    grouped = defaultdict(dict)
    for row in load_jsonl(args.data):
        grouped[row["pair_id"]][row["variant"]] = row
    pairs = [pair for pair in grouped.values() if {"base", "counterfactual"}.issubset(pair)][: args.max_pairs]
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records = []
    for pair in tqdm(pairs, desc="p2a_text_gate"):
        base, counterfactual = pair["base"], pair["counterfactual"]
        allowed = list(dict.fromkeys([*base["candidate_answers"], *counterfactual["candidate_answers"]]))
        conditions = [
            ("question_only", base, student_prompt(tokenizer, base), base["answer"]),
            ("full_text_base", base, full_text_prompt(tokenizer, base), base["answer"]),
            (
                "full_text_counterfactual",
                counterfactual,
                full_text_prompt(tokenizer, counterfactual),
                counterfactual["answer"],
            ),
        ]
        for condition, row, prompt, target in conditions:
            text, tokens = generate(model, tokenizer, prompt, device, args.max_new_tokens)
            prediction, method = extract_answer(text, allowed)
            records.append(
                {
                    "pair_id": row["pair_id"],
                    "condition": condition,
                    "target": target,
                    "prediction": prediction,
                    "generated_text": text,
                    "generated_tokens": tokens,
                    "extraction_method": method,
                    "exact_match": float(normalize_answer(prediction) == normalize_answer(target)),
                }
            )

    summary = {}
    for condition in ("question_only", "full_text_base", "full_text_counterfactual"):
        selected = [row for row in records if row["condition"] == condition]
        summary[condition] = {
            "n": len(selected),
            "exact_match": float(np.mean([row["exact_match"] for row in selected])),
        }
    by_pair = defaultdict(dict)
    for row in records:
        by_pair[row["pair_id"]][row["condition"]] = row
    consistency = float(
        np.mean(
            [
                value["full_text_base"]["exact_match"] * value["full_text_counterfactual"]["exact_match"]
                for value in by_pair.values()
            ]
        )
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "args": vars(args),
                "conditions": summary,
                "full_text_counterfactual_consistency": consistency,
                "gate_passed": consistency >= 0.8 and summary["question_only"]["exact_match"] <= 0.2,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
