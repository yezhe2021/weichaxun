import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer

from cache_qwen35_native_kv import qwen35_sender_prompt
from p2a_common import extract_answer, load_jsonl, normalize_answer, parse_dtype, resolve_device, write_jsonl


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens, device):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    output = model.generate(
        input_ids=encoded.input_ids.to(device),
        attention_mask=encoded.attention_mask.to(device),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output[0, encoded.input_ids.shape[1] :].tolist()
    return generated, tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B direct answerability gate")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3___5-4B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    rows = load_jsonl(args.data)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["variant"]] = row
    pairs = [
        variants
        for variants in grouped.values()
        if {"base", "counterfactual"}.issubset(variants)
    ][: args.max_pairs]
    answers = sorted(
        {row["answer"] for variants in grouped.values() for row in variants.values()}
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records = []
    for variants in tqdm(pairs, desc="qwen35_sender_gate"):
        for variant in ("base", "counterfactual"):
            row = variants[variant]
            prompt, _ = qwen35_sender_prompt(tokenizer, row)
            token_ids, text = generate(
                model, tokenizer, prompt, args.max_new_tokens, device
            )
            prediction, method = extract_answer(text, answers)
            records.append(
                {
                    "pair_id": row["pair_id"],
                    "variant": variant,
                    "target": row["answer"],
                    "prediction": prediction,
                    "generated_text": text,
                    "generated_token_ids": token_ids,
                    "extraction_method": method,
                    "correct": float(
                        normalize_answer(prediction) == normalize_answer(row["answer"])
                    ),
                }
            )

    by_pair = defaultdict(dict)
    for row in records:
        by_pair[row["pair_id"]][row["variant"]] = row
    complete = [pair for pair in by_pair.values() if len(pair) == 2]
    summary = {
        "status": "complete",
        "model": args.model,
        "pairs": len(complete),
        "base_em": float(np.mean([pair["base"]["correct"] for pair in complete])),
        "counterfactual_em": float(
            np.mean([pair["counterfactual"]["correct"] for pair in complete])
        ),
        "paired_consistency": float(
            np.mean(
                [pair["base"]["correct"] * pair["counterfactual"]["correct"] for pair in complete]
            )
        ),
        "prediction_switch_rate": float(
            np.mean(
                [pair["base"]["prediction"] != pair["counterfactual"]["prediction"] for pair in complete]
            )
        ),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample.jsonl", records)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
