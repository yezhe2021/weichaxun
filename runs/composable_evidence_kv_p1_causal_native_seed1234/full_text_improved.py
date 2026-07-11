import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from causal_common import load_jsonl, parse_dtype, resolve_device, write_jsonl


def render_prompt(tokenizer, row):
    system = (
        "You solve evidence-grounded two-hop lookup questions. Use only the supplied evidence. "
        "First identify the entity linked to the question entity in Evidence Block 1. Then find "
        "that exact entity in Evidence Block 2 and return the entity linked to it. Ignore unrelated "
        "facts. End with exactly one line in the form FINAL: <answer_identifier>."
    )
    user = (
        f"QUESTION\n{row['question']}\n\n"
        f"EVIDENCE BLOCK 1\n{row['evidence_a']}\n\n"
        f"EVIDENCE BLOCK 2\n{row['evidence_b']}\n\n"
        "Follow the two-hop chain and give the answer identifier."
    )
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\nFINAL:"


def normalize_identifier(text):
    return re.sub(r"^[\s`*\"']+|[\s`*\"'.,;:!?]+$", "", str(text)).upper()


def extract_prediction(text, candidates):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.IGNORECASE | re.DOTALL)
    if "</think>" in clean.lower():
        clean = re.split(r"</think>", clean, flags=re.IGNORECASE)[-1]

    candidate_map = {normalize_identifier(value): value for value in candidates}
    escaped = sorted((re.escape(key) for key in candidate_map), key=len, reverse=True)
    pattern = re.compile(r"(?<![A-Z0-9_])(" + "|".join(escaped) + r")(?![A-Z0-9_])", re.IGNORECASE)

    anchored = re.findall(r"(?:FINAL|ANSWER|答案)\s*[:：]\s*([^\n\r]+)", clean, flags=re.IGNORECASE)
    for region in reversed(anchored):
        matches = pattern.findall(region)
        if matches:
            key = normalize_identifier(matches[-1])
            return candidate_map[key], "final_anchor", len(set(map(normalize_identifier, matches))) > 1

    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if lines:
        matches = pattern.findall(lines[-1])
        if matches:
            key = normalize_identifier(matches[-1])
            return candidate_map[key], "last_line", len(set(map(normalize_identifier, matches))) > 1

    matches = pattern.findall(clean)
    if matches:
        unique = set(map(normalize_identifier, matches))
        key = normalize_identifier(matches[-1])
        return candidate_map[key], "last_valid_candidate", len(unique) > 1
    return "", "not_found", False


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
    parser = argparse.ArgumentParser(description="Improved full-text capability gate")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=256)
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
    rows = load_jsonl(args.data, args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()

    records = []
    for row in tqdm(rows, desc="full_text_improved"):
        prompt = render_prompt(tokenizer, row)
        text, token_count = generate(model, tokenizer, prompt, device, args.max_new_tokens)
        prediction, extraction_method, ambiguous = extract_prediction(text, row["candidate_answers"])
        strict = normalize_identifier(text) == normalize_identifier(row["answer"])
        correct = normalize_identifier(prediction) == normalize_identifier(row["answer"])
        records.append(
            {
                "id": row["id"],
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "schema": row["schema"],
                "target": row["answer"],
                "prediction": prediction,
                "generated_text": text,
                "generated_tokens": token_count,
                "extraction_method": extraction_method,
                "ambiguous_generation": ambiguous,
                "answer_found": bool(prediction),
                "extracted_exact_match": float(correct),
                "strict_exact_match": float(strict),
            }
        )

    n = len(records)
    summary = {
        "status": "complete",
        "args": vars(args),
        "n": n,
        "full_text_extracted_em": float(np.mean([r["extracted_exact_match"] for r in records])) if n else 0.0,
        "full_text_strict_em": float(np.mean([r["strict_exact_match"] for r in records])) if n else 0.0,
        "answer_found_rate": float(np.mean([r["answer_found"] for r in records])) if n else 0.0,
        "ambiguous_generation_rate": float(np.mean([r["ambiguous_generation"] for r in records])) if n else 0.0,
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
