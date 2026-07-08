import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_rows(path, limit):
    rows = []
    with open(path, encoding="utf-8") as handle:
        source = (json.loads(line) for line in handle if line.strip()) if str(path).endswith(".jsonl") else iter(json.load(handle))
        for idx, row in enumerate(source):
            if row.get("question") and row.get("answer") is not None:
                row = dict(row)
                row.setdefault("id", idx)
                rows.append(row)
            if 0 < limit <= len(rows):
                break
    return rows


def parse_dtype(name):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def normalize_number(text):
    return str(text).replace(",", "").strip()


def extract_strict(text):
    matches = re.findall(r"####\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", str(text))
    return normalize_number(matches[-1]) if matches else ""


def extract_flexible(text):
    text = str(text)
    strict = extract_strict(text)
    if strict:
        return strict
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", boxed[-1])
        if nums:
            return normalize_number(nums[-1])
    patterns = [
        r"(?:final answer|answer|therefore|so|thus)\s*(?:is|:)?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_number(matches[-1])
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(nums[-1]) if nums else ""


def gold_final_answer(answer):
    strict = extract_strict(answer)
    if strict:
        return strict
    return extract_flexible(answer)


def equivalent_number(a, b):
    try:
        return float(str(a).replace(",", "")) == float(str(b).replace(",", ""))
    except ValueError:
        return normalize_number(a) == normalize_number(b)


def trim_at_stop(text, stop_strings):
    end = len(text)
    for stop in stop_strings:
        if not stop:
            continue
        pos = text.find(stop)
        if pos >= 0:
            end = min(end, pos)
    return text[:end].strip()


def make_fewshot_prompt(examples):
    chunks = []
    for row in examples:
        chunks.append(f"Question: {row['question']}\nAnswer: {str(row['answer']).strip()}")
    return "\n\n".join(chunks)


def make_prompt(fewshot_prefix, question):
    current = f"Question: {question}\nAnswer:"
    if fewshot_prefix:
        return fewshot_prefix + "\n\n" + current
    return current


@torch.inference_mode()
def generate_trace(model, tokenizer, prompt, args):
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(args.device_obj)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    generated_ids = output[0, input_ids.shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return trim_at_stop(text, args.stop_strings)


def main():
    parser = argparse.ArgumentParser(description="Generate receiver-self GSM8K traces for paper dense KV alignment Phase II")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-4B")
    parser.add_argument("--data", default="/home/yezhe/数据集/gsm8k/train.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rejected-out")
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--max-kept", type=int, default=256)
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--filter-mode", choices=["strict", "flexible"], default="strict")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--stop-strings", nargs="*", default=["Question:", "</s>", "<|im_end|>"])
    args = parser.parse_args()

    if args.device == "cpu" and args.dtype in {"float16", "bfloat16"}:
        raise ValueError(f"{args.dtype} on CPU is unsupported")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device_obj = torch.device(args.device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = Path(args.rejected_out) if args.rejected_out else out_path.with_name(out_path.stem + "_rejected.jsonl")
    summary_path = out_path.with_name(out_path.stem + "_summary.json")

    rows = load_rows(args.data, args.max_candidates + args.num_fewshot if args.max_candidates > 0 else 0)
    if len(rows) <= args.num_fewshot:
        raise ValueError("Not enough rows to build few-shot prompt and candidates")
    fewshot_rows = rows[: args.num_fewshot]
    candidate_rows = rows[args.num_fewshot :]
    fewshot_prefix = make_fewshot_prompt(fewshot_rows)

    dtype = parse_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.receiver_model, dtype=dtype, trust_remote_code=True).to(args.device_obj).eval()

    kept = 0
    rejected = 0
    with open(out_path, "w", encoding="utf-8") as accepted_handle, open(rejected_path, "w", encoding="utf-8") as rejected_handle:
        for row in tqdm(candidate_rows, desc="receiver_self_traces"):
            if args.max_kept > 0 and kept >= args.max_kept:
                break
            prompt = make_prompt(fewshot_prefix, row["question"])
            trace = generate_trace(model, tokenizer, prompt, args)
            strict_pred = extract_strict(trace)
            flexible_pred = extract_flexible(trace)
            gold = gold_final_answer(row["answer"])
            if args.filter_mode == "strict":
                accepted = bool(strict_pred) and equivalent_number(strict_pred, gold)
                pred = strict_pred
            else:
                accepted = bool(flexible_pred) and equivalent_number(flexible_pred, gold)
                pred = flexible_pred

            payload = {
                **row,
                "source_text": f"Question: {row['question']}\n",
                "generation_prompt": "Answer:",
                "continuation_prompt": "Answer:",
                "receiver_trace": trace,
                "receiver_trace_pred_final": pred,
                "receiver_trace_strict_final": strict_pred,
                "receiver_trace_flexible_final": flexible_pred,
                "gold_final_answer": gold,
                "self_trace_filter_mode": args.filter_mode,
                "self_trace_correct": accepted,
            }
            if accepted:
                accepted_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                kept += 1
            else:
                rejected_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                rejected += 1

    summary = {
        "receiver_model": args.receiver_model,
        "data": args.data,
        "out": str(out_path),
        "rejected_out": str(rejected_path),
        "max_candidates": args.max_candidates,
        "max_kept": args.max_kept,
        "num_fewshot": args.num_fewshot,
        "filter_mode": args.filter_mode,
        "kept": kept,
        "rejected": rejected,
        "accept_rate_over_processed": kept / max(1, kept + rejected),
        "prompt_format": "GSM8K benchmark style: few-shot Question/Answer examples, then Question: X\\nAnswer:",
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
