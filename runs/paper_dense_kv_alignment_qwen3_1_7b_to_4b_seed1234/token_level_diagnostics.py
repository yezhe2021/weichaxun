import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from paper_dense_common import (
    answer_f1,
    assert_tokenizer_compatible,
    build_paper_example,
    final_answer_exact_match,
    load_rows,
    normalize_answer,
    run_generation,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
for path in (PROJECT_ROOT, REAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_kv_common import extract_cache  # noqa: E402
from real_kv_translator import load_real_translator  # noqa: E402


def load_model(path, dtype, device):
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, trust_remote_code=True).to(device).eval()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decode_prediction(tokenizer, logits):
    ids = logits.argmax(dim=-1)[0]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def token_texts(tokenizer, token_ids):
    return [tokenizer.decode([int(token_id)], skip_special_tokens=True) for token_id in token_ids[0].tolist()]


def critical_mask(tokenizer, answer_ids, answer_text, mode):
    tokens = token_texts(tokenizer, answer_ids)
    mask = []
    normalized_gold = set(normalize_answer(answer_text).split())
    for token in tokens:
        normalized_token = normalize_answer(token)
        is_numeric = any(ch.isdigit() for ch in token)
        is_answer_word = bool(normalized_token and normalized_token in normalized_gold)
        if mode == "numeric":
            mask.append(is_numeric)
        elif mode == "answer":
            mask.append(is_numeric or is_answer_word)
        else:
            raise ValueError(f"Unknown critical mode: {mode}")
    if not any(mask):
        mask = [True for _ in tokens]
    return torch.tensor(mask, dtype=torch.bool)


def masked_mean(values, mask):
    if mask.numel() == 0 or not bool(mask.any()):
        return float("nan")
    return values[mask].float().mean().item()


def native_upper_bound_row(tokenizer, native_logits, answer_ids, answer_text):
    prediction = decode_prediction(tokenizer, native_logits.detach().float().cpu())
    final_em, pred_final, gold_final = final_answer_exact_match(prediction, answer_text)
    token_correct = native_logits.detach().cpu().argmax(-1) == answer_ids.cpu()
    ce = F.cross_entropy(
        native_logits.detach().float().cpu().reshape(-1, native_logits.shape[-1]),
        answer_ids.cpu().reshape(-1),
    ).item()
    return {
        "native_teacher_forced_prediction": prediction,
        "native_teacher_forced_answer_f1": answer_f1(prediction, answer_text),
        "native_teacher_forced_final_answer_exact_match": final_em,
        "native_pred_final_answer": pred_final,
        "gold_final_answer": gold_final,
        "native_ce": ce,
        "native_top1_gold_acc": token_correct.float().mean().item(),
    }


def token_diagnostic_row(tokenizer, native_logits, translated_logits, answer_ids, answer_text, critical_mode):
    native_argmax = native_logits.detach().cpu().argmax(-1)
    translated_argmax = translated_logits.detach().cpu().argmax(-1)
    gold = answer_ids.cpu()
    n = min(native_argmax.shape[1], translated_argmax.shape[1], gold.shape[1])
    native_argmax = native_argmax[:, :n]
    translated_argmax = translated_argmax[:, :n]
    gold = gold[:, :n]
    native_correct = native_argmax == gold
    translated_correct = translated_argmax == gold
    top1_match = translated_argmax == native_argmax
    critical = critical_mask(tokenizer, gold, answer_text, critical_mode)
    noncritical = ~critical
    matched_mask = top1_match[0]
    matched_native_correct = native_correct[0] & matched_mask
    return {
        "all_token_top1_match": top1_match.float().mean().item(),
        "all_native_gold_acc": native_correct.float().mean().item(),
        "all_translated_gold_acc": translated_correct.float().mean().item(),
        "matched_token_native_gold_acc": masked_mean(native_correct[0], matched_mask),
        "matched_token_translated_gold_acc": masked_mean(translated_correct[0], matched_mask),
        "matched_token_count": int(matched_mask.sum().item()),
        "matched_native_correct_count": int(matched_native_correct.sum().item()),
        "critical_token_count": int(critical.sum().item()),
        "critical_top1_match": masked_mean(top1_match[0], critical),
        "critical_native_gold_acc": masked_mean(native_correct[0], critical),
        "critical_translated_gold_acc": masked_mean(translated_correct[0], critical),
        "critical_matched_native_gold_acc": masked_mean(native_correct[0], critical & matched_mask),
        "noncritical_top1_match": masked_mean(top1_match[0], noncritical),
        "noncritical_native_gold_acc": masked_mean(native_correct[0], noncritical),
        "noncritical_translated_gold_acc": masked_mean(translated_correct[0], noncritical),
    }


def summarize(rows):
    keys = [
        "native_teacher_forced_answer_f1",
        "native_teacher_forced_final_answer_exact_match",
        "native_ce",
        "native_top1_gold_acc",
        "all_token_top1_match",
        "all_native_gold_acc",
        "all_translated_gold_acc",
        "matched_token_native_gold_acc",
        "matched_token_translated_gold_acc",
        "matched_token_count",
        "matched_native_correct_count",
        "critical_token_count",
        "critical_top1_match",
        "critical_native_gold_acc",
        "critical_translated_gold_acc",
        "critical_matched_native_gold_acc",
        "noncritical_top1_match",
        "noncritical_native_gold_acc",
        "noncritical_translated_gold_acc",
    ]
    output = []
    for method in sorted({row["method"] for row in rows}):
        for mode in sorted({row["receiver_prompt_mode"] for row in rows if row["method"] == method}):
            selected = [row for row in rows if row["method"] == method and row["receiver_prompt_mode"] == mode]
            item = {"method": method, "receiver_prompt_mode": mode, "n": len(selected)}
            for key in keys:
                values = [row[key] for row in selected if key in row and np.isfinite(row[key])]
                if values:
                    item[key] = float(np.mean(values))
            output.append(item)
    return output


def eval_method(sender, receiver, tokenizer, adapter, rows, args):
    per_example = []
    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc=args.method_label)):
            example = build_paper_example(tokenizer, row, args.max_source_tokens)
            source_ids = example["source_ids"].to(args.device_obj)
            sender_pairs = extract_cache(sender(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            native_pairs = extract_cache(receiver(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            translated_pairs = adapter(sender_pairs)
            for mode in ("context_aware", "context_unaware"):
                if mode == "context_aware":
                    tail_ids = example["aware_tail_ids"].to(args.device_obj)
                    prefix_len = example["aware_prefix_len"]
                else:
                    tail_ids = example["unaware_tail_ids"].to(args.device_obj)
                    prefix_len = example["unaware_prefix_len"]
                answer_ids = example["answer_ids"].to(args.device_obj)
                answer_len = answer_ids.shape[1]
                native_logits, _, _ = run_generation(receiver, native_pairs, tail_ids, prefix_len, answer_len, capture_trace=False)
                translated_logits, _, _ = run_generation(receiver, translated_pairs, tail_ids, prefix_len, answer_len, capture_trace=False)
                native_row = native_upper_bound_row(tokenizer, native_logits, answer_ids, example["answer"])
                token_row = token_diagnostic_row(
                    tokenizer,
                    native_logits,
                    translated_logits,
                    answer_ids,
                    example["answer"],
                    args.critical_mode,
                )
                per_example.append(
                    {
                        "sample": sample,
                        "id": example["id"],
                        "task_type": example["task_type"],
                        "method": args.method_label,
                        "receiver_prompt_mode": mode,
                        **native_row,
                        **token_row,
                    }
                )
    return per_example


def main():
    parser = argparse.ArgumentParser(description="Token-level diagnostics for native upper bound and answer-critical top1 match")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--critical-mode", choices=["answer", "numeric"], default="answer")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--tokenizer-check-samples", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cpu" and args.dtype in {"float16", "bfloat16"}:
        raise ValueError(f"{args.dtype} on CPU is unsupported")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device_obj = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    rows = load_rows(args.data, args.max_samples)
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(sender_tokenizer, receiver_tokenizer, rows, args.max_source_tokens, args.tokenizer_check_samples)

    sender = load_model(args.sender_model, dtype, args.device_obj)
    receiver = load_model(args.receiver_model, dtype, args.device_obj)
    adapter, adapter_metadata = load_real_translator(args.adapter_checkpoint, map_location=args.device_obj)
    adapter = adapter.to(args.device_obj).eval()
    for module in (sender, receiver, adapter):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_example = eval_method(sender, receiver, receiver_tokenizer, adapter, rows, args)
    summary = summarize(per_example)
    write_jsonl(out / "per_example.jsonl", per_example)
    write_csv(out / "summary.csv", summary)
    payload = {
        "args": {key: value for key, value in vars(args).items() if key != "device_obj"},
        "adapter_metadata": adapter_metadata,
        "diagnostic_table": summary,
        "definitions": {
            "receiver_native_upper_bound": "receiver native teacher-forced answer prediction under each receiver prompt mode",
            "top1_match": "translated argmax equals receiver native argmax at the answer token position",
            "native_top1_gold_acc": "receiver native argmax equals gold token",
            "critical_token": "answer-mode: numeric token or token whose normalized text appears in the normalized gold answer; numeric-mode: numeric tokens only",
        },
    }
    with open(out / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", "method": args.method_label, "samples": len(rows)}, handle, indent=2)


if __name__ == "__main__":
    main()
