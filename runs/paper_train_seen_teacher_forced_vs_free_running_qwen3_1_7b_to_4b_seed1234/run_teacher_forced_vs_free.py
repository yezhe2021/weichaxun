import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_ROOT = PROJECT_ROOT / "runs" / "paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234"
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
for path in (PROJECT_ROOT, BASE_ROOT, REAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_dense_common import build_paper_example, load_rows  # noqa: E402
from real_kv_common import extract_cache, make_cache  # noqa: E402
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


def gold_final_answer(row):
    if row.get("gold_final_answer") is not None:
        return normalize_number(row["gold_final_answer"])
    strict = extract_strict(row.get("answer", ""))
    if strict:
        return strict
    return extract_flexible(row.get("answer", ""))


def equivalent_number(a, b):
    try:
        return float(str(a).replace(",", "")) == float(str(b).replace(",", ""))
    except ValueError:
        return normalize_number(a) == normalize_number(b)


def answer_f1(pred, gold):
    pred_tokens = re.sub(r"[^a-z0-9.\- ]", " ", str(pred).lower()).split()
    gold_tokens = re.sub(r"[^a-z0-9.\- ]", " ", str(gold).lower()).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum(min(pred_tokens.count(token), gold_tokens.count(token)) for token in set(pred_tokens) & set(gold_tokens))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def has_stop_string(text, stop_strings):
    return any(stop and stop in text for stop in stop_strings)


@torch.no_grad()
def translated_pairs_for_row(sender, receiver, adapter, example, device):
    source_ids = example["source_ids"].to(device)
    sender_pairs = extract_cache(sender(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
    receiver_pairs = extract_cache(receiver(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
    return adapter(sender_pairs), receiver_pairs


def teacher_forced_eval(receiver, tokenizer, translated_pairs, example, row, target_label, device):
    answer_ids = example["answer_ids"].to(device)
    tail_ids = example["unaware_tail_ids"].to(device)
    prefix_len = example["unaware_prefix_len"]
    cache = make_cache(translated_pairs, receiver.config)
    out = receiver(input_ids=tail_ids, past_key_values=cache, use_cache=True)
    start = prefix_len - 1
    logits = out.logits[:, start : start + answer_ids.shape[1]]
    n = min(logits.shape[1], answer_ids.shape[1])
    ce = F.cross_entropy(logits[:, :n].float().reshape(-1, logits.shape[-1]), answer_ids[:, :n].reshape(-1)).item()
    argmax_ids = logits[:, :n].argmax(dim=-1)[0].detach().cpu()
    argmax_text = tokenizer.decode(argmax_ids, skip_special_tokens=True)
    pred = extract_flexible(argmax_text)
    gold = gold_final_answer(row)
    return {
        "eval_type": "teacher_forced_argmax",
        "target_label": target_label,
        "ce": ce,
        "pred_final": pred,
        "gold_final": gold,
        "em_flexible": float(bool(pred) and equivalent_number(pred, gold)),
        "answer_f1": answer_f1(pred, gold),
        "generated_tokens": int(n),
        "generated_text": argmax_text,
    }


@torch.no_grad()
def free_running_eval(receiver, tokenizer, translated_pairs, row, args):
    input_ids = tokenizer("Answer:", return_tensors="pt", add_special_tokens=False).input_ids.to(args.device_obj)
    past = make_cache(translated_pairs, receiver.config)
    current = input_ids
    generated = []
    stopped_by_eos = False
    for _ in range(args.max_new_tokens):
        out = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        next_id = int(out.logits[:, -1, :].argmax(dim=-1)[0].detach().cpu())
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            stopped_by_eos = True
            break
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if has_stop_string(text, args.stop_strings):
            break
        past = out.past_key_values
        current = torch.tensor([[next_id]], device=args.device_obj, dtype=torch.long)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    pred = extract_flexible(text)
    gold = gold_final_answer(row)
    return {
        "eval_type": "free_running",
        "target_label": "none",
        "ce": float("nan"),
        "pred_final": pred,
        "gold_final": gold,
        "em_flexible": float(bool(pred) and equivalent_number(pred, gold)),
        "answer_f1": answer_f1(pred, gold),
        "generated_tokens": len(generated),
        "stopped_by_eos": float(stopped_by_eos),
        "generated_text": text,
    }


def summarize(rows):
    output = []
    for key in sorted({(row["eval_type"], row["target_label"]) for row in rows}):
        eval_type, target_label = key
        selected = [row for row in rows if row["eval_type"] == eval_type and row["target_label"] == target_label]
        item = {"eval_type": eval_type, "target_label": target_label, "n": len(selected)}
        for metric in ["ce", "em_flexible", "answer_f1", "generated_tokens", "stopped_by_eos"]:
            values = [row[metric] for row in selected if metric in row and np.isfinite(row[metric])]
            if values:
                item[metric] = float(np.mean(values))
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description="Compare paper checkpoint free-running vs teacher-forced argmax on seen self-trace samples")
    parser.add_argument("--sender-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-4B")
    parser.add_argument("--data", default="/home/yezhe/伪查询/runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234/self_traces/receiver_self_traces_256.jsonl")
    parser.add_argument("--adapter-checkpoint", default="/home/yezhe/伪查询/runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234/train_paper_sweep/256_e1e5/checkpoint_final.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-source-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--stop-strings", nargs="*", default=["Question:", "</s>", "<|im_end|>"])
    args = parser.parse_args()

    if args.device == "cpu" and args.dtype in {"float16", "bfloat16"}:
        raise ValueError(f"{args.dtype} on CPU is unsupported")
    torch.manual_seed(args.seed)
    args.device_obj = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    rows = load_rows(args.data, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    sender = load_model(args.sender_model, dtype, args.device_obj)
    receiver = load_model(args.receiver_model, dtype, args.device_obj)
    adapter, metadata = load_real_translator(args.adapter_checkpoint, map_location=args.device_obj)
    adapter = adapter.to(args.device_obj).eval()
    for module in (sender, receiver, adapter):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc="tf_vs_free")):
            trace_example = build_paper_example(tokenizer, row, args.max_source_tokens, answer_mode="full", target_field="receiver_trace")
            gold_example = build_paper_example(tokenizer, row, args.max_source_tokens, answer_mode="full", target_field="answer")
            translated_pairs, _ = translated_pairs_for_row(sender, receiver, adapter, trace_example, args.device_obj)
            for payload in (
                free_running_eval(receiver, tokenizer, translated_pairs, row, args),
                teacher_forced_eval(receiver, tokenizer, translated_pairs, trace_example, row, "receiver_self_trace", args.device_obj),
                teacher_forced_eval(receiver, tokenizer, translated_pairs, gold_example, row, "gold_answer", args.device_obj),
            ):
                payload.update(
                    {
                        "sample": sample,
                        "question": row.get("question"),
                        "gold_answer": row.get("answer"),
                        "receiver_trace": row.get("receiver_trace"),
                    }
                )
                all_rows.append(payload)

    summary = summarize(all_rows)
    write_jsonl(out_dir / "per_sample_tf_vs_free.jsonl", all_rows)
    write_csv(out_dir / "tf_vs_free_summary.csv", summary)
    with open(out_dir / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "args": {k: str(v) if k == "device_obj" else v for k, v in vars(args).items()},
                "adapter_metadata": metadata,
                "outputs": {
                    "summary": "tf_vs_free_summary.csv",
                    "per_sample": "per_sample_tf_vs_free.jsonl",
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
