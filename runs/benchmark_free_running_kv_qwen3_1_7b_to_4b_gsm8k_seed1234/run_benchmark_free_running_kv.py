import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


def add_import_roots(base_experiment_root):
    for root in (
        PROJECT_ROOT,
        Path(base_experiment_root).resolve(),
        PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234",
    ):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def load_rows(path, limit):
    rows = []
    with open(path, encoding="utf-8") as handle:
        source = (json.loads(line) for line in handle if line.strip()) if str(path).endswith(".jsonl") else iter(json.load(handle))
        for row in source:
            if row.get("question") and row.get("answer") is not None:
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


def load_model(path, dtype, device):
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, trust_remote_code=True).to(device).eval()


def checkpoint_for(base_root, method):
    checkpoint = Path(base_root) / "train" / method / "checkpoint_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {method}: {checkpoint}")
    return checkpoint


def split_benchmark_prompt(question):
    source_text = f"Question: {question}\n"
    continuation_prompt = "Answer:"
    native_prompt = source_text + continuation_prompt
    return source_text, continuation_prompt, native_prompt


def normalize_number(text):
    return str(text).replace(",", "").strip()


def gold_final_answer(answer):
    text = str(answer)
    hash_matches = re.findall(r"####\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if hash_matches:
        return normalize_number(hash_matches[-1])
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else text.strip()


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


def normalize_answer_text(text):
    text = re.sub(r"[^a-z0-9.\- ]", " ", str(text).lower())
    return " ".join(text.split())


def answer_f1(pred, gold):
    pred_tokens = normalize_answer_text(pred).split()
    gold_tokens = normalize_answer_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum(min(pred_tokens.count(token), gold_tokens.count(token)) for token in set(pred_tokens) & set(gold_tokens))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def comma_variants(number_text):
    raw = normalize_number(number_text)
    variants = {raw}
    if re.fullmatch(r"[-+]?\d+", raw):
        sign = ""
        digits = raw
        if digits.startswith(("-", "+")):
            sign, digits = digits[0], digits[1:]
        groups = []
        while len(digits) > 3:
            groups.append(digits[-3:])
            digits = digits[:-3]
        groups.append(digits)
        variants.add(sign + ",".join(reversed(groups)))
    return variants


def contains_gold(text, gold):
    for variant in comma_variants(gold):
        if re.search(r"(?<![\d,])" + re.escape(variant) + r"(?![\d,])", str(text)):
            return True
    return False


def has_stop_string(text, stop_strings):
    return any(stop and stop in text for stop in stop_strings)


def make_zero_pairs(pairs):
    return [(torch.zeros_like(k), torch.zeros_like(v)) for k, v in pairs]


def make_shuffled_pairs(pairs, generator):
    output = []
    for k, v in pairs:
        seq_len = k.shape[-2]
        perm = torch.randperm(seq_len, generator=generator, device=k.device)
        output.append((k.index_select(-2, perm), v.index_select(-2, perm)))
    return output


def clone_pairs(pairs):
    return [(k.detach().clone(), v.detach().clone()) for k, v in pairs]


def build_mismatched_indices(rows):
    n = len(rows)
    if n <= 1:
        return [0 for _ in rows]
    return [(idx + 1) % n for idx in range(n)]


def greedy_generate_with_cache(receiver, tokenizer, input_ids, cache, args):
    from real_kv_common import make_cache

    past = make_cache(cache, receiver.config) if cache is not None else None
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
    return generated, tokenizer.decode(generated, skip_special_tokens=True), stopped_by_eos


def summarize(rows):
    keys = [
        "em_strict",
        "em_flexible",
        "answer_f1",
        "oracle_contains_gold",
        "generated_tokens",
        "stopped_by_eos",
    ]
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["cache_variant"], row["input_regime"]), []).append(row)
    output = []
    for (method, cache_variant, input_regime), selected in sorted(groups.items()):
        item = {
            "method": method,
            "cache_variant": cache_variant,
            "input_regime": input_regime,
            "n": len(selected),
        }
        for key in keys:
            item[key] = float(np.mean([row[key] for row in selected]))
        output.append(item)
    return output


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


def main():
    parser = argparse.ArgumentParser(description="Benchmark-protocol free-running KV generation on local GSM8K")
    parser.add_argument("--base-experiment-root", default="/home/yezhe/伪查询/runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234")
    parser.add_argument("--sender-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-4B")
    parser.add_argument("--data", default="/home/yezhe/数据集/gsm8k/test.jsonl")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--methods", default="native,mse_only,mse_then_ce,paper_rec_then_mixed_generation,q_aware_functional")
    parser.add_argument("--cache-variants", default="correct,zero,shuffled,mismatched")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--stop-strings", nargs="*", default=["Question:", "</s>", "<|im_end|>"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    if args.device == "cpu" and args.dtype in {"float16", "bfloat16"}:
        raise ValueError("float16/bfloat16 on CPU is unsupported")
    add_import_roots(args.base_experiment_root)
    from paper_dense_common import assert_tokenizer_compatible
    from real_kv_common import extract_cache
    from real_kv_translator import load_real_translator

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device_obj = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    cache_variants = [v.strip() for v in args.cache_variants.split(",") if v.strip()]

    rows = load_rows(args.data, args.max_samples)
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(sender_tokenizer, receiver_tokenizer, rows, args.max_source_tokens, max_checks=min(8, len(rows)))
    if receiver_tokenizer.pad_token_id is None:
        receiver_tokenizer.pad_token = receiver_tokenizer.eos_token

    sender = load_model(args.sender_model, dtype, args.device_obj)
    receiver = load_model(args.receiver_model, dtype, args.device_obj)
    adapters = {}
    adapter_metadata = {}
    for method in methods:
        if method == "native":
            continue
        adapter, metadata = load_real_translator(checkpoint_for(args.base_experiment_root, method), map_location=args.device_obj)
        adapters[method] = adapter.to(args.device_obj).eval()
        adapter_metadata[method] = metadata
    for module in [sender, receiver, *adapters.values()]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mismatched = build_mismatched_indices(rows)
    rng = torch.Generator(device=args.device_obj)
    rng.manual_seed(args.seed)

    per_sample = []
    with torch.no_grad():
        source_ids_all = []
        sender_pairs_all = []
        translated_by_method = {method: [] for method in methods if method != "native"}
        native_prompt_ids_all = []
        continuation_ids_all = []
        gold_all = []
        for row in tqdm(rows, desc="prefill"):
            source_text, continuation_prompt, native_prompt = split_benchmark_prompt(row["question"])
            source_ids = receiver_tokenizer(
                source_text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_source_tokens,
            ).input_ids.to(args.device_obj)
            continuation_ids = receiver_tokenizer(continuation_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device_obj)
            native_prompt_ids = receiver_tokenizer(native_prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(args.device_obj)
            source_ids_all.append(source_ids)
            continuation_ids_all.append(continuation_ids)
            native_prompt_ids_all.append(native_prompt_ids)
            gold_all.append(gold_final_answer(row["answer"]))
            sender_pairs = extract_cache(sender(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            sender_pairs_all.append(sender_pairs)
            for method, adapter in adapters.items():
                translated_by_method[method].append(clone_pairs(adapter(sender_pairs)))

        for sample, row in enumerate(tqdm(rows, desc="generate")):
            gold = gold_all[sample]
            for method in methods:
                if method == "native":
                    generated_ids, generated_text, stopped_by_eos = greedy_generate_with_cache(
                        receiver, receiver_tokenizer, native_prompt_ids_all[sample], None, args
                    )
                    pred_strict = extract_strict(generated_text)
                    pred_flexible = extract_flexible(generated_text)
                    per_sample.append(
                        {
                            "sample": sample,
                            "method": "native",
                            "cache_variant": "text",
                            "input_regime": "native_text",
                            "question": row["question"],
                            "generated_text": generated_text,
                            "pred_final_answer_strict": pred_strict,
                            "pred_final_answer_flexible": pred_flexible,
                            "gold_final_answer": gold,
                            "em_strict": float(pred_strict == gold),
                            "em_flexible": float(pred_flexible == gold),
                            "answer_f1": answer_f1(pred_flexible, gold),
                            "oracle_contains_gold": float(contains_gold(generated_text, gold)),
                            "generated_tokens": len(generated_ids),
                            "stopped_by_eos": float(stopped_by_eos),
                        }
                    )
                    continue

                base_pairs = translated_by_method[method][sample]
                for variant in cache_variants:
                    if variant == "correct":
                        pairs = base_pairs
                    elif variant == "zero":
                        pairs = make_zero_pairs(base_pairs)
                    elif variant == "shuffled":
                        pairs = make_shuffled_pairs(base_pairs, rng)
                    elif variant == "mismatched":
                        pairs = translated_by_method[method][mismatched[sample]]
                    else:
                        raise ValueError(variant)
                    generated_ids, generated_text, stopped_by_eos = greedy_generate_with_cache(
                        receiver, receiver_tokenizer, continuation_ids_all[sample], pairs, args
                    )
                    pred_strict = extract_strict(generated_text)
                    pred_flexible = extract_flexible(generated_text)
                    per_sample.append(
                        {
                            "sample": sample,
                            "method": method,
                            "cache_variant": variant,
                            "input_regime": "cache_only",
                            "question": row["question"],
                            "generated_text": generated_text,
                            "pred_final_answer_strict": pred_strict,
                            "pred_final_answer_flexible": pred_flexible,
                            "gold_final_answer": gold,
                            "em_strict": float(pred_strict == gold),
                            "em_flexible": float(pred_flexible == gold),
                            "answer_f1": answer_f1(pred_flexible, gold),
                            "oracle_contains_gold": float(contains_gold(generated_text, gold)),
                            "generated_tokens": len(generated_ids),
                            "stopped_by_eos": float(stopped_by_eos),
                        }
                    )

    summary = summarize(per_sample)
    write_jsonl(out / "per_sample_generation.jsonl", per_sample)
    write_csv(out / "free_running_summary.csv", summary)
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "args": {key: value for key, value in vars(args).items() if key != "device_obj"},
                "methods": methods,
                "cache_variants": cache_variants,
                "adapter_metadata": adapter_metadata,
                "outputs": {
                    "per_sample_generation": "per_sample_generation.jsonl",
                    "free_running_summary": "free_running_summary.csv",
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
