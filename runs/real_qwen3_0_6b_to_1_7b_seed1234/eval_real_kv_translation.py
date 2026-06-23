import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_kv_common import (
    assert_tokenizer_compatible,
    build_example,
    evaluate_translated_context,
    extract_cache,
    head_dim_from_config,
    load_rows,
    native_cache_equivalence,
    rope_roundtrip_error,
    rope_theta_from_config,
    summarize,
)
from real_kv_translator import load_real_translator


def load_model(path, dtype, device, eager=False):
    kwargs = {"dtype": dtype, "trust_remote_code": True}
    if eager:
        kwargs["attn_implementation"] = "eager"
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Evaluate real cross-model context KV translation")
    parser.add_argument("--sender-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl")
    parser.add_argument("--translator-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method-label", default="pure_translate")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--attention-topk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--tokenizer-check-samples", type=int, default=8)
    parser.add_argument("--equivalence-atol", type=float, default=None)
    args = parser.parse_args()
    if args.device == "cpu" and args.dtype == "float16":
        raise ValueError("float16 on CPU is unsupported")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device_obj = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    rows = load_rows(args.data, args.max_samples)
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(sender_tokenizer, receiver_tokenizer, rows, args.max_context_tokens, args.tokenizer_check_samples)
    tokenizer = receiver_tokenizer
    sender = load_model(args.sender_model, dtype, args.device_obj)
    receiver = load_model(args.receiver_model, dtype, args.device_obj, eager=True)
    translator, translator_metadata = load_real_translator(args.translator_checkpoint, map_location=args.device_obj)
    translator = translator.to(args.device_obj).eval()
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    rope_checks = {
        "sender_rope_roundtrip_max_abs": rope_roundtrip_error(
            (1, sender.config.num_key_value_heads, 8, head_dim_from_config(sender.config)),
            rope_theta_from_config(sender.config),
            args.device_obj,
            dtype,
        ),
        "receiver_rope_roundtrip_max_abs": rope_roundtrip_error(
            (1, receiver.config.num_key_value_heads, 8, head_dim_from_config(receiver.config)),
            rope_theta_from_config(receiver.config),
            args.device_obj,
            dtype,
        ),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    all_layers = []
    equivalence = []
    atol = args.equivalence_atol if args.equivalence_atol is not None else (0.25 if args.dtype == "float16" else 1e-3)
    with torch.no_grad():
        for idx, row in enumerate(tqdm(rows, desc="samples")):
            example = build_example(tokenizer, row, args.max_context_tokens)
            context_ids = example["context_ids"].to(args.device_obj)
            eq = native_cache_equivalence(
                receiver,
                context_ids,
                example["tail_ids"].to(args.device_obj),
                example["query_ids"].shape[1],
                example["answer_ids"].shape[1],
                atol,
            )
            equivalence.append({"sample": idx, "id": example["id"], **eq})
            sender_context = extract_cache(sender(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values)
            receiver_context = extract_cache(receiver(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values)
            translated = translator(sender_context)
            result_row, layer_rows = evaluate_translated_context(receiver, tokenizer, example, receiver_context, translated, args, idx)
            all_results.append(result_row)
            all_layers.extend(layer_rows)
    summary = summarize(all_results)
    with open(out_dir / "per_example.jsonl", "w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "per_layer.jsonl", "w", encoding="utf-8") as f:
        for row in all_layers:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "args": {k: v for k, v in vars(args).items() if k != "device_obj"},
        "translator_metadata": translator_metadata,
        "rope_checks": rope_checks,
        "native_context_cache_equivalence": equivalence,
        "diagnostic_table": summary,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if summary:
        fieldnames = sorted({key for row in summary for key in row})
        with open(out_dir / "diagnostic_table.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
