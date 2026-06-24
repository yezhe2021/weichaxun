import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from quant_kv_common import dtype_from_name, load_quantized_model, mean_finite, write_json

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
sys.path.insert(0, str(REAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from real_kv_common import (  # noqa: E402
    assert_tokenizer_compatible,
    build_example,
    evaluate_translated_context,
    extract_cache,
    load_rows,
)
from real_kv_translator import load_real_translator  # noqa: E402
from translated_kv_diagnostics import answer_f1, greedy_generate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate translated KV relative to the same receiver native cache")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--sender-precision", choices=["fp16", "int4"], required=True)
    parser.add_argument("--receiver-precision", choices=["fp16", "int4"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--translator-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--attention-topk", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device_obj = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = load_rows(args.data, args.max_samples)
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(
        sender_tokenizer, receiver_tokenizer, rows, args.max_context_tokens, min(8, len(rows))
    )
    sender, sender_audit = load_quantized_model(
        args.sender_model, args.sender_precision, dtype, args.device_obj
    )
    receiver, receiver_audit = load_quantized_model(
        args.receiver_model, args.receiver_precision, dtype, args.device_obj, eager=True
    )
    translator, translator_metadata = load_real_translator(
        args.translator_checkpoint, map_location=args.device_obj
    )
    translator = translator.to(args.device_obj).eval()
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    results, layer_results = [], []
    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc="translation eval")):
            example = build_example(receiver_tokenizer, row, args.max_context_tokens)
            context_ids = example["context_ids"].to(args.device_obj)
            sender_context = extract_cache(
                sender(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values
            )
            receiver_context = extract_cache(
                receiver(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values
            )
            translated = translator(sender_context)
            metric_row, metric_layers = evaluate_translated_context(
                receiver, receiver_tokenizer, example, receiver_context, translated, args, sample
            )
            native_prediction = greedy_generate(
                receiver,
                receiver_tokenizer,
                receiver_context,
                example["query_ids"].to(args.device_obj),
                receiver.config,
                args.max_new_tokens,
            )
            translated_prediction = greedy_generate(
                receiver,
                receiver_tokenizer,
                translated,
                example["query_ids"].to(args.device_obj),
                receiver.config,
                args.max_new_tokens,
            )
            native_ce = metric_row.pop("native_ce")
            translated_ce = metric_row["translated_ce"]
            metric_row.update(
                {
                    "receiver_native_ce": native_ce,
                    "receiver_native_logit_quality": -native_ce,
                    "receiver_native_prediction": native_prediction,
                    "receiver_native_f1": answer_f1(native_prediction, example["answer"]),
                    "translated_prediction": translated_prediction,
                    "translated_f1": answer_f1(translated_prediction, example["answer"]),
                    "ce_delta": translated_ce - native_ce,
                    "top1_match_reference": "same_receiver_native_cache",
                    "attention_reference": "same_receiver_native_cache",
                }
            )
            results.append(metric_row)
            layer_results.extend(metric_layers)

    numeric_keys = [
        key
        for key, value in results[0].items()
        if isinstance(value, (int, float)) and key not in {"sample"}
    ]
    summary = {key: mean_finite(results, key) for key in numeric_keys}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (("per_example.jsonl", results), ("per_layer.jsonl", layer_results)):
        with open(out / name, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    with open(out / "diagnostic_table.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(summary))
        writer.writeheader()
        writer.writerow(summary)
    write_json(
        out / "summary.json",
        {
            "args": {key: value for key, value in vars(args).items() if key != "device_obj"},
            "sender_quantization": sender_audit,
            "receiver_quantization": receiver_audit,
            "translator_metadata": translator_metadata,
            "comparison_rule": {
                "ce_delta": "translated_ce - receiver_native_ce for this exact receiver",
                "top1_match": "translated logits versus this exact receiver native-cache logits",
                "attention_output_cos": "translated cache versus this exact receiver native cache",
            },
            "summary": summary,
        },
    )


if __name__ == "__main__":
    main()
