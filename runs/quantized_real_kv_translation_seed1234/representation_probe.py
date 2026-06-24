import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from quant_kv_common import (
    alignment_plan,
    attention_route_rows,
    dtype_from_name,
    geometry_rows,
    load_quantized_model,
    mean_finite,
    write_json,
)

import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
sys.path.insert(0, str(REAL_ROOT))

from real_kv_common import (  # noqa: E402
    assert_tokenizer_compatible,
    build_example,
    extract_cache,
    load_rows,
)


def main():
    parser = argparse.ArgumentParser(description="Untrained cross-model KV geometry probe")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--sender-precision", choices=["fp16", "int4"], required=True)
    parser.add_argument("--receiver-precision", choices=["fp16", "int4"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--attention-topk", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    rows = load_rows(args.data, args.max_samples)
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(
        sender_tokenizer, receiver_tokenizer, rows, args.max_context_tokens, min(8, len(rows))
    )
    sender, sender_audit = load_quantized_model(
        args.sender_model, args.sender_precision, dtype, device, eager=True
    )
    receiver, receiver_audit = load_quantized_model(
        args.receiver_model, args.receiver_precision, dtype, device, eager=True
    )
    plan = alignment_plan(sender.config, receiver.config)
    per_layer = []
    per_example = []
    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc="geometry")):
            example = build_example(receiver_tokenizer, row, args.max_context_tokens)
            context_ids = example["context_ids"].to(device)
            sender_out = sender(
                input_ids=context_ids, use_cache=True, output_attentions=True, logits_to_keep=1
            )
            receiver_out = receiver(
                input_ids=context_ids, use_cache=True, output_attentions=True, logits_to_keep=1
            )
            sender_pairs = extract_cache(sender_out.past_key_values, cpu=True)
            receiver_pairs = extract_cache(receiver_out.past_key_values, cpu=True)
            geometry = geometry_rows(sender_pairs, receiver_pairs, plan, args.knn_k)
            routes = attention_route_rows(
                [item.cpu() for item in sender_out.attentions],
                [item.cpu() for item in receiver_out.attentions],
                plan["layer_map"],
                args.attention_topk,
            )
            route_by_layer = {item["receiver_layer"]: item for item in routes}
            combined = []
            for item in geometry:
                merged = {
                    "sample": sample,
                    "id": example["id"],
                    **item,
                    **{
                        key: value
                        for key, value in route_by_layer[item["receiver_layer"]].items()
                        if key.startswith("self_attention")
                    },
                }
                combined.append(merged)
                per_layer.append(merged)
            keys = [key for key in combined[0] if key not in {"sample", "id", "receiver_layer", "sender_layer"}]
            per_example.append(
                {"sample": sample, "id": example["id"], **{key: mean_finite(combined, key) for key in keys}}
            )
    metric_keys = [key for key in per_example[0] if key not in {"sample", "id"}]
    summary = {key: mean_finite(per_example, key) for key in metric_keys}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (("per_layer.jsonl", per_layer), ("per_example.jsonl", per_example)):
        with open(out / name, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    write_json(
        out / "summary.json",
        {
            "args": vars(args),
            "alignment": plan,
            "sender_quantization": sender_audit,
            "receiver_quantization": receiver_audit,
            "metric_scope": {
                "geometry": "cross-model relation geometry; no receiver readability claim",
                "self_attention_route": "sender-native route versus receiver-native route",
                "receiver_read_metrics": "not computed before translator",
            },
            "summary": summary,
        },
    )


if __name__ == "__main__":
    main()
