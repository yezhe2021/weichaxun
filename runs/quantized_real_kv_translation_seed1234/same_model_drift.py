import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from quant_kv_common import (
    alignment_plan,
    dtype_from_name,
    geometry_rows,
    load_quantized_model,
    mean_finite,
    write_json,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
sys.path.insert(0, str(REAL_ROOT))

from real_kv_common import build_example, extract_cache, load_rows  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Same-model FP16 versus INT4 KV drift")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device, dtype = torch.device(args.device), dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = load_rows(args.data, args.max_samples)
    fp16, fp16_audit = load_quantized_model(args.model, "fp16", dtype, device)
    int4, int4_audit = load_quantized_model(args.model, "int4", dtype, device)
    plan = alignment_plan(fp16.config, int4.config)
    per_example, per_layer = [], []
    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc=f"{args.model_label} drift")):
            example = build_example(tokenizer, row, args.max_context_tokens)
            ids = example["context_ids"].to(device)
            fp16_pairs = extract_cache(fp16(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values, cpu=True)
            int4_pairs = extract_cache(int4(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values, cpu=True)
            layer_rows = geometry_rows(fp16_pairs, int4_pairs, plan, args.knn_k)
            for item in layer_rows:
                per_layer.append({"sample": sample, "id": example["id"], **item})
            keys = [key for key in layer_rows[0] if key not in {"receiver_layer", "sender_layer"}]
            per_example.append(
                {"sample": sample, "id": example["id"], **{key: mean_finite(layer_rows, key) for key in keys}}
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (("per_layer.jsonl", per_layer), ("per_example.jsonl", per_example)):
        with open(out / name, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    keys = [key for key in per_example[0] if key not in {"sample", "id"}]
    write_json(
        out / "summary.json",
        {
            "args": vars(args),
            "fp16_quantization": fp16_audit,
            "int4_quantization": int4_audit,
            "summary": {key: mean_finite(per_example, key) for key in keys},
        },
    )


if __name__ == "__main__":
    main()
