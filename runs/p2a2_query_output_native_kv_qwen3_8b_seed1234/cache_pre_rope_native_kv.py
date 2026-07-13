import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2a_common import get_layers, load_jsonl, parse_dtype, resolve_device, sender_text


def evidence_indices(offsets, spans):
    selected = []
    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            selected.append(token_index)
    if not selected:
        raise ValueError("No evidence tokens were selected")
    return selected


def answer_mask(text, offsets, selected, answer, evidence_start):
    start = text.find(answer, evidence_start)
    if start < 0:
        raise ValueError(f"Answer {answer!r} is absent from evidence")
    end = start + len(answer)
    mask = [offsets[index][0] < end and offsets[index][1] > start for index in selected]
    if not any(mask):
        raise ValueError(f"No answer tokens selected for {answer!r}")
    return torch.tensor(mask, dtype=torch.bool)


@torch.inference_mode()
def encode_example(model, tokenizer, row, device, max_length):
    text, spans = sender_text(row)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
    )
    if encoded.input_ids.shape[1] > max_length:
        raise ValueError(f"{row['id']} requires {encoded.input_ids.shape[1]} tokens, max_length={max_length}")
    indices = evidence_indices(encoded.offset_mapping[0].tolist(), spans)
    target_mask = answer_mask(text, encoded.offset_mapping[0].tolist(), indices, row["answer"], spans[1][0])
    model_inputs = {
        "input_ids": encoded.input_ids.to(device),
        "attention_mask": encoded.attention_mask.to(device),
    }
    captured_keys = [None] * len(get_layers(model))
    captured_values = [None] * len(get_layers(model))
    handles = []
    for layer_index, layer in enumerate(get_layers(model)):
        attention = layer.self_attn

        def hook(module, args, kwargs, layer_index=layer_index):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
            value = module.v_proj(hidden).view(shape).transpose(1, 2)
            captured_keys[layer_index] = key[0, :, indices, :].detach().to(dtype=torch.float16, device="cpu")
            captured_values[layer_index] = value[0, :, indices, :].detach().to(dtype=torch.float16, device="cpu")

        handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        model(**model_inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured_keys + captured_values):
        raise RuntimeError(f"Failed to capture all layers for {row['id']}")
    return {
        **row,
        "memory": {"keys": captured_keys, "values": captured_values, "answer_token_mask": target_mask},
        "evidence_token_ids": encoded.input_ids[0, indices].tolist(),
        "evidence_tokens": len(indices),
        "sender_tokens": int(encoded.input_ids.shape[1]),
    }


def save_pair(output, pair_index, pair_id, examples):
    name = f"pair_{pair_index:05d}.pt"
    torch.save({"pair_id": pair_id, "examples": examples}, output / name)
    return name


def main():
    parser = argparse.ArgumentParser(description="Cache full per-layer pre-RoPE K and native V for P2-A")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    rows = load_jsonl(args.data, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["variant"]] = row
    pair_rows = [
        (pair_id, variants)
        for pair_id, variants in grouped.items()
        if {"base", "counterfactual"}.issubset(variants)
    ]
    pair_files = []
    token_counts = []
    for pair_index, (pair_id, variants) in enumerate(tqdm(pair_rows, desc="pre_rope_native_kv_pairs")):
        examples = []
        for variant in ("base", "counterfactual"):
            example = encode_example(model, tokenizer, variants[variant], device, args.max_length)
            token_counts.append(example["evidence_tokens"])
            examples.append(example)
        pair_files.append(
            {
                "pair_id": pair_id,
                "file": save_pair(output, pair_index, pair_id, examples),
                "base_answer": variants["base"]["answer"],
                "counterfactual_answer": variants["counterfactual"]["answer"],
            }
        )

    with open(output / "index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "format_version": 3,
                "coordinate_system": "pre_rope_qk_native_v",
                "answer_token_mask": True,
                "model": args.model,
                "data": args.data,
                "n": len(rows),
                "pairs": len(pair_files),
                "layers": int(model.config.num_hidden_layers),
                "query_heads": int(model.config.num_attention_heads),
                "kv_heads": int(model.config.num_key_value_heads),
                "head_dim": int(model.config.head_dim),
                "min_evidence_tokens": min(token_counts),
                "max_evidence_tokens": max(token_counts),
                "pair_files": pair_files,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
