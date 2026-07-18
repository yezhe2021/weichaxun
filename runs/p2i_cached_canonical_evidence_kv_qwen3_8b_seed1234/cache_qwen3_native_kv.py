import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from canonical_modules import decoder_layers
from p2i_common import load_jsonl, parse_dtype, resolve_device


def sender_text(row):
    text = (
        f"QUESTION\n{row['question']}\n\nEVIDENCE A\n{row['evidence_a']}\n\n"
        f"EVIDENCE B\n{row['evidence_b']}"
    )
    a_start = text.index(row["evidence_a"])
    a_end = a_start + len(row["evidence_a"])
    b_start = text.index(row["evidence_b"], a_end)
    return text, ((a_start, a_end), (b_start, b_start + len(row["evidence_b"])))


def selected_indices(offsets, spans):
    selected = [
        index for index, (start, end) in enumerate(offsets)
        if start != end and any(start < right and end > left for left, right in spans)
    ]
    if not selected:
        raise ValueError("No evidence tokens selected")
    return selected


@torch.inference_mode()
def encode(model, tokenizer, row, device, max_length):
    text, spans = sender_text(row)
    encoded = tokenizer(
        text, return_tensors="pt", return_offsets_mapping=True,
        add_special_tokens=True, truncation=False,
    )
    if encoded.input_ids.shape[1] > max_length:
        raise ValueError(f"{row['id']} exceeds max_length")
    offsets = encoded.offset_mapping[0].tolist()
    indices = selected_indices(offsets, spans)
    answer_start = text.find(row["answer"], spans[1][0])
    if answer_start < 0:
        raise ValueError(f"Answer {row['answer']!r} is absent from Evidence B")
    answer_end = answer_start + len(row["answer"])
    answer_mask = torch.tensor([
        offsets[index][0] < answer_end and offsets[index][1] > answer_start for index in indices
    ], dtype=torch.bool)
    keys = [None] * len(decoder_layers(model))
    values = [None] * len(decoder_layers(model))
    handles = []
    for layer_index, layer in enumerate(decoder_layers(model)):
        attention = layer.self_attn

        def hook(module, args, kwargs, layer_index=layer_index):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            if hasattr(module, "k_norm"):
                key = module.k_norm(key)
            value = module.v_proj(hidden).view(shape)
            keys[layer_index] = key.transpose(1, 2)[0, :, indices].detach().half().cpu()
            values[layer_index] = value.transpose(1, 2)[0, :, indices].detach().half().cpu()

        handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        model(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in keys + values):
        raise RuntimeError("Native KV capture was incomplete")
    return {
        **row,
        "memory": {"keys": keys, "values": values, "answer_token_mask": answer_mask},
        "evidence_token_ids": encoded.input_ids[0, indices].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Cache full Qwen3 pre-RoPE K/native V")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    grouped = defaultdict(dict)
    for row in load_jsonl(args.data):
        grouped[row["pair_id"]][row["variant"]] = row
    pairs = [value for value in grouped.values() if {"base", "counterfactual"}.issubset(value)]
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    pair_files = []
    for index, pair in enumerate(tqdm(pairs, desc="cache_qwen3_native_kv")):
        examples = [encode(model, tokenizer, pair[variant], device, args.max_length) for variant in ("base", "counterfactual")]
        filename = f"pair_{index:05d}.pt"
        torch.save({"pair_id": examples[0]["pair_id"], "examples": examples}, output / filename)
        pair_files.append({
            "pair_id": examples[0]["pair_id"], "file": filename,
            "base_answer": examples[0]["answer"], "counterfactual_answer": examples[1]["answer"],
        })
    config = model.config
    metadata = {
        "format_version": 3,
        "coordinate_system": "pre_rope_qk_native_v",
        "model": args.model,
        "data": args.data,
        "pairs": len(pair_files),
        "layers": int(config.num_hidden_layers),
        "query_heads": int(config.num_attention_heads),
        "kv_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "pair_files": pair_files,
    }
    with open(output / "index.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    with open(output / "CACHE_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", **metadata}, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
