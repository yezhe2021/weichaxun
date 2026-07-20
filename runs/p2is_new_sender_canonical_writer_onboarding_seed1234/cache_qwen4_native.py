import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2is_common import PairCache, file_sha256, parse_dtype, resolve_device, write_json


def sender_text(row):
    text = f"QUESTION\n{row['question']}\n\nEVIDENCE A\n{row['evidence_a']}\n\nEVIDENCE B\n{row['evidence_b']}"
    a = text.index(row["evidence_a"]); b = text.index(row["evidence_b"], a + len(row["evidence_a"]))
    return text, ((a, a + len(row["evidence_a"])), (b, b + len(row["evidence_b"])))


def evidence_indices(offsets, spans):
    return [index for index, (start, end) in enumerate(offsets) if start != end and any(start < right and end > left for left, right in spans)]


@torch.inference_mode()
def encode(model, tokenizer, row, device):
    text, spans = sender_text(row)
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True, truncation=False)
    indices = evidence_indices(encoded.offset_mapping[0].tolist(), spans)
    token_ids = encoded.input_ids[0, indices].tolist()
    if token_ids != row["evidence_token_ids"]:
        raise RuntimeError(f"Strict Qwen3-4B/Qwen3-8B token alignment failed for {row['id']}")
    captured = {}
    attention = model.model.layers[-1].self_attn
    def hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        key = module.k_proj(hidden).view(shape)
        if hasattr(module, "k_norm"): key = module.k_norm(key)
        value = module.v_proj(hidden).view(shape)
        captured["key"] = key[0, indices].reshape(len(indices), -1).detach().half().cpu()
        captured["value"] = value[0, indices].reshape(len(indices), -1).detach().half().cpu()
    handle = attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        model(input_ids=encoded.input_ids.to(device), attention_mask=encoded.attention_mask.to(device), use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if captured["key"].shape[-1] != 1024 or captured["value"].shape != captured["key"].shape:
        raise RuntimeError(f"Unexpected Qwen3-4B final Native KV shape {captured['key'].shape}")
    return {
        "pair_id": row["pair_id"], "id": row["id"], "variant": row["variant"],
        "question": row["question"], "answer": row["answer"], "token_ids": token_ids,
        "answer_mask": row["memory"]["answer_token_mask"].bool(),
        "key_flat": captured["key"], "value_flat": captured["value"],
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--q8-native-index", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--dtype", default="float16")
    args = parser.parse_args(); device = resolve_device(args.device); dtype = parse_dtype(args.dtype, device)
    source = PairCache(args.q8_native_index, 1)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True, local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); entries = []
    for index in tqdm(range(len(source)), desc="cache_qwen3_4b_final_native"):
        pair = source.load(index); rows = [encode(model, tokenizer, pair[variant], device) for variant in ("base", "counterfactual")]
        filename = f"pair_{index:05d}.pt"; torch.save({"pair_id": rows[0]["pair_id"], "variants": rows}, output / filename)
        entries.append({"pair_id": rows[0]["pair_id"], "file": filename, "base_answer": rows[0]["answer"], "counterfactual_answer": rows[1]["answer"]})
    metadata = {
        "format_version": 1, "model": args.model, "pairs": len(entries), "coordinate_system": "qwen3_4b_final_pre_rope_k_native_v",
        "kv_heads": int(model.config.num_key_value_heads), "head_dim": int(model.config.head_dim), "flat_dim": 1024,
        "q8_native_index_sha256": file_sha256(args.q8_native_index), "strict_token_alignment": True, "pair_files": entries,
    }
    write_json(output / "index.json", metadata); write_json(output / "CACHE_SUCCESS.json", {"status": "complete", **metadata})


if __name__ == "__main__": main()
