import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from translated_kv_diagnostics import (  # noqa: E402
    answer_logits,
    build_example,
    cache_metrics,
    capture_attention,
    distribution_metrics,
    extract_cache,
    flatten_context,
    load_rows,
    mean_metric,
    attention_metrics,
)


def make_cache(pairs, config):
    return DynamicCache(ddp_cache_data=[(k, v) for k, v in pairs], config=config)


def head_dim_from_config(config):
    value = getattr(config, "head_dim", None)
    if value is not None:
        return int(value)
    return int(config.hidden_size // config.num_attention_heads)


def rope_theta_from_config(config):
    direct = getattr(config, "rope_theta", None)
    if direct is not None:
        return float(direct)
    parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    if isinstance(parameters, dict):
        for key in ("rope_theta", "theta", "base"):
            if key in parameters:
                return float(parameters[key])
        for value in parameters.values():
            if isinstance(value, dict):
                for key in ("rope_theta", "theta", "base"):
                    if key in value:
                        return float(value[key])
    return 10_000.0


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_transform(k, rope_theta, inverse=False, offset=0):
    dim = k.shape[-1]
    if dim % 2 != 0:
        raise ValueError(f"RoPE head dim must be even, got {dim}")
    positions = torch.arange(offset, offset + k.shape[-2], device=k.device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=k.device, dtype=torch.float32) / dim))
    angles = torch.outer(positions, inv_freq)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(k.dtype).view(1, 1, k.shape[-2], dim)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(k.dtype).view(1, 1, k.shape[-2], dim)
    if inverse:
        sin = -sin
    return k * cos + rotate_half(k) * sin


def rope_roundtrip_error(shape, rope_theta, device, dtype):
    k = torch.randn(shape, device=device, dtype=dtype)
    restored = rope_transform(rope_transform(k, rope_theta, inverse=True), rope_theta, inverse=False)
    return (restored.float() - k.float()).abs().max().item()


def fixed_index_map(source_count, target_count):
    if source_count <= 0 or target_count <= 0:
        raise ValueError("source_count and target_count must be positive")
    if target_count == 1:
        return [0]
    return [round(i * (source_count - 1) / (target_count - 1)) for i in range(target_count)]


def tokenizer_signature(tokenizer):
    return {
        "class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "len": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "special_tokens_map": tokenizer.special_tokens_map,
        "chat_template": getattr(tokenizer, "chat_template", None),
    }


def assert_tokenizer_compatible(sender_tokenizer, receiver_tokenizer, rows, max_context_tokens, max_checks=8):
    sender_sig = tokenizer_signature(sender_tokenizer)
    receiver_sig = tokenizer_signature(receiver_tokenizer)
    strict_keys = [
        "vocab_size",
        "len",
        "eos_token",
        "eos_token_id",
        "pad_token",
        "pad_token_id",
        "bos_token",
        "bos_token_id",
        "unk_token",
        "unk_token_id",
        "special_tokens_map",
        "chat_template",
    ]
    mismatches = [
        key for key in strict_keys
        if json.dumps(sender_sig.get(key), sort_keys=True, ensure_ascii=False)
        != json.dumps(receiver_sig.get(key), sort_keys=True, ensure_ascii=False)
    ]
    if mismatches:
        raise ValueError(
            "Sender/receiver tokenizers are not strictly compatible: "
            + ", ".join(mismatches)
            + f"\nsender={sender_sig}\nreceiver={receiver_sig}"
        )
    for idx, row in enumerate(rows[:max_checks]):
        context_text = f"Context:\n{flatten_context(row['context'])}"
        query_text = f"\n\nQuestion:\n{row['question']}\n\nAnswer:"
        answer_text = " " + str(row["answer"]).strip()
        probes = {
            "context": (context_text, True, max_context_tokens),
            "query": (query_text, False, None),
            "answer": (answer_text, False, None),
            "cqa": (context_text + query_text + answer_text, True, max_context_tokens + 128),
        }
        for name, (text, add_special_tokens, max_len) in probes.items():
            kwargs = {"add_special_tokens": add_special_tokens}
            if max_len is not None:
                kwargs.update({"truncation": True, "max_length": max_len})
            sid = sender_tokenizer(text, **kwargs).input_ids
            rid = receiver_tokenizer(text, **kwargs).input_ids
            if sid != rid:
                raise ValueError(f"Tokenizer id mismatch on sample={idx} probe={name}")


def normalized_kv_mse(native, translated):
    losses = []
    for (nk, nv), (tk, tv) in zip(native, translated):
        losses.append((tk.float() - nk.float()).pow(2).mean() / (nk.float().pow(2).mean() + 1e-8))
        losses.append((tv.float() - nv.float()).pow(2).mean() / (nv.float().pow(2).mean() + 1e-8))
    return torch.stack(losses).mean()


def answer_ce(receiver, context_pairs, tail_ids, query_len, answer_ids):
    cache = make_cache(context_pairs, receiver.config)
    out = receiver(input_ids=tail_ids, past_key_values=cache, use_cache=True)
    logits = answer_logits(out.logits, query_len, answer_ids.shape[1]).float()
    n = min(logits.shape[1], answer_ids.shape[1])
    return F.cross_entropy(logits[:, :n].reshape(-1, logits.shape[-1]), answer_ids[:, :n].reshape(-1))


def native_cache_equivalence(receiver, context_ids, tail_ids, query_len, answer_len, dtype_atol):
    with torch.no_grad():
        context_out = receiver(input_ids=context_ids, use_cache=True, logits_to_keep=1)
        context_pairs = extract_cache(context_out.past_key_values, detach=True, cpu=False)
        native_cache = make_cache(context_pairs, receiver.config)
        tail_out = receiver(input_ids=tail_ids, past_key_values=native_cache, use_cache=True)
        tail_logits = answer_logits(tail_out.logits, query_len, answer_len).float()
        full_ids = torch.cat([context_ids, tail_ids], dim=1)
        full_out = receiver(input_ids=full_ids, use_cache=False)
        start = context_ids.shape[1] + query_len - 1
        full_logits = full_out.logits[:, start:start + answer_len].float()
    n = min(tail_logits.shape[1], full_logits.shape[1])
    return {
        "max_abs_logit": (tail_logits[:, :n] - full_logits[:, :n]).abs().max().item(),
        "top1_match": (tail_logits[:, :n].argmax(-1) == full_logits[:, :n].argmax(-1)).float().mean().item(),
        "atol": dtype_atol,
        "passed": (tail_logits[:, :n] - full_logits[:, :n]).abs().max().item() <= dtype_atol,
    }


def evaluate_translated_context(receiver, tokenizer, example, receiver_native, translated, args, sample_idx):
    query_ids = example["query_ids"].to(args.device_obj)
    tail_ids = example["tail_ids"].to(args.device_obj)
    answer_ids = example["answer_ids"].to(args.device_obj)
    native_cpu = [(k.detach().float().cpu(), v.detach().float().cpu()) for k, v in receiver_native]
    translated_cpu = [(k.detach().float().cpu(), v.detach().float().cpu()) for k, v in translated]
    with torch.no_grad():
        with capture_attention(receiver) as native_capture:
            native_cache = make_cache(receiver_native, receiver.config)
            native_tail = receiver(input_ids=tail_ids, past_key_values=native_cache, use_cache=True)
        native_logits = answer_logits(native_tail.logits, query_ids.shape[1], answer_ids.shape[1]).float().cpu()
        with capture_attention(receiver) as translated_capture:
            translated_cache = make_cache(translated, receiver.config)
            translated_tail = receiver(input_ids=tail_ids, past_key_values=translated_cache, use_cache=True)
        translated_logits = answer_logits(translated_tail.logits, query_ids.shape[1], answer_ids.shape[1]).float().cpu()
    kv_rows = cache_metrics(native_cpu, translated_cpu)
    attn_rows = attention_metrics(
        native_capture.weights,
        translated_capture.weights,
        native_cpu,
        translated_cpu,
        example["context_ids"].shape[1],
        query_ids.shape[1],
        args.attention_topk,
    )
    row = {
        "sample": sample_idx,
        "id": example["id"],
        "method": args.method_label,
        **distribution_metrics(native_logits, translated_logits, answer_ids.cpu()),
        "kv_mse": mean_metric(kv_rows, "kv_mse"),
        "kv_relative_mse": mean_metric(kv_rows, "kv_relative_mse"),
        "k_cos": mean_metric(kv_rows, "k_cos"),
        "v_cos": mean_metric(kv_rows, "v_cos"),
        "kv_joint_consistency": mean_metric(kv_rows, "kv_joint_consistency"),
        "attention_route_overlap": mean_metric(attn_rows, "attention_route_overlap", "answer"),
        "attention_route_js": mean_metric(attn_rows, "attention_route_js", "answer"),
        "attention_output_cos": mean_metric(attn_rows, "attention_output_cos", "answer"),
    }
    layer_rows = []
    for item in kv_rows:
        layer_rows.append({"sample": sample_idx, "id": example["id"], "method": args.method_label, "kind": "cache", **item})
    for item in attn_rows:
        layer_rows.append({"sample": sample_idx, "id": example["id"], "method": args.method_label, "kind": "attention", **item})
    return row, layer_rows


def summarize(rows):
    keys = [
        "kv_mse",
        "kv_relative_mse",
        "k_cos",
        "v_cos",
        "logit_kl",
        "ce_delta",
        "top1_match",
        "attention_route_overlap",
        "attention_route_js",
        "attention_output_cos",
        "kv_joint_consistency",
    ]
    output = []
    rng = np.random.default_rng(1234)
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary = {"method": method, "n": len(selected)}
        for key in keys:
            values = [row[key] for row in selected if key in row and np.isfinite(row[key])]
            if values:
                summary[key] = float(np.mean(values))
                if len(values) > 1:
                    samples = rng.choice(values, size=(1000, len(values)), replace=True).mean(axis=1)
                    summary[f"{key}_ci95_low"] = float(np.percentile(samples, 2.5))
                    summary[f"{key}_ci95_high"] = float(np.percentile(samples, 97.5))
        output.append(summary)
    return output


__all__ = [
    "answer_ce",
    "assert_tokenizer_compatible",
    "build_example",
    "evaluate_translated_context",
    "extract_cache",
    "fixed_index_map",
    "head_dim_from_config",
    "load_rows",
    "make_cache",
    "native_cache_equivalence",
    "normalized_kv_mse",
    "rope_roundtrip_error",
    "rope_theta_from_config",
    "rope_transform",
    "summarize",
]
