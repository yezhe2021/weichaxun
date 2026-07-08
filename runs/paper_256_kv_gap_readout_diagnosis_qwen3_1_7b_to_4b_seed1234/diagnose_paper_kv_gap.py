import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from paper_dense_common import (
    assert_tokenizer_compatible,
    build_paper_example,
    cosine_mean,
    js_divergence,
    load_rows,
    mean_metric,
    offline_readout,
    topk_overlap,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
for path in (PROJECT_ROOT, REAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_kv_common import extract_cache  # noqa: E402
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


def kv_joint_consistency(k, v):
    return torch.matmul(k.float(), v.float().transpose(-1, -2))


def norm_ratio(translated, native):
    return (translated.float().norm() / native.float().norm().clamp_min(1e-12)).item()


def layer_kv_gap_rows(sample, method, native_pairs, translated_pairs):
    rows = []
    for layer, ((nk, nv), (tk, tv)) in enumerate(zip(native_pairs, translated_pairs)):
        k_mse = F.mse_loss(tk.float(), nk.float()).item()
        v_mse = F.mse_loss(tv.float(), nv.float()).item()
        k_cos = cosine_mean(nk, tk)
        v_cos = cosine_mean(nv, tv)
        rows.append(
            {
                "sample": sample,
                "method": method,
                "layer": layer,
                "k_mse": k_mse,
                "v_mse": v_mse,
                "kv_mse": 0.5 * (k_mse + v_mse),
                "k_cos": k_cos,
                "v_cos": v_cos,
                "k_norm_ratio": norm_ratio(tk, nk),
                "v_norm_ratio": norm_ratio(tv, nv),
                "kv_joint_consistency": cosine_mean(kv_joint_consistency(nk, nv), kv_joint_consistency(tk, tv)),
            }
        )
    return rows


class AnswerQueryCapture:
    def __init__(self, receiver, keep_device=True):
        self.query_states = {}
        self.handles = []
        self.keep_device = keep_device
        for layer_idx, layer in enumerate(receiver.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._hook(layer_idx), with_kwargs=True))

    def _hook(self, layer_idx):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                hidden_states = args[0]
            input_shape = hidden_states.shape[:-1]
            q = module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim)
            if hasattr(module, "q_norm"):
                q = module.q_norm(q)
            q = q.transpose(1, 2)
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is not None:
                from paper_dense_common import apply_rotary_pos_emb, rotate_half

                cos, sin = position_embeddings
                if apply_rotary_pos_emb is not None:
                    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
                else:
                    q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
            q = q.detach()
            self.query_states[layer_idx] = q if self.keep_device else q.float().cpu()
            return None

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def capture_answer_queries(receiver, input_ids):
    capture = AnswerQueryCapture(receiver, keep_device=True)
    try:
        receiver(input_ids=input_ids, use_cache=False)
    finally:
        capture.close()
    return capture.query_states


def answer_prompt_query_states(receiver, tokenizer, example, args):
    source_ids = example["source_ids"].to(args.device_obj)
    answer_prompt_ids = tokenizer("Answer:", return_tensors="pt", add_special_tokens=False).input_ids.to(args.device_obj)
    input_ids = torch.cat([source_ids, answer_prompt_ids], dim=1)
    query_states = capture_answer_queries(receiver, input_ids)
    answer_prompt_len = answer_prompt_ids.shape[1]
    prefix_len = input_ids.shape[1] - answer_prompt_len + 1
    return query_states, prefix_len, answer_prompt_len


def readout_gap_rows(sample, method, query_states, native_pairs, translated_pairs, receiver, prefix_len, answer_len, topk):
    native_routes, native_outputs = offline_readout(
        query_states,
        native_pairs,
        receiver.config.num_attention_heads,
        prefix_len,
        answer_len,
    )
    translated_routes, translated_outputs = offline_readout(
        query_states,
        translated_pairs,
        receiver.config.num_attention_heads,
        prefix_len,
        answer_len,
    )
    rows = []
    for layer in sorted(native_routes):
        native_route = native_routes[layer].detach().float().cpu()
        translated_route = translated_routes[layer].detach().float().cpu()
        native_output = native_outputs[layer].detach().float().cpu()
        translated_output = translated_outputs[layer].detach().float().cpu()
        output_cos = cosine_mean(native_output, translated_output)
        output_mse = F.mse_loss(translated_output, native_output).item()
        rows.append(
            {
                "sample": sample,
                "method": method,
                "layer": layer,
                "query_position": "answer_prompt",
                "route_topk_overlap": topk_overlap(native_route, translated_route, topk),
                "attention_js": js_divergence(native_route, translated_route),
                "attention_output_cos": output_cos,
                "attention_output_mse": output_mse,
                "readout_loss": output_mse + (1.0 - output_cos),
            }
        )
    return rows


def summarize_by_layer(rows, keys):
    output = []
    layers = sorted({row["layer"] for row in rows})
    for layer in layers:
        selected = [row for row in rows if row["layer"] == layer]
        item = {"layer": layer, "n": len(selected)}
        for key in keys:
            values = [row[key] for row in selected if key in row and np.isfinite(row[key])]
            if values:
                item[key] = float(np.mean(values))
        output.append(item)
    return output


def summarize_overall(kv_rows, readout_rows, method, adapter_checkpoint, n_samples):
    keys = [
        "k_mse",
        "v_mse",
        "kv_mse",
        "k_cos",
        "v_cos",
        "k_norm_ratio",
        "v_norm_ratio",
        "kv_joint_consistency",
    ]
    readout_keys = [
        "route_topk_overlap",
        "attention_js",
        "attention_output_cos",
        "attention_output_mse",
        "readout_loss",
    ]
    row = {"method": method, "adapter_checkpoint": adapter_checkpoint, "n_samples": n_samples}
    for key in keys:
        row[key] = mean_metric(kv_rows, key)
    for key in readout_keys:
        row[key] = mean_metric(readout_rows, key)
    if kv_rows:
        per_layer = summarize_by_layer(kv_rows, keys)
        worst = max(per_layer, key=lambda item: item.get("kv_mse", -math.inf))
        best = min(per_layer, key=lambda item: item.get("kv_mse", math.inf))
        row["worst_kv_mse_layer"] = worst["layer"]
        row["worst_kv_mse"] = worst.get("kv_mse")
        row["best_kv_mse_layer"] = best["layer"]
        row["best_kv_mse"] = best.get("kv_mse")
    if readout_rows:
        per_layer = summarize_by_layer(readout_rows, readout_keys)
        worst = min(per_layer, key=lambda item: item.get("attention_output_cos", math.inf))
        row["worst_readout_cos_layer"] = worst["layer"]
        row["worst_attention_output_cos"] = worst.get("attention_output_cos")
    return [row]


def main():
    parser = argparse.ArgumentParser(description="Diagnose translated KV gap and receiver readout gap for one paper checkpoint")
    parser.add_argument("--sender-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-4B")
    parser.add_argument("--data", default="/home/yezhe/数据集/gsm8k/test.jsonl")
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--method-label", default="paper_256_e1e5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-source-tokens", type=int, default=128)
    parser.add_argument("--answer-mode", choices=["full", "final_only"], default="full")
    parser.add_argument("--attention-topk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--tokenizer-check-samples", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cpu" and args.dtype in {"float16", "bfloat16"}:
        raise ValueError(f"{args.dtype} on CPU is unsupported")
    torch.manual_seed(args.seed)
    args.device_obj = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    rows = load_rows(args.data, args.max_samples)
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(sender_tokenizer, receiver_tokenizer, rows, args.max_source_tokens, args.tokenizer_check_samples)

    sender = load_model(args.sender_model, dtype, args.device_obj)
    receiver = load_model(args.receiver_model, dtype, args.device_obj)
    adapter, adapter_metadata = load_real_translator(args.adapter_checkpoint, map_location=args.device_obj)
    adapter = adapter.to(args.device_obj).eval()
    for module in (sender, receiver, adapter):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kv_rows = []
    readout_rows = []
    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc=args.method_label)):
            example = build_paper_example(receiver_tokenizer, row, args.max_source_tokens, args.answer_mode)
            source_ids = example["source_ids"].to(args.device_obj)
            sender_pairs = extract_cache(sender(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            native_pairs = extract_cache(receiver(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            translated_pairs = adapter(sender_pairs)

            kv_rows.extend(layer_kv_gap_rows(sample, args.method_label, native_pairs, translated_pairs))
            query_states, prefix_len, answer_len = answer_prompt_query_states(receiver, receiver_tokenizer, example, args)
            readout_rows.extend(
                readout_gap_rows(
                    sample,
                    args.method_label,
                    query_states,
                    native_pairs,
                    translated_pairs,
                    receiver,
                    prefix_len,
                    answer_len,
                    args.attention_topk,
                )
            )

    kv_keys = [
        "k_mse",
        "v_mse",
        "kv_mse",
        "k_cos",
        "v_cos",
        "k_norm_ratio",
        "v_norm_ratio",
        "kv_joint_consistency",
    ]
    readout_keys = [
        "route_topk_overlap",
        "attention_js",
        "attention_output_cos",
        "attention_output_mse",
        "readout_loss",
    ]
    per_layer_kv = summarize_by_layer(kv_rows, kv_keys)
    per_layer_readout = summarize_by_layer(readout_rows, readout_keys)
    summary = summarize_overall(kv_rows, readout_rows, args.method_label, args.adapter_checkpoint, len(rows))

    write_jsonl(out / "per_sample_layer_kv_gap.jsonl", kv_rows)
    write_jsonl(out / "per_sample_layer_readout_gap.jsonl", readout_rows)
    write_csv(out / "per_layer_kv_gap.csv", per_layer_kv)
    write_csv(out / "per_layer_readout_gap.csv", per_layer_readout)
    write_csv(out / "kv_readout_gap_summary.csv", summary)
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "args": {k: str(v) if k == "device_obj" else v for k, v in vars(args).items()},
                "adapter_metadata": adapter_metadata,
                "outputs": {
                    "summary": "kv_readout_gap_summary.csv",
                    "per_layer_kv_gap": "per_layer_kv_gap.csv",
                    "per_layer_readout_gap": "per_layer_readout_gap.csv",
                    "per_sample_layer_kv_gap": "per_sample_layer_kv_gap.jsonl",
                    "per_sample_layer_readout_gap": "per_sample_layer_readout_gap.jsonl",
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps(summary[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
