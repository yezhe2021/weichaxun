import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from paper_dense_common import (
    answer_f1,
    answer_query_slice,
    assert_tokenizer_compatible,
    build_paper_example,
    cache_metric_rows,
    cosine_mean,
    distribution_metrics,
    js_divergence,
    load_rows,
    mean_metric,
    offline_readout,
    receiver_cache_reconstruction_loss,
    run_generation,
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


def teacher_forced_prediction(tokenizer, logits):
    ids = logits.argmax(dim=-1)[0]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def attention_output_cos_rows(native_outputs, translated_outputs, prefix_len, answer_len):
    rows = []
    for layer in sorted(native_outputs):
        native = native_outputs[layer].detach().float().cpu()
        translated = translated_outputs[layer].detach().float().cpu()
        seq_len = min(native.shape[1], translated.shape[1])
        sl = answer_query_slice(prefix_len, answer_len, seq_len)
        if sl.stop <= sl.start:
            continue
        rows.append({"layer": layer, "attention_output_cos": cosine_mean(native[:, sl], translated[:, sl])})
    return rows


def readout_probe_rows(native_query_states, native_pairs, translated_pairs, num_attention_heads, prefix_len, answer_len, topk):
    native_routes, native_outputs = offline_readout(native_query_states, native_pairs, num_attention_heads, prefix_len, answer_len)
    translated_routes, translated_outputs = offline_readout(native_query_states, translated_pairs, num_attention_heads, prefix_len, answer_len)
    rows = []
    for layer in sorted(native_routes):
        native_route = native_routes[layer].detach().float().cpu()
        translated_route = translated_routes[layer].detach().float().cpu()
        native_output = native_outputs[layer].detach().float().cpu()
        translated_output = translated_outputs[layer].detach().float().cpu()
        output_mse = F.mse_loss(translated_output, native_output).item()
        output_cos = cosine_mean(native_output, translated_output)
        rows.append(
            {
                "layer": layer,
                "route_overlap": topk_overlap(native_route, translated_route, topk),
                "attention_js": js_divergence(native_route, translated_route),
                "readout_output_cos": output_cos,
                "readout_output_mse": output_mse,
                "readout_loss": output_mse + (1.0 - output_cos),
            }
        )
    return rows


def summarize(rows):
    keys = [
        "receiver_cache_reconstruction_loss",
        "receiver_native_ce",
        "translated_ce",
        "ce_delta",
        "logit_kl",
        "top1_match",
        "answer_f1",
        "attention_output_cos",
        "route_overlap",
        "attention_js",
        "readout_output_cos",
        "readout_output_mse",
        "readout_loss",
        "kv_mse",
        "k_cos",
        "v_cos",
        "kv_joint_consistency",
    ]
    output = []
    for method in sorted({row["method"] for row in rows}):
        for mode in sorted({row["receiver_prompt_mode"] for row in rows if row["method"] == method}):
            selected = [row for row in rows if row["method"] == method and row["receiver_prompt_mode"] == mode]
            item = {"method": method, "receiver_prompt_mode": mode, "n": len(selected)}
            for key in keys:
                values = [row[key] for row in selected if key in row and np.isfinite(row[key])]
                if values:
                    item[key] = float(np.mean(values))
            output.append(item)
    return output


def eval_mode(receiver, tokenizer, example, native_pairs, translated_pairs, mode, args):
    if mode == "context_aware":
        tail_ids = example["aware_tail_ids"].to(args.device_obj)
        prefix_len = example["aware_prefix_len"]
    elif mode == "context_unaware":
        tail_ids = example["unaware_tail_ids"].to(args.device_obj)
        prefix_len = example["unaware_prefix_len"]
    else:
        raise ValueError(mode)
    answer_ids = example["answer_ids"].to(args.device_obj)
    answer_len = answer_ids.shape[1]
    native_logits, native_q, native_attention_outputs = run_generation(
        receiver, native_pairs, tail_ids, prefix_len, answer_len, capture_trace=True
    )
    translated_logits, _, translated_attention_outputs = run_generation(
        receiver, translated_pairs, tail_ids, prefix_len, answer_len, capture_trace=True
    )
    distribution = distribution_metrics(native_logits.detach().float().cpu(), translated_logits.detach().float().cpu(), answer_ids.detach().cpu())
    attn_rows = attention_output_cos_rows(native_attention_outputs, translated_attention_outputs, prefix_len, answer_len)
    readout_rows = readout_probe_rows(
        native_q,
        native_pairs,
        translated_pairs,
        receiver.config.num_attention_heads,
        prefix_len,
        answer_len,
        args.attention_topk,
    )
    prediction = teacher_forced_prediction(tokenizer, translated_logits.detach().float().cpu())
    return {
        **distribution,
        "receiver_prompt_mode": mode,
        "answer_prediction": prediction,
        "answer_prediction_mode": "teacher_forced_token_argmax",
        "answer_f1": answer_f1(prediction, example["answer"]),
        "attention_output_cos": mean_metric(attn_rows, "attention_output_cos"),
        "route_overlap": mean_metric(readout_rows, "route_overlap"),
        "attention_js": mean_metric(readout_rows, "attention_js"),
        "readout_output_cos": mean_metric(readout_rows, "readout_output_cos"),
        "readout_output_mse": mean_metric(readout_rows, "readout_output_mse"),
        "readout_loss": mean_metric(readout_rows, "readout_loss"),
    }, attn_rows, readout_rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate paper-style dense KV cache alignment adapter")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--attention-topk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--tokenizer-check-samples", type=int, default=8)
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
    all_rows = []
    layer_rows = []
    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc=args.method_label)):
            example = build_paper_example(receiver_tokenizer, row, args.max_source_tokens)
            source_ids = example["source_ids"].to(args.device_obj)
            sender_pairs = extract_cache(sender(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            native_pairs = extract_cache(receiver(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            translated_pairs = adapter(sender_pairs)
            rec_loss = receiver_cache_reconstruction_loss(native_pairs, translated_pairs).item()
            kv_rows = cache_metric_rows(
                [(k.detach().float().cpu(), v.detach().float().cpu()) for k, v in native_pairs],
                [(k.detach().float().cpu(), v.detach().float().cpu()) for k, v in translated_pairs],
            )
            for mode in ("context_aware", "context_unaware"):
                result, attn_rows, readout_rows = eval_mode(receiver, receiver_tokenizer, example, native_pairs, translated_pairs, mode, args)
                result.update(
                    {
                        "sample": sample,
                        "id": example["id"],
                        "method": args.method_label,
                        "receiver_cache_reconstruction_loss": rec_loss,
                        "kv_mse": mean_metric(kv_rows, "kv_mse"),
                        "k_cos": mean_metric(kv_rows, "k_cos"),
                        "v_cos": mean_metric(kv_rows, "v_cos"),
                        "kv_joint_consistency": mean_metric(kv_rows, "kv_joint_consistency"),
                    }
                )
                all_rows.append(result)
                for item in attn_rows:
                    layer_rows.append({"sample": sample, "id": example["id"], "method": args.method_label, "receiver_prompt_mode": mode, "kind": "attention_output", **item})
                for item in readout_rows:
                    layer_rows.append({"sample": sample, "id": example["id"], "method": args.method_label, "receiver_prompt_mode": mode, "kind": "readout_probe", **item})
            for item in kv_rows:
                layer_rows.append({"sample": sample, "id": example["id"], "method": args.method_label, "receiver_prompt_mode": "cache", "kind": "cache", **item})

    summary = summarize(all_rows)
    write_jsonl(out / "per_example.jsonl", all_rows)
    write_jsonl(out / "per_layer.jsonl", layer_rows)
    write_csv(out / "diagnostic_table.csv", summary)
    payload = {
        "args": {key: value for key, value in vars(args).items() if key != "device_obj"},
        "adapter_metadata": adapter_metadata,
        "source_x": "context + question, no gold answer",
        "context_aware_receiver_input": "X + Answer: + answer_prefix",
        "context_unaware_receiver_input": "Answer: + answer_prefix",
        "diagnostic_table": summary,
        "answer_f1_definition": "F1 of teacher-forced per-position argmax answer tokens; not free-running generation",
    }
    with open(out / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", "method": args.method_label, "samples": len(rows)}, handle, indent=2)


if __name__ == "__main__":
    main()
