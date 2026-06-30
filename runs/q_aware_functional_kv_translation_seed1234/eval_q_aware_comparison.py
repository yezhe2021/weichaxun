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

from q_aware_common import (
    answer_position_slice,
    cosine_mean,
    offline_readout,
    route_js_value,
    tail_logits,
    topk_overlap,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
for path in (PROJECT_ROOT, REAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_kv_common import (  # noqa: E402
    assert_tokenizer_compatible,
    build_example,
    cache_metrics,
    distribution_metrics,
    extract_cache,
    load_rows,
    mean_metric,
    native_cache_equivalence,
)
from real_kv_translator import load_real_translator  # noqa: E402
from translated_kv_diagnostics import answer_f1  # noqa: E402


def load_model(path, dtype, device, eager=False):
    kwargs = {"dtype": dtype, "trust_remote_code": True}
    if eager:
        kwargs["attn_implementation"] = "eager"
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()


def teacher_forced_prediction(tokenizer, logits):
    ids = logits.argmax(dim=-1)[0]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def attention_output_rows(native_outputs, translated_outputs, query_len, answer_len):
    rows = []
    for layer in sorted(native_outputs):
        native = native_outputs[layer].detach().float().cpu()
        translated = translated_outputs[layer].detach().float().cpu()
        seq_len = min(native.shape[1], translated.shape[1])
        sl = answer_position_slice(query_len, answer_len, seq_len)
        if sl.stop <= sl.start:
            continue
        rows.append(
            {
                "layer": layer,
                "attention_output_cos": cosine_mean(native[:, sl], translated[:, sl]),
            }
        )
    return rows


def readout_rows(query_states, native_pairs, translated_pairs, num_attention_heads, query_len, answer_len, topk):
    native_routes, native_outputs = offline_readout(
        query_states,
        native_pairs,
        num_attention_heads,
        query_len,
        answer_len,
    )
    translated_routes, translated_outputs = offline_readout(
        query_states,
        translated_pairs,
        num_attention_heads,
        query_len,
        answer_len,
    )
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
                "attention_js": route_js_value(native_route, translated_route),
                "readout_output_cos": output_cos,
                "readout_output_mse": output_mse,
                "readout_loss": output_mse + (1.0 - output_cos),
            }
        )
    return rows


def summarize(rows):
    keys = [
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
        "kv_joint_consistency",
        "kv_mse",
        "k_cos",
        "v_cos",
    ]
    output = []
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        item = {"method": method, "n": len(selected)}
        for key in keys:
            values = [row[key] for row in selected if key in row and np.isfinite(row[key])]
            if values:
                item[key] = float(np.mean(values))
        output.append(item)
    return output


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate q-aware functional KV translation comparison")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--translator-checkpoint", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--out", required=True)
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
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    rows = load_rows(args.data, args.max_samples)
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(
        sender_tokenizer,
        receiver_tokenizer,
        rows,
        args.max_context_tokens,
        args.tokenizer_check_samples,
    )
    tokenizer = receiver_tokenizer
    sender = load_model(args.sender_model, dtype, device)
    receiver = load_model(args.receiver_model, dtype, device, eager=True)
    translator, translator_metadata = load_real_translator(args.translator_checkpoint, map_location=device)
    translator = translator.to(device).eval()
    for module in (sender, receiver, translator):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    layer_rows = []
    equivalence_rows = []
    atol = args.equivalence_atol if args.equivalence_atol is not None else (0.5 if args.dtype == "float16" else 1e-3)

    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc=args.method_label)):
            example = build_example(tokenizer, row, args.max_context_tokens)
            context_ids = example["context_ids"].to(device)
            tail_ids = example["tail_ids"].to(device)
            answer_ids = example["answer_ids"].to(device)
            query_len = example["query_ids"].shape[1]
            answer_len = answer_ids.shape[1]

            eq = native_cache_equivalence(receiver, context_ids, tail_ids, query_len, answer_len, atol)
            equivalence_rows.append({"sample": sample, "id": example["id"], **eq})

            sender_pairs = extract_cache(
                sender(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values
            )
            native_pairs = extract_cache(
                receiver(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values
            )
            translated_pairs = translator(sender_pairs)

            native_logits, query_states, native_attention_outputs = tail_logits(
                receiver,
                native_pairs,
                tail_ids,
                query_len,
                answer_len,
                capture_q=True,
            )
            translated_logits, _, translated_attention_outputs = tail_logits(
                receiver,
                translated_pairs,
                tail_ids,
                query_len,
                answer_len,
                capture_q=True,
            )

            distribution = distribution_metrics(
                native_logits.detach().float().cpu(),
                translated_logits.detach().float().cpu(),
                answer_ids.detach().cpu(),
            )
            kv_rows = cache_metrics(
                [(k.detach().float().cpu(), v.detach().float().cpu()) for k, v in native_pairs],
                [(k.detach().float().cpu(), v.detach().float().cpu()) for k, v in translated_pairs],
            )
            attn_rows = attention_output_rows(native_attention_outputs, translated_attention_outputs, query_len, answer_len)
            probe_rows = readout_rows(
                query_states,
                native_pairs,
                translated_pairs,
                receiver.config.num_attention_heads,
                query_len,
                answer_len,
                args.attention_topk,
            )
            prediction = teacher_forced_prediction(tokenizer, translated_logits.detach().float().cpu())
            result = {
                "sample": sample,
                "id": example["id"],
                "method": args.method_label,
                "receiver_native_ce": distribution["native_ce"],
                "translated_ce": distribution["translated_ce"],
                "ce_delta": distribution["ce_delta"],
                "logit_kl": distribution["logit_kl"],
                "top1_match": distribution["top1_match"],
                "answer_prediction": prediction,
                "answer_prediction_mode": "teacher_forced_token_argmax",
                "answer_f1": answer_f1(prediction, example["answer"]),
                "attention_output_cos": float(np.mean([r["attention_output_cos"] for r in attn_rows])),
                "route_overlap": mean_metric(probe_rows, "route_overlap"),
                "attention_js": mean_metric(probe_rows, "attention_js"),
                "readout_output_cos": mean_metric(probe_rows, "readout_output_cos"),
                "readout_output_mse": mean_metric(probe_rows, "readout_output_mse"),
                "readout_loss": mean_metric(probe_rows, "readout_loss"),
                "kv_joint_consistency": mean_metric(kv_rows, "kv_joint_consistency"),
                "kv_mse": mean_metric(kv_rows, "kv_mse"),
                "k_cos": mean_metric(kv_rows, "k_cos"),
                "v_cos": mean_metric(kv_rows, "v_cos"),
            }
            all_rows.append(result)
            for kv_row in kv_rows:
                layer_rows.append(
                    {"sample": sample, "id": example["id"], "method": args.method_label, "kind": "cache", **kv_row}
                )
            for attn_row in attn_rows:
                layer_rows.append(
                    {
                        "sample": sample,
                        "id": example["id"],
                        "method": args.method_label,
                        "kind": "attention_output",
                        **attn_row,
                    }
                )
            for probe_row in probe_rows:
                layer_rows.append(
                    {"sample": sample, "id": example["id"], "method": args.method_label, "kind": "readout_probe", **probe_row}
                )

    summary = summarize(all_rows)
    write_jsonl(out / "per_example.jsonl", all_rows)
    write_jsonl(out / "per_layer.jsonl", layer_rows)
    write_csv(out / "diagnostic_table.csv", summary)
    payload = {
        "args": vars(args),
        "translator_metadata": translator_metadata,
        "native_context_cache_equivalence": equivalence_rows,
        "diagnostic_table": summary,
        "readout_definition": "Receiver-native Q reads native and translated context KV offline; readout outputs are not fed back into the receiver.",
        "answer_f1_definition": "F1 of teacher-forced per-position argmax tokens; not free-running generation.",
    }
    with open(out / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", "method": args.method_label, "samples": len(rows)}, handle, indent=2)


if __name__ == "__main__":
    main()
