import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_specific_reader import LlamaSpecificExternalReader
from p2a_common import (
    NativeKVExternalReader,
    extract_answer,
    full_text_prefixed_prompt,
    memory_to,
    mismatched_memory,
    normalize_answer,
    parse_dtype,
    resolve_device,
    student_prefixed_prompt,
    summarize_diagnostics,
    write_jsonl,
    zero_memory,
)
from train_llama_reader import LazyPairCache, compatible_negative


def fixed_negative(cache, index):
    for offset in range(1, len(cache)):
        candidate = (index + offset) % len(cache)
        if compatible_negative(cache, index, candidate):
            return candidate
    raise RuntimeError(f"No answer-disjoint negative for pair {index}")


def build_native_reader(receiver, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    reader = NativeKVExternalReader(
        receiver,
        max_gate=float(args["max_gate"]),
        gate_init=float(args["gate_init"]),
        query_rank=int(args["query_rank"]),
        output_rank=int(args["output_rank"]),
    ).to(device).eval()
    reader.load_state_dict(checkpoint["adapter"])
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    return reader


@torch.inference_mode()
def generate(receiver, tokenizer, reader, prompt, memory, max_new_tokens, device, enabled):
    current = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    past = None
    token_ids = []
    diagnostics = {}
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_reached = False
    for _ in range(max_new_tokens):
        if enabled:
            with reader.inject(receiver, memory, diagnostics):
                output = receiver(
                    input_ids=current, past_key_values=past, use_cache=True, return_dict=True
                )
        else:
            output = receiver(
                input_ids=current, past_key_values=past, use_cache=True, return_dict=True
            )
        token = int(output.logits[:, -1, :].argmax(dim=-1).item())
        token_ids.append(token)
        past = output.past_key_values
        if token in eos_ids:
            eos_reached = True
            break
        current = torch.tensor([[token]], dtype=torch.long, device=device)
    return {
        "token_ids": token_ids,
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "eos_reached": eos_reached,
        "diagnostics": summarize_diagnostics(diagnostics),
    }


def condition_summary(records):
    rows = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "n": len(selected),
                "target_em": float(np.mean([row["target_em"] for row in selected])),
                "memory_answer_hit_rate": (
                    float(np.mean([row["memory_answer_hit"] for row in selected]))
                    if any(row["memory_answer"] is not None for row in selected)
                    else None
                ),
                "eos_rate": float(np.mean([row["eos_reached"] for row in selected])),
                "mean_generated_tokens": float(
                    np.mean([row["generated_tokens"] for row in selected])
                ),
            }
        )
    return rows


def paired_consistency(records, base_condition, cf_condition):
    base = {row["pair_id"]: row for row in records if row["condition"] == base_condition}
    cf = {row["pair_id"]: row for row in records if row["condition"] == cf_condition}
    common = sorted(base.keys() & cf.keys())
    return float(np.mean([base[key]["target_em"] * cf[key]["target_em"] for key in common]))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Free-running evaluation of a Llama-specific Reader")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--native-reader-checkpoint", required=True)
    parser.add_argument("--sender-index", required=True)
    parser.add_argument("--native-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    sender_cache = LazyPairCache(args.sender_index)
    native_cache = LazyPairCache(args.native_index)
    pair_count = min(len(sender_cache), args.max_pairs) if args.max_pairs > 0 else len(sender_cache)
    for index in range(pair_count):
        if sender_cache.entries[index]["pair_id"] != native_cache.entries[index]["pair_id"]:
            raise ValueError("Llama and Native-Qwen caches are not pair aligned")
    allowed_answers = sorted(
        {
            answer
            for entry in sender_cache.entries
            for answer in (entry["base_answer"], entry["counterfactual_answer"])
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.receiver_model, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    native_reader = build_native_reader(receiver, args.native_reader_checkpoint, device)

    checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    geometry = checkpoint["sender_geometry"]
    reader = LlamaSpecificExternalReader(
        receiver,
        sender_layers=geometry["layers"],
        sender_kv_heads=geometry["kv_heads"],
        sender_head_dim=geometry["head_dim"],
        variant=train_args["variant"],
        top_k=int(train_args["top_k"]),
        query_rank=int(train_args["query_rank"]),
        output_rank=int(train_args["output_rank"]),
        max_gate=float(train_args["max_gate"]),
        gate_init=float(train_args["gate_init"]),
    ).to(device).eval()
    reader.load_state_dict(checkpoint["reader"])
    for parameter in reader.parameters():
        parameter.requires_grad_(False)

    records = []
    layer_rows = []
    for pair_index in tqdm(range(pair_count), desc=f"eval_{train_args['variant']}"):
        pair = sender_cache.load(pair_index)
        native_pair = native_cache.load(pair_index)
        unrelated_pair = sender_cache.load(fixed_negative(sender_cache, pair_index))
        base = pair["base"]
        cf = pair["counterfactual"]
        llama_base = memory_to(base["memory"], device, dtype)
        llama_cf = memory_to(cf["memory"], device, dtype)
        llama_other = memory_to(unrelated_pair["base"]["memory"], device, dtype)
        native_base = memory_to(native_pair["base"]["memory"], device, dtype)
        native_cf = memory_to(native_pair["counterfactual"]["memory"], device, dtype)

        conditions = [
            ("full_text_base", None, full_text_prefixed_prompt(tokenizer, base), None, base["answer"], base["answer"], False),
            ("full_text_counterfactual", None, full_text_prefixed_prompt(tokenizer, cf), None, cf["answer"], cf["answer"], False),
            ("native_8b_base", native_reader, student_prefixed_prompt(tokenizer, base), native_base, base["answer"], base["answer"], True),
            ("native_8b_counterfactual", native_reader, student_prefixed_prompt(tokenizer, base), native_cf, cf["answer"], cf["answer"], True),
            ("llama_reader_base", reader, student_prefixed_prompt(tokenizer, base), llama_base, base["answer"], base["answer"], True),
            ("llama_reader_counterfactual", reader, student_prefixed_prompt(tokenizer, base), llama_cf, cf["answer"], cf["answer"], True),
            ("llama_reader_shuffled", reader, student_prefixed_prompt(tokenizer, base), llama_other, base["answer"], unrelated_pair["base"]["answer"], True),
            ("llama_reader_mismatched", reader, student_prefixed_prompt(tokenizer, base), mismatched_memory(llama_base, llama_other), base["answer"], unrelated_pair["base"]["answer"], True),
            ("llama_reader_zero", reader, student_prefixed_prompt(tokenizer, base), zero_memory(llama_base), base["answer"], None, True),
            ("llama_reader_off", None, student_prefixed_prompt(tokenizer, base), None, base["answer"], None, False),
        ]
        for condition, active_reader, prompt, memory, target, memory_answer, enabled in conditions:
            result = generate(
                receiver, tokenizer, active_reader, prompt, memory,
                args.max_new_tokens, device, enabled,
            )
            prediction, method = extract_answer(result["text"], allowed_answers)
            records.append(
                {
                    "pair_id": base["pair_id"],
                    "condition": condition,
                    "target": target,
                    "memory_answer": memory_answer,
                    "prediction": prediction,
                    "generated_text": result["text"],
                    "generated_token_ids": result["token_ids"],
                    "generated_tokens": len(result["token_ids"]),
                    "eos_reached": result["eos_reached"],
                    "extraction_method": method,
                    "target_em": float(normalize_answer(prediction) == normalize_answer(target)),
                    "memory_answer_hit": float(
                        memory_answer is not None
                        and normalize_answer(prediction) == normalize_answer(memory_answer)
                    ),
                }
            )
            for row in result["diagnostics"]:
                layer_rows.append({"pair_id": base["pair_id"], "condition": condition, **row})

    conditions = condition_summary(records)
    summary = {
        "status": "complete",
        "variant": train_args["variant"],
        "args": vars(args),
        "conditions": conditions,
        "paired_consistency": {
            "full_text": paired_consistency(records, "full_text_base", "full_text_counterfactual"),
            "native_8b": paired_consistency(records, "native_8b_base", "native_8b_counterfactual"),
            "llama_specific_reader": paired_consistency(
                records, "llama_reader_base", "llama_reader_counterfactual"
            ),
        },
        "reader_diagnostics": {
            "mean_target_attention_mass": float(
                np.mean(
                    [
                        row["target_attention_mass"]
                        for row in layer_rows
                        if row["condition"] in {"llama_reader_base", "llama_reader_counterfactual"}
                    ]
                )
            ),
            "mean_abs_gate": float(reader.gates().detach().abs().mean().cpu()),
        },
        "trainable_component": "Llama-specific Reader only; Qwen backbone frozen",
        "memory_object": "evidence-token-only raw Llama pre-RoPE K and native V",
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_jsonl(output / "per_layer_diagnostics.jsonl", layer_rows)
    write_jsonl(output / "global_routing.jsonl", reader.routing_diagnostics())
    write_csv(output / "condition_summary.csv", conditions)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
