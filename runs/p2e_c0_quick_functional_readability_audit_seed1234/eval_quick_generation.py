import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_common import LazyPairCache, load_manifest, verify_manifest_cache, write_jsonl
from cache_reader_chain import build_reader
from p2a_common import (
    extract_answer,
    full_text_prefixed_prompt,
    memory_to,
    mismatched_memory,
    normalize_answer,
    parse_dtype,
    resolve_device,
    student_prefixed_prompt,
    zero_memory,
)
from train_memory_probes import device_memory, load_writer


def negative_pair(cache, manifest_rows, position):
    target = manifest_rows[position]
    answers = {target["base_answer"], target["counterfactual_answer"]}
    for offset in range(1, len(manifest_rows)):
        candidate = manifest_rows[(position + offset) % len(manifest_rows)]
        if answers.isdisjoint(
            {candidate["base_answer"], candidate["counterfactual_answer"]}
        ):
            return cache.load(int(candidate["index"]))
    raise RuntimeError("No disjoint negative")


@torch.inference_mode()
def generate(receiver, tokenizer, reader, prompt, memory, max_new_tokens, device, enabled):
    current = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    past = None
    tokens = []
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_reached = False
    for _ in range(max_new_tokens):
        if enabled:
            with reader.inject(receiver, memory):
                output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        else:
            output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        token = int(output.logits[:, -1].argmax(dim=-1).item())
        tokens.append(token)
        past = output.past_key_values
        if token in eos_ids:
            eos_reached = True
            break
        current = torch.tensor([[token]], dtype=torch.long, device=device)
    return tokenizer.decode(tokens, skip_special_tokens=True), tokens, eos_reached


def paired(records, base_condition, cf_condition):
    base = {row["pair_id"]: row for row in records if row["condition"] == base_condition}
    cf = {row["pair_id"]: row for row in records if row["condition"] == cf_condition}
    keys = sorted(base.keys() & cf.keys())
    return float(np.mean([base[key]["correct"] * cf[key]["correct"] for key in keys]))


def main():
    parser = argparse.ArgumentParser(description="P2-E-C0 quick free-running evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-test-index", required=True)
    parser.add_argument("--native-test-index", required=True)
    parser.add_argument("--writer-checkpoint", required=True)
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    manifest = load_manifest(args.manifest)
    raw_cache = LazyPairCache(args.raw_test_index)
    native_cache = LazyPairCache(args.native_test_index)
    verify_manifest_cache(manifest, "test", raw_cache)
    verify_manifest_cache(manifest, "test", native_cache)
    labels = sorted(
        {
            answer
            for entry in raw_cache.entries
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
    reader = build_reader(receiver, args.reader_checkpoint, device)
    writer = load_writer(args.writer_checkpoint, device)
    records = []
    rows = manifest["test"]
    for position, record in enumerate(tqdm(rows, desc="quick_free_running")):
        pair = raw_cache.load(int(record["index"]))
        native_pair = native_cache.load(int(record["index"]))
        unrelated = negative_pair(raw_cache, rows, position)
        writer_memory = {}
        unrelated_memory = {}
        for variant in ("base", "counterfactual"):
            with torch.inference_mode():
                writer_memory[variant] = writer(
                    memory_to(pair[variant]["memory"], device, dtype), output_dtype=dtype
                )
                unrelated_memory[variant] = writer(
                    memory_to(unrelated[variant]["memory"], device, dtype), output_dtype=dtype
                )
        base = pair["base"]
        cf = pair["counterfactual"]
        conditions = [
            ("full_text_base", full_text_prefixed_prompt(tokenizer, base), None, base["answer"], base["answer"], False),
            ("full_text_counterfactual", full_text_prefixed_prompt(tokenizer, cf), None, cf["answer"], cf["answer"], False),
            ("native_base", student_prefixed_prompt(tokenizer, base), memory_to(native_pair["base"]["memory"], device, dtype), base["answer"], base["answer"], True),
            ("native_counterfactual", student_prefixed_prompt(tokenizer, base), memory_to(native_pair["counterfactual"]["memory"], device, dtype), cf["answer"], cf["answer"], True),
            ("writer_base", student_prefixed_prompt(tokenizer, base), writer_memory["base"], base["answer"], base["answer"], True),
            ("writer_counterfactual", student_prefixed_prompt(tokenizer, base), writer_memory["counterfactual"], cf["answer"], cf["answer"], True),
            ("writer_shuffled", student_prefixed_prompt(tokenizer, base), unrelated_memory["base"], base["answer"], unrelated["base"]["answer"], True),
            ("writer_mismatched", student_prefixed_prompt(tokenizer, base), mismatched_memory(writer_memory["base"], unrelated_memory["base"]), base["answer"], unrelated["base"]["answer"], True),
            ("writer_zero", student_prefixed_prompt(tokenizer, base), zero_memory(writer_memory["base"]), base["answer"], None, True),
            ("reader_off", student_prefixed_prompt(tokenizer, base), None, base["answer"], None, False),
        ]
        for condition, prompt, memory, target, memory_answer, enabled in conditions:
            text, tokens, eos = generate(
                receiver, tokenizer, reader, prompt, memory,
                args.max_new_tokens, device, enabled,
            )
            prediction, method = extract_answer(text, labels)
            records.append(
                {
                    "pair_id": base["pair_id"],
                    "condition": condition,
                    "target": target,
                    "memory_answer": memory_answer,
                    "prediction": prediction,
                    "generated_text": text,
                    "generated_token_ids": tokens,
                    "eos_reached": eos,
                    "extraction_method": method,
                    "correct": float(normalize_answer(prediction) == normalize_answer(target)),
                    "memory_answer_hit": float(
                        memory_answer is not None
                        and normalize_answer(prediction) == normalize_answer(memory_answer)
                    ),
                }
            )
    conditions = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        conditions.append(
            {
                "condition": condition,
                "accuracy": float(np.mean([row["correct"] for row in selected])),
                "memory_answer_hit_rate": float(
                    np.mean([row["memory_answer_hit"] for row in selected])
                ),
                "eos_rate": float(np.mean([row["eos_reached"] for row in selected])),
            }
        )
    result = {
        "status": "complete",
        "conditions": conditions,
        "paired_consistency": {
            "full_text": paired(records, "full_text_base", "full_text_counterfactual"),
            "native": paired(records, "native_base", "native_counterfactual"),
            "writer": paired(records, "writer_base", "writer_counterfactual"),
        },
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
