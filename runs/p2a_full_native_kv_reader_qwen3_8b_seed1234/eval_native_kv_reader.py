import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2a_common import (
    NativeKVExternalReader,
    extract_answer,
    full_text_prompt,
    iter_cache,
    memory_to,
    mismatched_memory,
    normalize_answer,
    parse_dtype,
    resolve_device,
    student_prompt,
    summarize_diagnostics,
    write_jsonl,
    zero_memory,
)


def load_pairs(index_path, max_pairs):
    grouped = defaultdict(dict)
    for example in iter_cache(index_path):
        grouped[example["pair_id"]][example["variant"]] = example
    pairs = [pair for pair in grouped.values() if {"base", "counterfactual"}.issubset(pair)]
    return pairs[:max_pairs] if max_pairs > 0 else pairs


@torch.inference_mode()
def generate(receiver, tokenizer, adapter, prompt, memory, max_new_tokens, device, enable_reader):
    current = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    past = None
    token_ids = []
    diagnostics = {}
    eos = tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos_reached = False
    for _ in range(max_new_tokens):
        if enable_reader:
            with adapter.inject(receiver, memory, diagnostics):
                output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        else:
            output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        token = int(output.logits[:, -1, :].argmax(dim=-1).item())
        token_ids.append(token)
        past = output.past_key_values
        if token in eos:
            eos_reached = True
            break
        current = torch.tensor([[token]], dtype=torch.long, device=device)
    return {
        "token_ids": token_ids,
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "eos_reached": eos_reached,
        "diagnostics": summarize_diagnostics(diagnostics),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_summary(records):
    output = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        output.append(
            {
                "condition": condition,
                "n": len(selected),
                "exact_match": float(np.mean([row["exact_match"] for row in selected])),
                "answer_found_rate": float(np.mean([row["answer_found"] for row in selected])),
                "eos_rate": float(np.mean([row["eos_reached"] for row in selected])),
                "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in selected])),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Free-running evaluation for the P2-A native-KV Reader")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    pairs = load_pairs(args.test_index, args.max_pairs)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    adapter = NativeKVExternalReader(
        receiver,
        max_gate=float(train_args["max_gate"]),
        gate_init=float(train_args["gate_init"]),
        reader_rank=int(train_args["reader_rank"]),
    ).to(device).eval()
    adapter.load_state_dict(checkpoint["adapter"])

    records = []
    layer_rows = []
    for pair_index, pair in enumerate(tqdm(pairs, desc="p2a_free_running")):
        base = pair["base"]
        counterfactual = pair["counterfactual"]
        other = pairs[(pair_index + 1) % len(pairs)]["base"]
        base_memory = memory_to(base["memory"], device, dtype)
        cf_memory = memory_to(counterfactual["memory"], device, dtype)
        other_memory = memory_to(other["memory"], device, dtype)
        conditions = [
            ("question_only", student_prompt(tokenizer, base), None, base["answer"], False),
            ("full_text_correct", full_text_prompt(tokenizer, base), None, base["answer"], False),
            (
                "full_text_counterfactual",
                full_text_prompt(tokenizer, counterfactual),
                None,
                counterfactual["answer"],
                False,
            ),
            ("correct_kv", student_prompt(tokenizer, base), base_memory, base["answer"], True),
            (
                "counterfactual_kv",
                student_prompt(tokenizer, base),
                cf_memory,
                counterfactual["answer"],
                True,
            ),
            ("shuffled_kv", student_prompt(tokenizer, base), other_memory, "INSUFFICIENT", True),
            (
                "mismatched_kv",
                student_prompt(tokenizer, base),
                mismatched_memory(base_memory, other_memory),
                "INSUFFICIENT",
                True,
            ),
            ("zero_kv", student_prompt(tokenizer, base), zero_memory(base_memory), "INSUFFICIENT", True),
            ("external_reader_off", student_prompt(tokenizer, base), base_memory, base["answer"], False),
        ]
        allowed = list(
            dict.fromkeys(
                [
                    *base["candidate_answers"],
                    *counterfactual["candidate_answers"],
                    *other["candidate_answers"],
                ]
            )
        )
        for condition, prompt, memory, target, enabled in conditions:
            result = generate(
                receiver,
                tokenizer,
                adapter,
                prompt,
                memory,
                args.max_new_tokens,
                device,
                enabled,
            )
            prediction, extraction_method = extract_answer(result["text"], allowed)
            exact = float(normalize_answer(prediction) == normalize_answer(target))
            records.append(
                {
                    "pair_id": base["pair_id"],
                    "condition": condition,
                    "target": target,
                    "base_answer": base["answer"],
                    "counterfactual_answer": counterfactual["answer"],
                    "prediction": prediction,
                    "generated_text": result["text"],
                    "generated_token_ids": result["token_ids"],
                    "generated_tokens": len(result["token_ids"]),
                    "eos_reached": result["eos_reached"],
                    "answer_found": bool(prediction),
                    "extraction_method": extraction_method,
                    "exact_match": exact,
                }
            )
            for layer in result["diagnostics"]:
                layer_rows.append({"pair_id": base["pair_id"], "condition": condition, **layer})

    conditions = condition_summary(records)
    by_name = {row["condition"]: row for row in conditions}
    correct = {row["pair_id"]: row for row in records if row["condition"] == "correct_kv"}
    counterfactual = {row["pair_id"]: row for row in records if row["condition"] == "counterfactual_kv"}
    consistency = float(
        np.mean(
            [
                correct[pair_id]["exact_match"] * counterfactual[pair_id]["exact_match"]
                for pair_id in correct.keys() & counterfactual.keys()
            ]
        )
    )
    summary = {
        "status": "complete",
        "args": vars(args),
        "conditions": conditions,
        "counterfactual_consistency": consistency,
        "correct_vs_shuffled_em_gap": by_name["correct_kv"]["exact_match"] - by_name["shuffled_kv"]["exact_match"],
        "correct_vs_reader_off_em_gap": by_name["correct_kv"]["exact_match"] - by_name["external_reader_off"]["exact_match"],
        "gates": adapter.gates().detach().float().cpu().tolist(),
    }

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_jsonl(output / "per_layer_reader_diagnostics.jsonl", layer_rows)
    write_csv(output / "condition_summary.csv", conditions)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
