import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from two_stage_common import (
    STAGE_CONDITIONS,
    assert_cache_shapes,
    evaluate_stage1_condition,
    readout_probe_rows,
    run_tail,
    sha256_file,
    summarize,
    swap_cache,
    write_json,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
sys.path.insert(0, str(REAL_ROOT))

from real_kv_common import (  # noqa: E402
    assert_tokenizer_compatible,
    build_example,
    extract_cache,
    load_rows,
    native_cache_equivalence,
)
from real_kv_translator import load_real_translator  # noqa: E402


def load_model(path, dtype, device, eager=False):
    kwargs = {"dtype": dtype, "trust_remote_code": True}
    if eager:
        kwargs["attn_implementation"] = "eager"
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()


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
    parser = argparse.ArgumentParser(description="Two-stage diagnosis for real heterogeneous KV translation")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--translator-checkpoint", required=True)
    parser.add_argument("--checkpoint-label", required=True)
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    checkpoint = Path(args.translator_checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

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

    sender = load_model(args.sender_model, dtype, device)
    receiver = load_model(args.receiver_model, dtype, device, eager=True)
    translator, translator_metadata = load_real_translator(checkpoint, map_location=device)
    translator = translator.to(device).eval()
    for module in (sender, receiver, translator):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_sha256": sha256_file(checkpoint),
        "translator_metadata": translator_metadata,
        "stage1_conditions": list(STAGE_CONDITIONS),
        "stage2_conditions": list(STAGE_CONDITIONS),
    }
    write_json(out / "checkpoint_manifest.json", manifest)

    stage1_rows = []
    stage1_layers = []
    stage2_rows = []
    equivalence_rows = []
    atol = args.equivalence_atol if args.equivalence_atol is not None else (0.5 if args.dtype == "float16" else 1e-3)

    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc=args.checkpoint_label)):
            example = build_example(receiver_tokenizer, row, args.max_context_tokens)
            context_ids = example["context_ids"].to(device)
            tail_ids = example["tail_ids"].to(device)
            query_len = example["query_ids"].shape[1]
            answer_len = example["answer_ids"].shape[1]

            equivalence = native_cache_equivalence(
                receiver,
                context_ids,
                tail_ids,
                query_len,
                answer_len,
                atol,
            )
            equivalence_rows.append({"sample": sample, "id": example["id"], **equivalence})
            if not equivalence["passed"]:
                raise RuntimeError(f"Native context-cache equivalence failed on sample {sample}: {equivalence}")

            sender_pairs = extract_cache(
                sender(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values
            )
            native_pairs = extract_cache(
                receiver(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values
            )
            translated_pairs = translator(sender_pairs)
            assert_cache_shapes(native_pairs, translated_pairs)

            native_run = run_tail(receiver, native_pairs, tail_ids, query_len, answer_len, capture_q=True)
            cached_runs = {"native": native_run}

            for condition in STAGE_CONDITIONS:
                condition_pairs = swap_cache(native_pairs, translated_pairs, condition)
                condition_run = cached_runs.get(condition)
                if condition_run is None:
                    condition_run = run_tail(receiver, condition_pairs, tail_ids, query_len, answer_len)
                    cached_runs[condition] = condition_run
                result, kv_rows, attention_rows = evaluate_stage1_condition(
                    condition,
                    receiver_tokenizer,
                    example,
                    native_pairs,
                    condition_pairs,
                    native_run,
                    condition_run,
                )
                result.update({"sample": sample, "id": example["id"], "checkpoint": args.checkpoint_label})
                stage1_rows.append(result)
                for kv_row in kv_rows:
                    stage1_layers.append(
                        {
                            "sample": sample,
                            "id": example["id"],
                            "checkpoint": args.checkpoint_label,
                            "condition": condition,
                            "stage": "stage1_cache",
                            **kv_row,
                        }
                    )
                for attention_row in attention_rows:
                    stage1_layers.append(
                        {
                            "sample": sample,
                            "id": example["id"],
                            "checkpoint": args.checkpoint_label,
                            "condition": condition,
                            "stage": "stage1_attention_output",
                            **attention_row,
                        }
                    )

            probe_rows = readout_probe_rows(
                native_run["query_states"],
                native_pairs,
                translated_pairs,
                receiver.config.num_attention_heads,
                args.attention_topk,
            )
            for probe_row in probe_rows:
                stage2_rows.append(
                    {
                        "sample": sample,
                        "id": example["id"],
                        "checkpoint": args.checkpoint_label,
                        **probe_row,
                    }
                )

    stage1_summary = summarize(
        stage1_rows,
        "condition",
        [
            "receiver_native_ce",
            "condition_ce",
            "ce_delta",
            "logit_kl",
            "top1_match",
            "answer_f1",
            "attention_output_cos",
            "kv_joint_consistency",
        ],
    )
    stage2_summary = summarize(
        stage2_rows,
        "condition",
        ["route_overlap", "attention_js", "attention_output_cos", "output_mse"],
    )

    write_jsonl(out / "stage1_per_example.jsonl", stage1_rows)
    write_jsonl(out / "stage1_per_layer.jsonl", stage1_layers)
    write_jsonl(out / "stage2_per_layer.jsonl", stage2_rows)
    write_csv(out / "stage1_cache_swap_summary.csv", stage1_summary)
    write_csv(out / "stage2_readout_probe_summary.csv", stage2_summary)
    write_json(
        out / "summary.json",
        {
            "args": vars(args),
            "checkpoint_manifest": manifest,
            "native_context_cache_equivalence": equivalence_rows,
            "stage1_cache_swap_summary": stage1_summary,
            "stage2_readout_probe_summary": stage2_summary,
            "answer_f1_definition": "F1 of teacher-forced per-position argmax tokens; not free-running generation",
            "stage2_definition": "Offline receiver-native-Q readout over context cache only; probe outputs are not fed back into the receiver.",
        },
    )
    write_json(
        out / "SUCCESS.json",
        {"status": "complete", "checkpoint": args.checkpoint_label, "samples": len(rows)},
    )


if __name__ == "__main__":
    main()
