import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p1_common import (
    CONDITIONS,
    OracleCacheDataset,
    OracleEvidenceAdapter,
    answer_f1,
    contains_answer,
    exact_match,
    extract_short_answer,
    greedy_generate,
    parse_dtype,
    resolve_device,
    summarize_generation,
    write_csv,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="Free-running evaluation of the P1 oracle reader")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--eval-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-style", choices=("chat", "plain"), default="chat")
    parser.add_argument("--p0-summary")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.out)
    success_path = output / "SUCCESS.json"
    if success_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {success_path}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    dataset = OracleCacheDataset(args.eval_cache)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    if dataset.raw_dim != int(checkpoint["raw_dim"]):
        raise ValueError("Evaluation cache hidden size does not match checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    adapter = OracleEvidenceAdapter(
        dataset.raw_dim,
        receiver.config.hidden_size,
        int(train_args["shared_dim"]),
        int(train_args["reader_heads"]),
        train_args["reader_layers"],
    ).to(device).eval()
    adapter.load_state_dict(checkpoint["adapter"])

    count = min(args.max_samples, len(dataset)) if args.max_samples > 0 else len(dataset)
    examples = [dataset[index] for index in range(count)]
    records = []
    with torch.inference_mode():
        for sample_index, example in enumerate(tqdm(examples, desc="p1_free_running")):
            mismatch = examples[(sample_index + 1) % len(examples)] if len(examples) > 1 else example
            for condition in args.conditions:
                mismatch_example = mismatch if condition == "mismatched_a_plus_b" else None
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                generated_ids, generated_text, diagnostics = greedy_generate(
                    receiver,
                    tokenizer,
                    adapter,
                    example,
                    condition,
                    device,
                    args.max_new_tokens,
                    args.prompt_style,
                    mismatch_example=mismatch_example,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                prediction = extract_short_answer(generated_text)
                gold = example["answer"]
                gate_means = [item["gate_mean"] for item in diagnostics.values()]
                records.append(
                    {
                        "sample": sample_index,
                        "id": example["id"],
                        "type": example["type"],
                        "condition": condition,
                        "question": example["question"],
                        "gold_answer": gold,
                        "prediction": prediction,
                        "generated_text": generated_text,
                        "generated_token_ids": generated_ids,
                        "exact_match": exact_match(prediction, gold),
                        "answer_f1": answer_f1(prediction, gold),
                        "contains_gold": contains_answer(generated_text, gold),
                        "generated_tokens": len(generated_ids),
                        "latency_seconds": time.perf_counter() - started,
                        "gate_mean": float(np.mean(gate_means)) if gate_means else 0.0,
                    }
                )

    p0_summary = None
    if args.p0_summary:
        with open(args.p0_summary, encoding="utf-8") as handle:
            p0_summary = json.load(handle)
    condition_summary, paired_summary = summarize_generation(
        records,
        examples,
        args.conditions,
        p0_summary,
    )
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_csv(output / "condition_summary.csv", condition_summary)
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"conditions": condition_summary, "paired": paired_summary},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with open(success_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "args": vars(args),
                "n": count,
                "conditions": condition_summary,
                "paired": paired_summary,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
