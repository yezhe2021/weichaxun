import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from causal_common import load_jsonl, parse_dtype, render_prompt, resolve_device, token_span_masks


def encode_question(model, tokenizer, row, device):
    prompt = render_prompt(tokenizer, row, "question_only", "chat")
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    return output.hidden_states[-1][:, -1, :].float().cpu()[0]


def encode_evidence(model, tokenizer, question, evidence, entities, layer, max_length, device):
    prefix = f"Question:\n{question}\n\nEvidence:\n"
    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    evidence_ids, masks = token_span_masks(evidence, entities, tokenizer)
    budget = max_length - len(prefix_ids)
    if len(evidence_ids) > budget:
        raise ValueError(f"Evidence requires {len(evidence_ids)} tokens but budget is {budget}")
    input_ids = torch.tensor([prefix_ids + evidence_ids], dtype=torch.long, device=device)
    output = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
    if layer < 0 or layer + 1 >= len(output.hidden_states):
        raise ValueError(f"Memory layer {layer} is invalid for {len(output.hidden_states) - 1} layers")
    hidden = output.hidden_states[layer + 1][:, len(prefix_ids) :, :].float().cpu()[0]
    return hidden.to(torch.float16), masks


def save_shard(output, shard_index, examples):
    name = f"shard_{shard_index:05d}.pt"
    torch.save({"examples": examples}, output / name)
    return name


def main():
    parser = argparse.ArgumentParser(description="C1 cache full receiver-native evidence-token memories")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--memory-layer", type=int, default=18)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    rows = load_jsonl(args.data, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    shards = []
    current = []
    with torch.inference_mode():
        for row in tqdm(rows, desc="native_memory"):
            question_state = encode_question(model, tokenizer, row, device)
            memory_a, bridge_a_mask = encode_evidence(
                model,
                tokenizer,
                row["question"],
                row["evidence_a"],
                [row["bridge"]],
                args.memory_layer,
                args.max_length,
                device,
            )
            memory_b, bridge_masks = encode_evidence(
                model,
                tokenizer,
                row["question"],
                row["evidence_b"],
                row["candidate_bridges"],
                args.memory_layer,
                args.max_length,
                device,
            )
            _, answer_masks = token_span_masks(row["evidence_b"], row["candidate_answers"], tokenizer)
            current.append(
                {
                    **row,
                    "question_state": question_state.to(torch.float16),
                    "memory_a": memory_a,
                    "memory_b": memory_b,
                    "a_bridge_mask": bridge_a_mask,
                    "b_bridge_masks": bridge_masks,
                    "b_answer_masks": answer_masks,
                }
            )
            if len(current) >= args.shard_size:
                shards.append(save_shard(output, len(shards), current))
                current = []
    if current:
        shards.append(save_shard(output, len(shards), current))
    with open(output / "index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "format_version": 1,
                "model": args.model,
                "memory_layer": args.memory_layer,
                "hidden_size": int(model.config.hidden_size),
                "data": args.data,
                "n": len(rows),
                "shards": shards,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
