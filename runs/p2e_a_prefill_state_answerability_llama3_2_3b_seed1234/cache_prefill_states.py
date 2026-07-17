import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prefill_prompt import choose_fixed_summary_token, render_prompt


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def parse_dtype(name, device):
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def parse_layers(value, total_layers):
    if value == "auto":
        candidates = [round(total_layers * fraction) for fraction in (0.25, 0.5, 0.75, 1.0)]
    else:
        candidates = [int(item) for item in value.split(",") if item.strip()]
    layers = sorted(set(max(1, min(total_layers, layer)) for layer in candidates))
    if total_layers not in layers:
        layers.append(total_layers)
    return layers


def evidence_indices(offsets, spans):
    selected = []
    for index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            selected.append(index)
    if not selected:
        raise ValueError("No evidence tokens were selected")
    return selected


@torch.inference_mode()
def encode_condition(
    model,
    tokenizer,
    row,
    condition,
    summary_token_id,
    summary_slots,
    layers,
    raw_layer,
    device,
    max_length,
):
    prompt, spans = render_prompt(tokenizer, row, condition)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    prompt_length = int(encoded.input_ids.shape[1])
    total_length = prompt_length + summary_slots
    if total_length > max_length:
        raise ValueError(f"{row['id']} needs {total_length} tokens, max_length={max_length}")
    summary_ids = torch.full((1, summary_slots), summary_token_id, dtype=torch.long)
    input_ids = torch.cat((encoded.input_ids, summary_ids), dim=1).to(device)
    attention_mask = torch.ones_like(input_ids)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    layer_states = {}
    for layer in layers:
        hidden = output.hidden_states[layer][0]
        layer_states[str(layer)] = {
            "end": hidden[prompt_length - 1].detach().half().cpu(),
            "summary": hidden[prompt_length : prompt_length + summary_slots].detach().half().cpu(),
        }
    payload = {
        "prompt_tokens": prompt_length,
        "end": {layer: values["end"] for layer, values in layer_states.items()},
        "summary": {layer: values["summary"] for layer, values in layer_states.items()},
    }
    if condition == "correct":
        selected = evidence_indices(encoded.offset_mapping[0].tolist(), spans)
        raw_hidden = output.hidden_states[raw_layer][0, selected]
        payload["raw_evidence"] = {str(raw_layer): raw_hidden.detach().half().cpu()}
        payload["evidence_token_ids"] = encoded.input_ids[0, selected].tolist()
        payload["evidence_tokens"] = len(selected)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Cache frozen Llama prefill states for Experiment A")
    parser.add_argument(
        "--model",
        default="/home/yezhe/all_models/models/LLM-Research/Llama-3___2-3B-Instruct",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--conditions", default="correct")
    parser.add_argument("--summary-slots", type=int, default=16)
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--raw-layer", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    total_layers = int(model.config.num_hidden_layers)
    layers = parse_layers(args.layers, total_layers)
    raw_layer = args.raw_layer or total_layers
    if raw_layer < 1 or raw_layer > total_layers:
        raise ValueError(f"raw-layer must be in [1, {total_layers}]")
    conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    if "correct" not in conditions:
        raise ValueError("The correct condition is required")
    summary_token_id, summary_token_text = choose_fixed_summary_token(tokenizer)

    grouped = defaultdict(dict)
    for row in load_jsonl(args.data):
        grouped[row["pair_id"]][row["variant"]] = row
    pairs = [
        (pair_id, variants)
        for pair_id, variants in grouped.items()
        if {"base", "counterfactual"}.issubset(variants)
    ]
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    evidence_counts = []
    answer_vocabulary = set()
    for pair_index, (pair_id, variants) in enumerate(tqdm(pairs, desc="prefill_state_pairs")):
        examples = []
        for variant in ("base", "counterfactual"):
            row = variants[variant]
            states = {}
            for condition in conditions:
                states[condition] = encode_condition(
                    model,
                    tokenizer,
                    row,
                    condition,
                    summary_token_id,
                    args.summary_slots,
                    layers,
                    raw_layer,
                    device,
                    args.max_length,
                )
            evidence_counts.append(states["correct"]["evidence_tokens"])
            answer_vocabulary.add(row["answer"])
            examples.append(
                {
                    "pair_id": pair_id,
                    "id": row["id"],
                    "variant": variant,
                    "answer": row["answer"],
                    "counterpart_answer": row["counterpart_answer"],
                    "question": row["question"],
                    "states": states,
                }
            )
        filename = f"pair_{pair_index:05d}.pt"
        torch.save({"pair_id": pair_id, "examples": examples}, output / filename)
        entries.append(
            {
                "pair_id": pair_id,
                "file": filename,
                "base_answer": variants["base"]["answer"],
                "counterfactual_answer": variants["counterfactual"]["answer"],
            }
        )

    index = {
        "format_version": 1,
        "experiment": "prefill_state_answerability",
        "model": args.model,
        "data": args.data,
        "pairs": len(entries),
        "conditions": list(conditions),
        "layers": layers,
        "raw_layer": raw_layer,
        "hidden_size": int(model.config.hidden_size),
        "summary_slots": args.summary_slots,
        "summary_token_id": summary_token_id,
        "summary_token_text": summary_token_text,
        "prompt_style": "llama_fewshot_join_reason",
        "answer_vocabulary": sorted(answer_vocabulary),
        "min_evidence_tokens": min(evidence_counts),
        "max_evidence_tokens": max(evidence_counts),
        "pair_files": entries,
    }
    with open(output / "index.json", "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, ensure_ascii=False)
    with open(output / "CACHE_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", "pairs": len(entries)}, handle, indent=2)


if __name__ == "__main__":
    main()
