import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2a_common import get_layers, load_jsonl, parse_dtype, resolve_device, sender_text


SYSTEM = (
    "Answer by joining the exact person in Evidence A to an employer, then joining that exact employer "
    "in Evidence B to a city. Ignore distractors. All required facts are present. Finish with FINAL: <city>."
)
EXAMPLE = (
    "EXAMPLE\n"
    "QUESTION\nIn which city is the employer of Mina Cole located?\n\n"
    "EVIDENCE A\nTheo Park works for Cedar Labs. Mina Cole works for Aurora Systems.\n\n"
    "EVIDENCE B\nCedar Labs is located in Rome. Aurora Systems is located in Oslo.\n\n"
    "REASONING\nMina Cole -> Aurora Systems -> Oslo\nFINAL: Oslo\n\n"
    "NOW SOLVE\n"
)


class TeacherCache:
    def __init__(self, index_path):
        with open(index_path, encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.root = Path(index_path).parent
        self.entries = self.index["pair_files"]

    def load(self, index):
        payload = torch.load(
            self.root / self.entries[index]["file"], map_location="cpu", weights_only=False
        )
        return {example["variant"]: example for example in payload["examples"]}


def evidence_block(row):
    return (
        f"QUESTION\n{row['question']}\n\n"
        f"EVIDENCE A\n{row['evidence_a']}\n\n"
        f"EVIDENCE B\n{row['evidence_b']}"
    )


def llama_sender_prompt(tokenizer, row):
    user = EXAMPLE + evidence_block(row)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{SYSTEM}\n\n{user}\n\n"
    marker = prompt.rfind("NOW SOLVE")
    if marker < 0:
        raise ValueError("The calibrated Llama prompt lost its NOW SOLVE marker")
    a_start = prompt.find(row["evidence_a"], marker)
    b_start = prompt.find(row["evidence_b"], a_start + len(row["evidence_a"]))
    if a_start < 0 or b_start < 0:
        raise ValueError(f"Could not locate evidence spans for {row['id']}")
    return prompt, (
        (a_start, a_start + len(row["evidence_a"])),
        (b_start, b_start + len(row["evidence_b"])),
    )


def selected_indices(offsets, prompt_spans):
    indices = []
    for index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in prompt_spans):
            indices.append(index)
    if not indices:
        raise ValueError("No evidence tokens were selected")
    return indices


def canonical_token_spans(offsets, indices, prompt_spans, row):
    a_length = len(row["evidence_a"])
    canonical_b_start = a_length + 1
    output = []
    for token_index in indices:
        start, end = offsets[token_index]
        pieces = []
        for span_index, (span_start, span_end) in enumerate(prompt_spans):
            overlap_start = max(start, span_start)
            overlap_end = min(end, span_end)
            if overlap_start < overlap_end:
                base = 0 if span_index == 0 else canonical_b_start
                pieces.append((base + overlap_start - span_start, base + overlap_end - span_start))
        if not pieces:
            raise ValueError("Selected token has no canonical evidence overlap")
        output.append((min(piece[0] for piece in pieces), max(piece[1] for piece in pieces)))
    return output


def named_character_spans(row):
    a = row["evidence_a"]
    b = row["evidence_b"]
    b_offset = len(a) + 1

    def locate(text, value, offset=0):
        start = text.find(value)
        if start < 0:
            raise ValueError(f"Could not locate {value!r} in evidence")
        return (offset + start, offset + start + len(value))

    return OrderedDict(
        evidence_a=(0, len(a)),
        evidence_b=(b_offset, b_offset + len(b)),
        target_person=locate(a, row["target_person"]),
        organization_a=locate(a, row["target_organization"]),
        organization_b=locate(b, row["target_organization"], b_offset),
        answer=locate(b, row["answer"], b_offset),
    )


def span_masks(token_spans, named_spans):
    return {
        name: torch.tensor(
            [start < span_end and end > span_start for start, end in token_spans],
            dtype=torch.bool,
        )
        for name, (span_start, span_end) in named_spans.items()
    }


def overlap_transport(sender_spans, teacher_spans):
    overlap = torch.zeros(len(teacher_spans), len(sender_spans), dtype=torch.float32)
    for teacher_index, (teacher_start, teacher_end) in enumerate(teacher_spans):
        for sender_index, (sender_start, sender_end) in enumerate(sender_spans):
            overlap[teacher_index, sender_index] = max(
                0, min(teacher_end, sender_end) - max(teacher_start, sender_start)
            )
    if (overlap.sum(dim=1) == 0).any() or (overlap.sum(dim=0) == 0).any():
        raise ValueError("Cross-tokenizer transport contains an unmapped evidence token")
    teacher_pool = overlap / overlap.sum(dim=1, keepdim=True)
    sender_mass = overlap.transpose(0, 1)
    sender_mass = sender_mass / sender_mass.sum(dim=1, keepdim=True)
    return teacher_pool.half(), sender_mass.half()


def qwen_teacher_metadata(tokenizer, row, teacher_example):
    text, prompt_spans = sender_text(row)
    encoded = tokenizer(
        text, return_offsets_mapping=True, add_special_tokens=True, truncation=False
    )
    offsets = encoded["offset_mapping"]
    indices = selected_indices(offsets, prompt_spans)
    token_ids = [encoded["input_ids"][index] for index in indices]
    if token_ids != teacher_example.get("evidence_token_ids"):
        raise RuntimeError(f"Reconstructed Qwen tokens do not match teacher cache for {row['id']}")
    return canonical_token_spans(offsets, indices, prompt_spans, row)


def project_native_kv(attention, hidden):
    head_dim = int(getattr(attention, "head_dim"))
    key = attention.k_proj(hidden).view(*hidden.shape[:-1], -1, head_dim)
    if hasattr(attention, "k_norm"):
        key = attention.k_norm(key)
    value = attention.v_proj(hidden).view(*hidden.shape[:-1], -1, head_dim)
    return key.transpose(1, 2), value.transpose(1, 2)


@torch.inference_mode()
def encode_example(model, tokenizer, qwen_tokenizer, row, teacher_example, device, max_length):
    prompt, prompt_spans = llama_sender_prompt(tokenizer, row)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    if encoded.input_ids.shape[1] > max_length:
        raise ValueError(
            f"{row['id']} requires {encoded.input_ids.shape[1]} tokens, max_length={max_length}"
        )
    offsets = encoded.offset_mapping[0].tolist()
    indices = selected_indices(offsets, prompt_spans)
    sender_spans = canonical_token_spans(offsets, indices, prompt_spans, row)
    teacher_spans = qwen_teacher_metadata(qwen_tokenizer, row, teacher_example)
    named_spans = named_character_spans(row)
    teacher_pool, sender_mass = overlap_transport(sender_spans, teacher_spans)
    sender_masks = span_masks(sender_spans, named_spans)
    teacher_masks = span_masks(teacher_spans, named_spans)
    answer_mask = sender_masks["answer"]
    if not answer_mask.any() or not teacher_masks["answer"].any():
        raise RuntimeError(f"Answer span is absent after tokenization for {row['id']}")

    model_inputs = {
        "input_ids": encoded.input_ids.to(device),
        "attention_mask": encoded.attention_mask.to(device),
    }
    layers = get_layers(model)
    captured_keys = [None] * len(layers)
    captured_values = [None] * len(layers)
    handles = []
    for layer_index, layer in enumerate(layers):
        attention = layer.self_attn

        def hook(module, args, kwargs, layer_index=layer_index):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            key, value = project_native_kv(module, hidden)
            captured_keys[layer_index] = key[0, :, indices, :].detach().half().cpu()
            captured_values[layer_index] = value[0, :, indices, :].detach().half().cpu()

        handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        model(**model_inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if any(tensor is None for tensor in captured_keys + captured_values):
        raise RuntimeError(f"Failed to capture every Llama layer for {row['id']}")
    return {
        **row,
        "memory": {
            "keys": captured_keys,
            "values": captured_values,
            "answer_token_mask": answer_mask,
        },
        "evidence_token_ids": encoded.input_ids[0, indices].tolist(),
        "canonical_token_spans": sender_spans,
        "named_character_spans": dict(named_spans),
        "transport": {
            "teacher_pool_from_sender": teacher_pool,
            "sender_mass_to_teacher": sender_mass,
            "sender_span_masks": sender_masks,
            "teacher_span_masks": teacher_masks,
        },
        "evidence_tokens": len(indices),
        "teacher_evidence_tokens": len(teacher_spans),
        "sender_tokens": int(encoded.input_ids.shape[1]),
        "prompt_style": "llama_fewshot_join_reason",
    }


def main():
    parser = argparse.ArgumentParser(description="Cache Llama Native KV with Qwen token transport")
    parser.add_argument(
        "--model",
        default="/home/yezhe/all_models/models/LLM-Research/Llama-3___2-3B-Instruct",
    )
    parser.add_argument("--qwen-tokenizer", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    rows = load_jsonl(args.data)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["variant"]] = row
    pairs = [
        (pair_id, variants)
        for pair_id, variants in grouped.items()
        if {"base", "counterfactual"}.issubset(variants)
    ]
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    teacher_cache = TeacherCache(args.teacher_index)
    if len(pairs) > len(teacher_cache.entries):
        raise ValueError("Teacher cache is shorter than requested Llama cache")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_tokenizer, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    pair_files = []
    sender_counts = []
    teacher_counts = []
    for pair_index, (pair_id, variants) in enumerate(tqdm(pairs, desc="llama_native_kv_pairs")):
        teacher_entry = teacher_cache.entries[pair_index]
        if teacher_entry["pair_id"] != pair_id:
            raise RuntimeError(f"Pair order mismatch: {pair_id} != {teacher_entry['pair_id']}")
        teacher_pair = teacher_cache.load(pair_index)
        examples = []
        for variant in ("base", "counterfactual"):
            example = encode_example(
                model,
                tokenizer,
                qwen_tokenizer,
                variants[variant],
                teacher_pair[variant],
                device,
                args.max_length,
            )
            sender_counts.append(example["evidence_tokens"])
            teacher_counts.append(example["teacher_evidence_tokens"])
            examples.append(example)
        name = f"pair_{pair_index:05d}.pt"
        torch.save({"pair_id": pair_id, "examples": examples}, output / name)
        pair_files.append(
            {
                "pair_id": pair_id,
                "file": name,
                "base_answer": variants["base"]["answer"],
                "counterfactual_answer": variants["counterfactual"]["answer"],
            }
        )

    config = model.config
    with open(output / "index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "format_version": 4,
                "coordinate_system": "llama_pre_rope_k_native_v_with_character_transport",
                "model": args.model,
                "teacher_index": args.teacher_index,
                "data": args.data,
                "prompt_style": "llama_fewshot_join_reason",
                "pairs": len(pair_files),
                "layers": int(config.num_hidden_layers),
                "query_heads": int(config.num_attention_heads),
                "kv_heads": int(config.num_key_value_heads),
                "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
                "min_evidence_tokens": min(sender_counts),
                "max_evidence_tokens": max(sender_counts),
                "min_teacher_tokens": min(teacher_counts),
                "max_teacher_tokens": max(teacher_counts),
                "pair_files": pair_files,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with open(output / "CACHE_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", "pairs": len(pair_files)}, handle, indent=2)


if __name__ == "__main__":
    main()
