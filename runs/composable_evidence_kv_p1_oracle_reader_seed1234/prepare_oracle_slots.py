import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p1_common import parse_dtype, resolve_device


def unique_titles(supporting_facts):
    titles = []
    for title, _ in supporting_facts:
        if title not in titles:
            titles.append(title)
    return titles


def row_to_example(row, source_titles, slots_per_source):
    context = {title: sentences for title, sentences in row.get("context", [])}
    support_by_title = defaultdict(list)
    for title, sentence_index in row.get("supporting_facts", []):
        sentences = context.get(title, [])
        if 0 <= sentence_index < len(sentences):
            text = sentences[sentence_index].strip()
            if text:
                support_by_title[title].append(text)
    if len(source_titles) != 2 or any(not support_by_title[title] for title in source_titles):
        return None

    source_sentences = []
    for title in source_titles:
        sentences = support_by_title[title]
        if len(sentences) > slots_per_source:
            sentences = sentences[: slots_per_source - 1] + [" ".join(sentences[slots_per_source - 1 :])]
        source_sentences.append(sentences)
    return {
        "id": row.get("_id", ""),
        "question": str(row.get("question", "")).strip(),
        "answer": str(row.get("answer", "")).strip(),
        "type": row.get("type", ""),
        "level": row.get("level", ""),
        "source_titles": list(source_titles),
        "source_sentences": source_sentences,
    }


def load_examples(data_path, max_samples, slots_per_source, seed, manifest_path=None):
    with open(data_path, encoding="utf-8") as handle:
        rows = json.load(handle)
    by_id = {row.get("_id", ""): row for row in rows}
    examples = []
    if manifest_path:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = [json.loads(line) for line in handle if line.strip()]
        for item in manifest[:max_samples if max_samples > 0 else None]:
            row = by_id.get(item["id"])
            if row is None:
                raise KeyError(f"Manifest id is absent from raw data: {item['id']}")
            source_titles = [item["source_a_titles"][0], item["source_b_titles"][0]]
            example = row_to_example(row, source_titles, slots_per_source)
            if example is None:
                raise ValueError(f"Manifest example cannot produce oracle slots: {item['id']}")
            examples.append(example)
        return examples

    random.Random(seed).shuffle(rows)
    for row in rows:
        source_titles = unique_titles(row.get("supporting_facts", []))
        if len(source_titles) != 2:
            continue
        sample_rng = random.Random(f"{seed}:{row.get('_id', '')}")
        sample_rng.shuffle(source_titles)
        example = row_to_example(row, source_titles, slots_per_source)
        if example is None or not example["question"] or not example["answer"]:
            continue
        examples.append(example)
        if max_samples > 0 and len(examples) >= max_samples:
            break
    return examples


def pool_sentences(model, tokenizer, question, sentences, layer, max_length, device):
    prefixes = [f"Question:\n{question}\n\nEvidence sentence:\n" for _ in sentences]
    prompts = [prefix + sentence for prefix, sentence in zip(prefixes, sentences)]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    ).to(device)
    prefix_lengths = [
        len(tokenizer(prefix, add_special_tokens=True, truncation=True, max_length=max_length).input_ids)
        for prefix in prefixes
    ]
    output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    if layer < 0 or layer + 1 >= len(output.hidden_states):
        raise ValueError(f"sender layer {layer} is invalid for {len(output.hidden_states) - 1} layers")
    hidden = output.hidden_states[layer + 1]
    pooled = []
    for index, prefix_length in enumerate(prefix_lengths):
        sequence_length = int(encoded.attention_mask[index].sum().item())
        start = min(prefix_length, sequence_length - 1)
        pooled.append(hidden[index, start:sequence_length].float().mean(dim=0))
    return torch.stack(pooled)


def main():
    parser = argparse.ArgumentParser(description="Build frozen-sender oracle sentence slots for P1")
    parser.add_argument("--sender-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--slots-per-source", type=int, default=4)
    parser.add_argument("--sender-layer", type=int, default=14)
    parser.add_argument("--max-sender-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    examples = load_examples(
        args.data,
        args.max_samples,
        args.slots_per_source,
        args.seed,
        args.manifest,
    )
    if not examples:
        raise RuntimeError("No P1 examples were selected")

    tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.sender_model,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cached = []
    with torch.inference_mode():
        for example in tqdm(examples, desc="oracle_slots"):
            slots = torch.zeros(
                2,
                args.slots_per_source,
                model.config.hidden_size,
                dtype=torch.float16,
            )
            slot_mask = torch.zeros(2, args.slots_per_source, dtype=torch.bool)
            for source_index, sentences in enumerate(example["source_sentences"]):
                pooled = pool_sentences(
                    model,
                    tokenizer,
                    example["question"],
                    sentences,
                    args.sender_layer,
                    args.max_sender_tokens,
                    device,
                ).cpu().to(torch.float16)
                slots[source_index, : pooled.shape[0]] = pooled
                slot_mask[source_index, : pooled.shape[0]] = True
            cached.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "answer": example["answer"],
                    "type": example["type"],
                    "level": example["level"],
                    "source_titles": example["source_titles"],
                    "source_sentences": example["source_sentences"],
                    "slots": slots,
                    "slot_mask": slot_mask,
                }
            )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "sender_model": args.sender_model,
            "sender_layer": args.sender_layer,
            "slots_per_source": args.slots_per_source,
            "hidden_size": int(model.config.hidden_size),
            "seed": args.seed,
            "data": args.data,
            "manifest": args.manifest,
            "examples": cached,
        },
        output,
    )
    print(json.dumps({"status": "complete", "out": str(output), "n": len(cached)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
