import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import answer_scores, apply_chat, extract_prediction, normalize_answer, write_json, write_jsonl
from .modeling import load_frozen_model, load_tokenizer, validate_architecture
from .train_utils import HiddenCache, build_reader, build_writer, cached_memory, different_answer_partner


@torch.inference_mode()
def generate(model, tokenizer, reader, row, memory, condition, max_new_tokens):
    evidence = row["evidence_text"] if condition == "full_text" else None
    prompt = apply_chat(tokenizer, row["question"], include_evidence=evidence)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    kwargs = dict(
        **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    if memory is None:
        output = model.generate(**kwargs)
    else:
        with reader.inject(model, memory):
            output = model.generate(**kwargs)
    generated = output[0, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), generated.tolist()


def summarize(records):
    result = {}
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        result[condition] = {
            "n": len(selected),
            "em": float(np.mean([row["em"] for row in selected])),
            "f1": float(np.mean([row["f1"] for row in selected])),
            "generation_tokens": float(np.mean([row["generation_tokens"] for row in selected])),
            "source_answer_em": float(np.mean([row.get("source_answer_em", 0.0) for row in selected])),
        }
    question = result.get("question_only")
    text = result.get("full_text")
    public = result.get("correct_public")
    if question and text and public:
        for metric in ("em", "f1"):
            denominator = text[metric] - question[metric]
            result["gap_recovery_" + metric] = (
                (public[metric] - question[metric]) / denominator if denominator else None
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cache = HiddenCache(args.cache)
    count = min(len(cache), args.limit) if args.limit else len(cache)
    payloads = [cache.load(index) for index in range(count)]
    rows = [payload["row"] for payload in payloads]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = load_tokenizer(args.receiver_model)
    model = load_frozen_model(args.receiver_model, "qwen3", args.device, torch.float16)
    validate_architecture(model, "qwen3")
    reader = build_reader(model, checkpoint["reader_metadata"]).to(device=args.device, dtype=torch.float16).eval()
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    writer = build_writer().to(device=args.device, dtype=torch.float16).eval()
    writer.load_state_dict(checkpoint["writer"])
    writer.requires_grad_(False)
    records = []
    for index, payload in enumerate(tqdm(payloads, desc="free_running_eval")):
        row = payload["row"]
        other_index = different_answer_partner(rows, index, args.seed)
        other = payloads[other_index]
        correct = cached_memory(writer, payload["correct"], args.device)
        shuffled = cached_memory(writer, other["correct"], args.device)
        conditions = [
            ("question_only", None, ""),
            ("full_text", None, ""),
            ("correct_public", correct, ""),
            ("shuffled_public", shuffled, other["row"]["answer"]),
            ("zero_public", correct.zero(), ""),
            ("reader_off", None, ""),
        ]
        if "answer_removed" in payload:
            conditions.append(("answer_sentence_removed", cached_memory(writer, payload["answer_removed"], args.device), ""))
        for condition, memory, source_answer in conditions:
            text, token_ids = generate(model, tokenizer, reader, row, memory, condition, args.max_new_tokens)
            prediction = extract_prediction(text)
            em, f1 = answer_scores(prediction, row["answer"])
            source_em = float(normalize_answer(prediction) == normalize_answer(source_answer)) if source_answer else 0.0
            records.append({
                "id": row["id"], "condition": condition, "target": row["answer"],
                "source_memory_answer": source_answer, "prediction": prediction, "generated_text": text,
                "token_ids": token_ids, "generation_tokens": len(token_ids), "em": em, "f1": f1,
                "source_answer_em": source_em,
            })
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_json(output / "SUCCESS.json", {"status": "complete", "samples": count, "metrics": summarize(records)})


if __name__ == "__main__":
    main()
