import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from p3d2_common import (
    OracleEvidenceReader, aggregate_scores, answer_scores, build_cache, extract_prediction,
    hard_negative_mapping, load_receiver, load_span_probe, memory_from_payload,
    normalize_answer, oracle_view, question_prompt, read_json, resize_memory, seed_everything,
    span_teacher, write_json, write_jsonl, zero_memory,
)


@torch.inference_mode()
def generate(model, tokenizer, reader, row, memory, enabled, max_new_tokens):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    kwargs = dict(
        **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    started = time.perf_counter()
    if enabled:
        with reader.inject(model, memory):
            output = model.generate(**kwargs)
    else:
        output = model.generate(**kwargs)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
    return tokenizer.decode(tokens, skip_special_tokens=True), tokens, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--span-probe", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    protocol = read_json(args.protocol)
    cache = build_cache(protocol, args.split)
    negative = hard_negative_mapping(cache)
    checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=False)
    model, tokenizer = load_receiver(args.model, device)
    metadata = checkpoint["reader_metadata"]
    reader = OracleEvidenceReader(
        model, metadata["groups"], metadata["memory_dim"], metadata["rank"],
        metadata["adapter_rank"], active_layers=metadata["active_layers"],
    ).to(device)
    if reader.metadata() != metadata:
        raise RuntimeError("Oracle Reader interface differs from P3-D checkpoint")
    reader.load_state_dict(checkpoint["reader"], strict=True)
    reader.eval().requires_grad_(False)
    probe = load_span_probe(args.span_probe, device)
    modes = ("standard", "oracle_token", "oracle_token_layer")
    conditions = ("correct", "shuffled", "zero", "reader_off")
    limit = min(len(cache), args.max_samples or len(cache))
    records = []
    for index in tqdm(range(limit), desc="p3d2_oracle_grounding"):
        payload, wrong_payload = cache.load(index), cache.load(negative[index])
        current = memory_from_payload(payload, device)
        wrong = resize_memory(memory_from_payload(wrong_payload, device), current["keys"].shape[1])
        current_probe = span_teacher(probe, payload, device)
        wrong_probe = span_teacher(probe, wrong_payload, device)
        for mode in modes:
            for condition in conditions:
                enabled, source_answer = True, ""
                if condition == "correct":
                    memory, teacher = current, current_probe
                elif condition == "shuffled":
                    memory, teacher = wrong, wrong_probe
                    source_answer = wrong_payload["row"]["answer"]
                elif condition == "zero":
                    memory, teacher = zero_memory(current), current_probe
                else:
                    memory, teacher, enabled = current, current_probe, False
                if enabled and condition != "zero":
                    memory = oracle_view(memory, mode, teacher, args.top_layers)
                text, token_ids, elapsed = generate(
                    model, tokenizer, reader, payload["row"], memory, enabled, args.max_new_tokens
                )
                prediction, _ = extract_prediction(text)
                em, f1 = answer_scores(prediction, payload["row"]["answer"])
                records.append({
                    "sample_index": index, "sample_id": payload["row"]["id"],
                    "mode": mode, "condition": condition, "question_type": payload["row"].get("type", "unknown"),
                    "gold_answer": payload["row"]["answer"], "source_memory_answer": source_answer,
                    "prediction": prediction, "raw_generation": text, "token_ids": token_ids,
                    "em": em, "f1": f1,
                    "source_answer_em": float(source_answer != "" and normalize_answer(prediction) == normalize_answer(source_answer)),
                    "elapsed_seconds": elapsed,
                })
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    summaries = {}
    for mode in modes:
        summary = aggregate_scores([{**row, "condition": row["condition"]} for row in records if row["mode"] == mode])
        summaries[mode] = summary
    standard = summaries["standard"]["correct"]
    oracle = summaries["oracle_token_layer"]["correct"]
    write_json(output / "SUCCESS.json", {
        "status": "complete", "n": limit, "modes": summaries,
        "oracle_gain": {
            "f1": oracle["f1"] - standard["f1"],
            "bridge_f1": oracle.get("bridge", {}).get("f1", 0.0) - standard.get("bridge", {}).get("f1", 0.0),
            "correct_shuffled_gap_gain": summaries["oracle_token_layer"]["correct_minus_shuffled_f1"] - summaries["standard"]["correct_minus_shuffled_f1"],
        },
        "diagnosis_rule": "Large Oracle bridge/gap gains imply grounding failure; weak Oracle gains imply execution incompatibility.",
        "writer_frozen": True, "receiver_backbone_frozen": True, "reader_checkpoint_unchanged": True,
        "args": vars(args),
    })


if __name__ == "__main__":
    main()
