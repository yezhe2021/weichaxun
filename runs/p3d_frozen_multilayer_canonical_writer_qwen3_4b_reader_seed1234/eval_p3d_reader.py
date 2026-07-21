import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from p3d_common import (
    INSUFFICIENT, MultiLayerEvidenceReader, answer_scores, compose_memory, extract_prediction,
    generate, load_receiver, memory_to, normalize_answer, permute_layers, permute_tokens,
    read_json, resize_memory, seed_everything, write_json, write_jsonl, zero_memory,
)
from train_p3d_reader import build_cache, negative_mapping, source_spec


VALID_MEMORY_CONDITIONS = {"correct", "layer_permutation", "token_permutation"}
REJECTION_CONDITIONS = {
    "shuffled", "zero", "k_correct_v_wrong", "k_wrong_v_correct",
    "k_correct_v_zero", "k_zero_v_correct",
}


def build_condition(name, current, wrong, seed):
    tokens = current["keys"].shape[1]
    wrong = resize_memory(wrong, tokens)
    if name == "correct": return current, True
    if name == "shuffled": return wrong, True
    if name == "zero": return zero_memory(current), True
    if name == "k_correct_v_wrong": return compose_memory(current, wrong, tokens), True
    if name == "k_wrong_v_correct": return compose_memory(wrong, current, tokens), True
    if name == "k_correct_v_zero": return compose_memory(current, zero_memory(current), tokens), True
    if name == "k_zero_v_correct": return compose_memory(zero_memory(current), current, tokens), True
    if name == "layer_permutation": return permute_layers(current), True
    if name == "token_permutation": return permute_tokens(current, seed), True
    if name == "reader_off": return current, False
    if name == "question_only": return zero_memory(current), False
    raise ValueError(name)


def aggregate(records):
    grouped = defaultdict(list)
    for row in records: grouped[row["condition"]].append(row)
    summary = {}
    for condition, rows in grouped.items():
        item = {
            "n": len(rows),
            "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
            "f1": sum(row["f1"] for row in rows) / len(rows),
            "eos_rate": sum(row["eos_reached"] for row in rows) / len(rows),
            "mean_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows) / len(rows),
            "insufficient_rate": sum(row["is_insufficient"] for row in rows) / len(rows),
            "source_answer_exact_match": sum(row["source_answer_exact_match"] for row in rows) / len(rows),
        }
        if condition in REJECTION_CONDITIONS:
            item["rejection_accuracy"] = item["insufficient_rate"]
        for kind in ("bridge", "comparison"):
            subset = [row for row in rows if row["question_type"] == kind]
            if subset:
                item[kind] = {
                    "n": len(subset),
                    "exact_match": sum(row["exact_match"] for row in subset) / len(subset),
                    "f1": sum(row["f1"] for row in subset) / len(subset),
                }
        summary[condition] = item
    return summary


def aggregate_diagnostics(records, receiver_layers, groups):
    buckets = {layer: {"n": 0, "gate": 0.0, "delta_norm": 0.0, "attention_entropy": 0.0, "router": [0.0] * groups} for layer in receiver_layers}
    for row in records:
        if row["condition"] != "correct": continue
        for diagnostic in row["reader_diagnostics"]:
            bucket = buckets[diagnostic["receiver_layer"]]; bucket["n"] += 1
            for name in ("gate", "delta_norm", "attention_entropy"): bucket[name] += diagnostic[name]
            bucket["router"] = [left + right for left, right in zip(bucket["router"], diagnostic["canonical_router"])]
    output = []
    for layer in receiver_layers:
        bucket = buckets[layer]; count = max(1, bucket.pop("n"))
        output.append({"receiver_layer": layer, **{name: bucket[name] / count for name in ("gate", "delta_norm", "attention_entropy")}, "canonical_router": [value / count for value in bucket["router"]]})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--protocol", required=True)
    parser.add_argument("--source", choices=("canonical16", "native16", "canonical36"), required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--out", required=True); parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device)
    protocol = read_json(args.protocol); cache = build_cache(protocol, args.source, args.split)
    _, groups, memory_dim, _ = source_spec(protocol, args.source, args.split)
    model, tokenizer = load_receiver(args.model, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["reader_metadata"]
    reader = MultiLayerEvidenceReader(model, groups, memory_dim, metadata["rank"], metadata["adapter_rank"], active_layers=metadata["active_layers"]).to(device)
    if reader.metadata() != metadata: raise RuntimeError("Reader checkpoint interface mismatch")
    reader.load_state_dict(checkpoint["reader"]); reader.eval()
    for parameter in reader.parameters(): parameter.requires_grad_(False)
    negative = negative_mapping(cache)
    conditions = [
        "correct", "shuffled", "zero", "k_correct_v_wrong", "k_wrong_v_correct",
        "k_correct_v_zero", "k_zero_v_correct", "layer_permutation", "token_permutation",
        "reader_off", "question_only",
    ]
    limit = len(cache) if args.max_samples <= 0 else min(args.max_samples, len(cache))
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    records, started = [], time.perf_counter()
    for index in tqdm(range(limit), desc=f"p3d_eval_{args.source}"):
        payload, other_payload = cache.load(index), cache.load(negative[index])
        current, wrong = memory_to(payload, device), memory_to(other_payload, device)
        for condition in conditions:
            memory, enabled = build_condition(condition, current, wrong, args.seed + index)
            result = generate(model, tokenizer, reader, payload["row"], memory, args.max_new_tokens, enabled)
            prediction, parse_status = extract_prediction(result["text"])
            exact, f1 = answer_scores(prediction, payload["row"]["answer"])
            source_exact, source_f1 = answer_scores(prediction, other_payload["row"]["answer"])
            records.append({
                "sample_index": index, "sample_id": payload["row"].get("id", str(index)), "condition": condition,
                "question_type": payload["row"].get("type", "unknown"), "question": payload["row"]["question"],
                "gold_answer": payload["row"]["answer"], "source_memory_answer": other_payload["row"]["answer"],
                "raw_generation": result["text"], "prediction": prediction, "parse_status": parse_status,
                "exact_match": exact, "f1": f1, "source_answer_exact_match": source_exact, "source_answer_f1": source_f1,
                "is_insufficient": float(normalize_answer(prediction) == normalize_answer(INSUFFICIENT)),
                "eos_reached": float(result["eos_reached"]), "generated_token_ids": result["token_ids"],
                "elapsed_seconds": result["elapsed_seconds"],
                "reader_diagnostics": result["diagnostics"] if condition == "correct" else [],
            })
            write_jsonl(output / "per_sample_generation.jsonl", records)
    by_sample = defaultdict(dict)
    for row in records: by_sample[row["sample_id"]][row["condition"]] = row["prediction"]
    off_consistency = sum(float(values.get("reader_off") == values.get("question_only")) for values in by_sample.values()) / max(1, len(by_sample))
    keys, values = memory_to(cache.load(0), torch.device("cpu"))["keys"], memory_to(cache.load(0), torch.device("cpu"))["values"]
    memory_bytes = (keys.numel() * keys.element_size()) + (values.numel() * values.element_size())
    summary = {
        "status": "complete", "source": args.source, "split": args.split, "n": limit,
        "conditions": aggregate(records), "reader_off_question_only_consistency": off_consistency,
        "reader_diagnostics_correct": aggregate_diagnostics(records, metadata["active_layers"], groups),
        "reader_parameters": sum(parameter.numel() for parameter in reader.parameters()),
        "receiver_parameters_updated": 0, "memory_groups": groups, "memory_dim": memory_dim,
        "example_memory_bytes_float32": memory_bytes, "total_elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(Path(args.checkpoint).resolve()), "reader_metadata": metadata, "args": vars(args),
    }
    write_json(output / "SUCCESS.json", summary)


if __name__ == "__main__": main()
