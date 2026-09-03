import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from experiment import (
    MemoryCache,
    answer_scores,
    build_adapters,
    full_text_prompt,
    generate_adapted,
    generate_plain,
    load_receiver,
    memory_to,
    normalize_answer,
    question_prompt,
    read_json,
    seed_everything,
    summarize_condition,
    trace_support_mass,
    write_json,
    write_jsonl,
)


def load_variant(model, checkpoint_path, memory_dim, seed, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reader, receiver_lora = build_adapters(model, memory_dim, checkpoint["variant"], seed)
    if reader is not None:
        reader.load_state_dict(checkpoint["reader"])
        reader.to(device).eval()
    if receiver_lora is not None:
        receiver_lora.load_state_dict(checkpoint["receiver_lora"])
        receiver_lora.to(device).eval()
    return reader, receiver_lora, checkpoint


def average_support_mass(items):
    result = {}
    for layer in sorted({layer for item in items for layer in item}):
        result[layer] = {}
        for phase in ("prompt", "decode"):
            values = [item[layer][phase] for item in items if layer in item and item[layer][phase] is not None]
            result[layer][phase] = torch.tensor(values).mean(0).tolist() if values else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--reader-only-checkpoint", required=True)
    parser.add_argument("--reader-lora-checkpoint", required=True)
    parser.add_argument("--lora-only-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = MemoryCache(args.memory)
    negatives = read_json(args.negatives)["mapping"]
    count = min(args.max_samples or len(cache), len(cache))
    model, tokenizer = load_receiver(args.receiver, device)
    memory_dim = int(cache.index["memory_dim"])
    reader_only, _, reader_only_checkpoint = load_variant(
        model, args.reader_only_checkpoint, memory_dim, args.seed, device
    )
    joint_reader, joint_lora, reader_lora_checkpoint = load_variant(
        model, args.reader_lora_checkpoint, memory_dim, args.seed, device
    )
    _, lora_only, lora_only_checkpoint = load_variant(
        model, args.lora_only_checkpoint, memory_dim, args.seed, device
    )
    conditions = (
        "question_only",
        "gold_full_text",
        "reader_only_correct",
        "reader_only_shuffled",
        "reader_lora_correct",
        "reader_lora_shuffled",
        "reader_lora_off",
        "lora_only",
    )
    records, pairs = [], []
    support_masses = {"reader_only_correct": [], "reader_lora_correct": []}
    for index in tqdm(range(count), desc="evaluate_cross_attention_reader"):
        payload = cache.load(index)
        wrong_payload = cache.load(negatives[index])
        row = payload["row"]
        correct_memory = memory_to(payload, device)
        wrong_memory = memory_to(wrong_payload, device)
        predictions = {}
        for condition in conditions:
            trace = {} if condition in support_masses else None
            if condition == "question_only":
                result = generate_plain(model, tokenizer, question_prompt(tokenizer, row), args.max_new_tokens)
            elif condition == "gold_full_text":
                result = generate_plain(model, tokenizer, full_text_prompt(tokenizer, row), args.max_new_tokens)
            elif condition == "reader_only_correct":
                result = generate_adapted(model, tokenizer, row, correct_memory, reader_only, None, args.max_new_tokens, True, False, trace)
            elif condition == "reader_only_shuffled":
                result = generate_adapted(model, tokenizer, row, wrong_memory, reader_only, None, args.max_new_tokens, True, False)
            elif condition == "reader_lora_correct":
                result = generate_adapted(model, tokenizer, row, correct_memory, joint_reader, joint_lora, args.max_new_tokens, True, True, trace)
            elif condition == "reader_lora_shuffled":
                result = generate_adapted(model, tokenizer, row, wrong_memory, joint_reader, joint_lora, args.max_new_tokens, True, True)
            elif condition == "reader_lora_off":
                result = generate_adapted(model, tokenizer, row, None, joint_reader, joint_lora, args.max_new_tokens, False, True)
            else:
                result = generate_adapted(model, tokenizer, row, None, None, lora_only, args.max_new_tokens, False, True)
            em, f1 = answer_scores(result["prediction"], row["answer"])
            predictions[condition] = result["prediction"]
            item = {
                "id": row["id"],
                "type": row["type"],
                "answer": row["answer"],
                "condition": condition,
                "em": em,
                "f1": f1,
                "output": result,
            }
            if condition.endswith("shuffled"):
                item["memory_source_id"] = wrong_payload["row"]["id"]
                item["memory_source_answer"] = wrong_payload["row"]["answer"]
            if trace is not None:
                mass = trace_support_mass(trace, correct_memory["support_mask"])
                item["supporting_fact_attention_mass"] = mass
                support_masses[condition].append(mass)
            records.append(item)
        pairs.append({
            "id": row["id"],
            "reader_only_prediction_switch": float(
                normalize_answer(predictions["reader_only_correct"]) != normalize_answer(predictions["reader_only_shuffled"])
            ),
            "reader_lora_prediction_switch": float(
                normalize_answer(predictions["reader_lora_correct"]) != normalize_answer(predictions["reader_lora_shuffled"])
            ),
            "joint_off_equals_question_only": float(predictions["reader_lora_off"] == predictions["question_only"]),
        })
    write_jsonl(output / "per_sample_generation.jsonl", records)
    metrics = {condition: summarize_condition(records, condition) for condition in conditions}
    question_f1 = metrics["question_only"]["f1"]
    result = {
        "status": "complete",
        "samples": count,
        "conditions": metrics,
        "reader_only_correct_shuffled_gap": metrics["reader_only_correct"]["f1"] - metrics["reader_only_shuffled"]["f1"],
        "reader_lora_correct_shuffled_gap": metrics["reader_lora_correct"]["f1"] - metrics["reader_lora_shuffled"]["f1"],
        "reader_only_question_gain": metrics["reader_only_correct"]["f1"] - question_f1,
        "reader_lora_question_gain": metrics["reader_lora_correct"]["f1"] - question_f1,
        "reader_only_prediction_switch": sum(row["reader_only_prediction_switch"] for row in pairs) / len(pairs),
        "reader_lora_prediction_switch": sum(row["reader_lora_prediction_switch"] for row in pairs) / len(pairs),
        "reader_lora_off_exact_output_consistency": sum(row["joint_off_equals_question_only"] for row in pairs) / len(pairs),
        "gates": {
            "reader_only": reader_only.gates().detach().cpu().tolist(),
            "reader_lora": joint_reader.gates().detach().cpu().tolist(),
        },
        "supporting_fact_attention_mass": {
            condition: average_support_mass(values) for condition, values in support_masses.items()
        },
        "checkpoints": {
            "reader_only": args.reader_only_checkpoint,
            "reader_lora": args.reader_lora_checkpoint,
            "lora_only": args.lora_only_checkpoint,
        },
        "checkpoint_training_losses": {
            "reader_only": reader_only_checkpoint["train_loss"],
            "reader_lora": reader_lora_checkpoint["train_loss"],
            "lora_only": lora_only_checkpoint["train_loss"],
        },
    }
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
