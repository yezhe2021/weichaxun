import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import (answer_scores, generate, hard_negative_mapping, load_receiver,
                         normalize_answer, question_prompt, seed_everything)
from p3e_c1_common import LearnableCanonicalHeadReader
from p3e_f_common import CanonicalCache, memory_to, read_json, write_json, write_jsonl


@torch.inference_mode()
def plain_generate(model, tokenizer, row, max_new_tokens):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
    from p3d3_common import extract_prediction
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    prediction, method = extract_prediction(text)
    return {"text": text, "prediction": prediction, "parse_method": method,
            "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens}


def trace_diagnostics(trace, support_mask):
    support = support_mask.float()
    values = []
    for calls in trace.values():
        for call in calls:
            attention = call["attention"].detach().float().cpu()
            values.append(float((attention * support[None, None, None, None, :]).sum(-1).mean()))
    return sum(values) / len(values) if values else 0.0


def summarize(records, condition):
    selected = [row for row in records if row["condition"] == condition]
    result = {
        "n": len(selected), "em": sum(row["em"] for row in selected) / len(selected),
        "f1": sum(row["f1"] for row in selected) / len(selected),
        "eos_rate": sum(row["output"]["eos_reached"] for row in selected) / len(selected),
        "average_output_tokens": sum(len(row["output"]["token_ids"]) for row in selected) / len(selected),
        "by_type": {},
    }
    for kind in ("bridge", "comparison"):
        group = [row for row in selected if row["type"] == kind]
        result["by_type"][kind] = {
            "n": len(group), "em": sum(row["em"] for row in group) / len(group),
            "f1": sum(row["f1"] for row in group) / len(group),
        }
    if condition == "correct_canonical":
        result["supporting_fact_attention_mass"] = sum(
            row.get("supporting_fact_attention_mass", 0.0) for row in selected
        ) / len(selected)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=int, choices=[1024, 2048], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = CanonicalCache(args.memory_index, args.data, capacity=2)
    if len(cache) != 64:
        raise RuntimeError("Evaluation must use the fixed 64-example validation set")
    if all("hard_negative_index" in entry for entry in cache.entries):
        negatives = [int(entry["hard_negative_index"]) for entry in cache.entries]
    else:
        negatives = hard_negative_mapping(cache)
    model, tokenizer = load_receiver(args.model, device)
    checkpoint = torch.load(args.reader, map_location="cpu", weights_only=False)
    metadata = checkpoint["reader_metadata"]
    reader = LearnableCanonicalHeadReader(
        model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"],
        metadata["top_k"], 0.25
    ).to(device)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    reader.eval()
    conditions = ["question_only", "correct_canonical", "hard_shuffled_canonical",
                  "oracle_support_canonical", "reader_off"]
    records, pair_rows = [], []
    for index in tqdm(range(64), desc=f"p3e_f_eval_train{args.scale}"):
        payload = cache.load(index)
        wrong = cache.load(negatives[index])
        row = payload["row"]
        predictions = {}
        for condition in conditions:
            trace = {} if condition == "correct_canonical" else None
            if condition == "question_only":
                result = plain_generate(model, tokenizer, row, args.max_new_tokens)
            elif condition == "reader_off":
                result = generate(model, tokenizer, reader, row, memory_to(payload, device),
                                  args.max_new_tokens, enabled=False)
            elif condition == "correct_canonical":
                result = generate(model, tokenizer, reader, row, memory_to(payload, device),
                                  args.max_new_tokens, trace=trace)
            elif condition == "hard_shuffled_canonical":
                result = generate(model, tokenizer, reader, row, memory_to(wrong, device),
                                  args.max_new_tokens)
            else:
                result = generate(model, tokenizer, reader, row, memory_to(payload, device, True),
                                  args.max_new_tokens)
            em, f1 = answer_scores(result["prediction"], row["answer"])
            item = {"id": row["id"], "type": row["type"], "answer": row["answer"],
                    "condition": condition, "em": em, "f1": f1, "output": result}
            if trace is not None:
                item["supporting_fact_attention_mass"] = trace_diagnostics(
                    trace, payload["support_mask"]
                )
            if condition == "hard_shuffled_canonical":
                item.update({"source_id": wrong["row"]["id"], "source_answer": wrong["row"]["answer"]})
            records.append(item)
            predictions[condition] = result["prediction"]
        pair_rows.append({
            "id": row["id"],
            "correct_shuffled_switch": float(
                normalize_answer(predictions["correct_canonical"]) !=
                normalize_answer(predictions["hard_shuffled_canonical"])
            ),
            "question_only_equals_reader_off": float(
                predictions["question_only"] == predictions["reader_off"]
            ),
        })
    write_jsonl(output / "per_sample_generation.jsonl", records)
    metrics = {condition: summarize(records, condition) for condition in conditions}
    correct = metrics["correct_canonical"]["f1"]
    shuffled = metrics["hard_shuffled_canonical"]["f1"]
    summary = {
        "status": "complete", "experiment": "P3-E-F Reader scale study",
        "train_scale": args.scale, "validation_samples": 64, "conditions": metrics,
        "correct_shuffled_f1_gap": correct - shuffled,
        "prediction_switch_rate": sum(row["correct_shuffled_switch"] for row in pair_rows) / 64,
        "reader_off_exact_output_consistency": sum(
            row["question_only_equals_reader_off"] for row in pair_rows
        ) / 64,
        "reader_gates": reader.gates().detach().cpu().tolist(),
        "reader_canonical_head_usage": reader.routes().detach().cpu().mean(dim=1).tolist(),
        "reader": args.reader, "memory_index": args.memory_index,
    }
    write_json(output / "SUCCESS.json", summary)


if __name__ == "__main__":
    main()
