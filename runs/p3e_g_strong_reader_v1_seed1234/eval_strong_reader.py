import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import (answer_scores, extract_prediction, generate, hard_negative_mapping,
                         load_receiver, normalize_answer, question_prompt, seed_everything)
from p3e_f_common import CanonicalCache, memory_to, write_json, write_jsonl
from p3e_g_common import StrongCanonicalReader, load_old_reader


@torch.inference_mode()
def plain_generate(model, tokenizer, row, max_new_tokens):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    prediction, method = extract_prediction(text)
    return {"text": text, "prediction": prediction, "parse_method": method,
            "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens}


def tensor_stats(values):
    flat = torch.cat([value.detach().float().cpu().reshape(-1) for value in values])
    return {"mean": float(flat.mean()), "std": float(flat.std(unbiased=False)),
            "min": float(flat.min()), "max": float(flat.max())}


def trace_diagnostics(trace, support_mask):
    support = support_mask.float()
    result = {}
    for layer, calls in trace.items():
        gates, norm_ratios, cosines, support_masses = [], [], [], []
        for call in calls:
            gates.append(call["token_gate"])
            projected = call["projected"].detach().float()
            update = call["adapter_update"].detach().float()
            adapted = call["adapted"].detach().float()
            norm_ratios.append(float(update.norm() / projected.norm().clamp_min(1e-8)))
            cosines.append(float(F.cosine_similarity(
                adapted.reshape(-1, adapted.shape[-1]),
                projected.reshape(-1, projected.shape[-1]), dim=-1
            ).mean()))
            attention = call["attention"].detach().float().cpu()
            support_masses.append(float(
                (attention * support[None, None, None, None, :]).sum(-1).mean()
            ))
        result[str(layer)] = {
            "token_gate": tensor_stats(gates),
            "adapter_update_to_o_proj_norm_ratio": sum(norm_ratios) / len(norm_ratios),
            "adapted_to_o_proj_cosine": sum(cosines) / len(cosines),
            "supporting_fact_attention_mass": sum(support_masses) / len(support_masses),
        }
    return result


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
    return result


def aggregate_diagnostics(rows):
    by_layer = {}
    for row in rows:
        for layer, values in row["layers"].items():
            by_layer.setdefault(layer, []).append(values)
    result = {}
    for layer, values in by_layer.items():
        result[layer] = {
            "token_gate_mean": sum(item["token_gate"]["mean"] for item in values) / len(values),
            "token_gate_std_mean": sum(item["token_gate"]["std"] for item in values) / len(values),
            "adapter_update_to_o_proj_norm_ratio": sum(
                item["adapter_update_to_o_proj_norm_ratio"] for item in values
            ) / len(values),
            "adapted_to_o_proj_cosine": sum(item["adapted_to_o_proj_cosine"] for item in values) / len(values),
            "supporting_fact_attention_mass": sum(
                item["supporting_fact_attention_mass"] for item in values
            ) / len(values),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-reader", required=True)
    parser.add_argument("--strong-reader", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = CanonicalCache(args.memory_index, args.data, capacity=2)
    if len(cache) != 64:
        raise RuntimeError("Expected fixed validation64 cache")
    negatives = ([int(entry["hard_negative_index"]) for entry in cache.entries]
                 if all("hard_negative_index" in entry for entry in cache.entries)
                 else hard_negative_mapping(cache))
    model, tokenizer = load_receiver(args.model, device)
    base_checkpoint = torch.load(args.base_reader, map_location="cpu", weights_only=False)
    old_reader = load_old_reader(model, base_checkpoint)
    strong_checkpoint = torch.load(args.strong_reader, map_location="cpu", weights_only=False)
    strong_reader = StrongCanonicalReader(
        model, base_checkpoint, int(strong_checkpoint["args"]["output_rank"])
    ).to(device)
    strong_reader.load_state_dict(strong_checkpoint["reader"])
    strong_reader.requires_grad_(False)
    strong_reader.eval()
    conditions = ["question_only", "old_current_reader", "strong_reader_v1",
                  "hard_shuffled_strong_reader", "oracle_support_strong_reader", "reader_off"]
    records, pairs, diagnostic_rows = [], [], []
    for index in tqdm(range(64), desc="p3e_g_eval64"):
        payload, wrong = cache.load(index), cache.load(negatives[index])
        row = payload["row"]
        predictions = {}
        for condition in conditions:
            trace = {} if condition == "strong_reader_v1" else None
            if condition == "question_only":
                result = plain_generate(model, tokenizer, row, args.max_new_tokens)
            elif condition == "old_current_reader":
                result = generate(model, tokenizer, old_reader, row, memory_to(payload, device), args.max_new_tokens)
            elif condition == "strong_reader_v1":
                result = generate(model, tokenizer, strong_reader, row, memory_to(payload, device),
                                  args.max_new_tokens, trace=trace)
            elif condition == "hard_shuffled_strong_reader":
                result = generate(model, tokenizer, strong_reader, row, memory_to(wrong, device), args.max_new_tokens)
            elif condition == "oracle_support_strong_reader":
                result = generate(model, tokenizer, strong_reader, row, memory_to(payload, device, True), args.max_new_tokens)
            else:
                result = generate(model, tokenizer, strong_reader, row, memory_to(payload, device),
                                  args.max_new_tokens, enabled=False)
            em, f1 = answer_scores(result["prediction"], row["answer"])
            item = {"id": row["id"], "type": row["type"], "answer": row["answer"],
                    "condition": condition, "em": em, "f1": f1, "output": result}
            if trace is not None:
                diagnostics = trace_diagnostics(trace, payload["support_mask"])
                item["strong_reader_diagnostics"] = diagnostics
                diagnostic_rows.append({"id": row["id"], "layers": diagnostics})
            if condition == "hard_shuffled_strong_reader":
                item.update({"source_id": wrong["row"]["id"], "source_answer": wrong["row"]["answer"]})
            records.append(item)
            predictions[condition] = result["prediction"]
        pairs.append({
            "id": row["id"],
            "prediction_switch": float(
                normalize_answer(predictions["strong_reader_v1"]) !=
                normalize_answer(predictions["hard_shuffled_strong_reader"])
            ),
            "reader_off_exact": float(predictions["question_only"] == predictions["reader_off"]),
        })
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_jsonl(output / "strong_reader_diagnostics.jsonl", diagnostic_rows)
    metrics = {condition: summarize(records, condition) for condition in conditions}
    summary = {
        "status": "complete", "experiment": "P3-E-G Strong Reader V1",
        "samples": 64, "conditions": metrics,
        "strong_correct_shuffled_f1_gap": (
            metrics["strong_reader_v1"]["f1"] - metrics["hard_shuffled_strong_reader"]["f1"]
        ),
        "strong_minus_old_f1": (
            metrics["strong_reader_v1"]["f1"] - metrics["old_current_reader"]["f1"]
        ),
        "prediction_switch_rate": sum(row["prediction_switch"] for row in pairs) / 64,
        "reader_off_exact_output_consistency": sum(row["reader_off_exact"] for row in pairs) / 64,
        "layer_diagnostics": aggregate_diagnostics(diagnostic_rows),
        "manual_semantic_evaluation": "pending_blinded_CPW_review",
        "strong_reader": args.strong_reader, "base_reader": args.base_reader,
    }
    write_json(output / "SUCCESS.json", summary)


if __name__ == "__main__":
    main()
