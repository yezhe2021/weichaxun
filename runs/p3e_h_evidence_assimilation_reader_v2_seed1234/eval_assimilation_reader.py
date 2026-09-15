import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import (answer_scores, extract_prediction, generate, hard_negative_mapping,
                         load_receiver, normalize_answer, question_prompt, seed_everything)
from p3e_f_common import CanonicalCache, memory_to, write_json, write_jsonl
from p3e_h_common import EvidenceAssimilationReader, load_c1_reader


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


def rms(value):
    return value.detach().float().pow(2).mean().sqrt()


def trace_diagnostics(trace):
    result = {}
    for layer, calls in trace.items():
        values = []
        for call in calls:
            item = {"old_gate": float(call["old_gate"])}
            if call["assimilation"] is not None:
                evidence = call["evidence"].detach().float()
                assimilation = call["assimilation"].detach().float()
                correction = call["correction"].detach().float()
                old_delta = call["old_delta"].detach().float()
                item.update({
                    "beta": float(call["beta"]),
                    "assimilation_to_evidence_rms_ratio": float(rms(assimilation) / rms(evidence).clamp_min(1e-8)),
                    "correction_to_old_external_rms_ratio": float(rms(correction) / rms(old_delta).clamp_min(1e-8)),
                    "assimilation_evidence_cosine": float(F.cosine_similarity(
                        assimilation.reshape(-1, assimilation.shape[-1]),
                        evidence.reshape(-1, evidence.shape[-1]), dim=-1
                    ).mean()),
                })
            values.append(item)
        keys = values[0].keys()
        result[str(layer)] = {
            key: sum(item[key] for item in values) / len(values) for key in keys
        }
    return result


def summarize(records, condition):
    selected = [row for row in records if row["condition"] == condition]
    result = {
        "n": len(selected), "em": sum(row["em"] for row in selected) / len(selected),
        "f1": sum(row["f1"] for row in selected) / len(selected),
        "eos_rate": sum(row["output"]["eos_reached"] for row in selected) / len(selected),
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
    layers = {}
    for row in rows:
        for layer, values in row["layers"].items():
            layers.setdefault(layer, []).append(values)
    return {
        layer: {key: sum(item[key] for item in values) / len(values)
                for key in values[0]}
        for layer, values in layers.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-reader", required=True)
    parser.add_argument("--assimilation-reader", required=True)
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
        raise RuntimeError("Expected fixed validation64")
    negatives = ([int(entry["hard_negative_index"]) for entry in cache.entries]
                 if all("hard_negative_index" in entry for entry in cache.entries)
                 else hard_negative_mapping(cache))
    model, tokenizer = load_receiver(args.model, device)
    c1_checkpoint = torch.load(args.base_reader, map_location="cpu", weights_only=False)
    current_reader = load_c1_reader(model, c1_checkpoint)
    checkpoint = torch.load(args.assimilation_reader, map_location="cpu", weights_only=False)
    reader = EvidenceAssimilationReader(
        model, c1_checkpoint, int(checkpoint["args"]["bottleneck"]),
        float(checkpoint["args"]["beta_init"])
    ).to(device)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    reader.eval()
    conditions = ["question_only", "current_c1_reader", "assimilation_reader_v2",
                  "hard_shuffled_assimilation", "oracle_support_assimilation", "reader_off"]
    records, pairs, diagnostics = [], [], []
    for index in tqdm(range(64), desc="p3e_h_eval64"):
        payload, wrong = cache.load(index), cache.load(negatives[index])
        row = payload["row"]
        predictions = {}
        for condition in conditions:
            trace = {} if condition == "assimilation_reader_v2" else None
            if condition == "question_only":
                result = plain_generate(model, tokenizer, row, args.max_new_tokens)
            elif condition == "current_c1_reader":
                result = generate(model, tokenizer, current_reader, row,
                                  memory_to(payload, device), args.max_new_tokens)
            elif condition == "assimilation_reader_v2":
                result = generate(model, tokenizer, reader, row, memory_to(payload, device),
                                  args.max_new_tokens, trace=trace)
            elif condition == "hard_shuffled_assimilation":
                result = generate(model, tokenizer, reader, row, memory_to(wrong, device),
                                  args.max_new_tokens)
            elif condition == "oracle_support_assimilation":
                result = generate(model, tokenizer, reader, row, memory_to(payload, device, True),
                                  args.max_new_tokens)
            else:
                result = generate(model, tokenizer, reader, row, memory_to(payload, device),
                                  args.max_new_tokens, enabled=False)
            em, f1 = answer_scores(result["prediction"], row["answer"])
            item = {"id": row["id"], "type": row["type"], "answer": row["answer"],
                    "condition": condition, "em": em, "f1": f1, "output": result}
            if trace is not None:
                layer_values = trace_diagnostics(trace)
                item["assimilation_diagnostics"] = layer_values
                diagnostics.append({"id": row["id"], "layers": layer_values})
            if condition == "hard_shuffled_assimilation":
                item.update({"source_id": wrong["row"]["id"], "source_answer": wrong["row"]["answer"]})
            records.append(item)
            predictions[condition] = result["prediction"]
        pairs.append({
            "id": row["id"],
            "prediction_switch": float(
                normalize_answer(predictions["assimilation_reader_v2"]) !=
                normalize_answer(predictions["hard_shuffled_assimilation"])
            ),
            "reader_off_exact": float(predictions["question_only"] == predictions["reader_off"]),
        })
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_jsonl(output / "assimilation_diagnostics.jsonl", diagnostics)
    metrics = {condition: summarize(records, condition) for condition in conditions}
    write_json(output / "SUCCESS.json", {
        "status": "complete", "experiment": "P3-E-H Evidence Assimilation Reader V2",
        "samples": 64, "conditions": metrics,
        "assimilation_minus_c1_f1": (
            metrics["assimilation_reader_v2"]["f1"] - metrics["current_c1_reader"]["f1"]
        ),
        "correct_shuffled_f1_gap": (
            metrics["assimilation_reader_v2"]["f1"] -
            metrics["hard_shuffled_assimilation"]["f1"]
        ),
        "prediction_switch_rate": sum(row["prediction_switch"] for row in pairs) / 64,
        "reader_off_exact_output_consistency": sum(row["reader_off_exact"] for row in pairs) / 64,
        "old_scalar_gates": reader.old_gates().cpu().tolist(),
        "assimilation_betas": reader.betas().detach().cpu().tolist(),
        "layer_diagnostics": aggregate_diagnostics(diagnostics),
        "manual_semantic_evaluation": "pending_blinded_CPW_review",
        "base_reader": args.base_reader, "assimilation_reader": args.assimilation_reader,
    })


if __name__ == "__main__":
    main()
