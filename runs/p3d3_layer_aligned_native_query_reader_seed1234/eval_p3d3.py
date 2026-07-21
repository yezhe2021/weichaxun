import argparse
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import (LayerAlignedNativeQueryReader, MemoryCache, answer_scores, generate, hard_negative_mapping,
                         load_receiver, memory_to, normalize_answer, read_json, seed_everything, write_json, write_jsonl)


def load_reader(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); metadata = checkpoint["reader_metadata"]
    reader = LayerAlignedNativeQueryReader(model, metadata["memory_dim"], metadata["selected_layers"], metadata["rank"], metadata["gate_init"]).to(device)
    reader.load_state_dict(checkpoint["reader"]); reader.eval(); return reader, checkpoint


def trace_support_mass(trace, support_mask):
    masses = []; support = support_mask.float()
    for calls in trace.values():
        for call in calls:
            attention = call["attention"].detach().float().cpu()
            masses.append(float((attention * support[None, None, :]).sum(-1).mean()))
    return sum(masses) / len(masses) if masses else None


def summarize(records, condition):
    selected = [row for row in records if row["condition"] == condition]
    result = {"n": len(selected), "em": sum(row["em"] for row in selected) / len(selected), "f1": sum(row["f1"] for row in selected) / len(selected),
              "eos_rate": sum(row["output"]["eos_reached"] for row in selected) / len(selected)}
    result["by_type"] = {kind: {"n": len(group), "em": sum(row["em"] for row in group) / len(group), "f1": sum(row["f1"] for row in group) / len(group)}
                         for kind in ("bridge", "comparison") if (group := [row for row in selected if row["type"] == kind])}
    masses = [row["support_attention_mass"] for row in selected if row.get("support_attention_mass") is not None]
    if masses: result["support_attention_mass"] = sum(masses) / len(masses)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--native-memory", required=True); parser.add_argument("--canonical-memory", required=True)
    parser.add_argument("--native-checkpoint", required=True); parser.add_argument("--canonical-checkpoint", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    native, canonical = MemoryCache(args.native_memory), MemoryCache(args.canonical_memory)
    if [entry["id"] for entry in native.entries] != [entry["id"] for entry in canonical.entries]: raise RuntimeError("Native/Canonical validation samples differ")
    model, tokenizer = load_receiver(args.model, device); native_reader, native_ckpt = load_reader(model, args.native_checkpoint, device); canonical_reader, canonical_ckpt = load_reader(model, args.canonical_checkpoint, device)
    negatives = hard_negative_mapping(canonical); records, paired = [], []
    conditions = ["question_only", "reader_off", "correct_native_projected16", "correct_canonical16", "hard_shuffled_canonical16", "oracle_support_canonical16"]
    for index in tqdm(range(len(canonical)), desc="p3d3_free_running_validation"):
        native_payload, canonical_payload, wrong = native.load(index), canonical.load(index), canonical.load(negatives[index]); row = canonical_payload["row"]; outputs = {}
        for condition in conditions:
            trace = {} if condition == "correct_canonical16" else None
            if condition == "question_only": result = generate(model, tokenizer, canonical_reader, row, None, args.max_new_tokens, enabled=False)
            elif condition == "reader_off": result = generate(model, tokenizer, canonical_reader, row, memory_to(canonical_payload, device), args.max_new_tokens, enabled=False)
            elif condition == "correct_native_projected16": result = generate(model, tokenizer, native_reader, row, memory_to(native_payload, device), args.max_new_tokens)
            elif condition == "correct_canonical16": result = generate(model, tokenizer, canonical_reader, row, memory_to(canonical_payload, device), args.max_new_tokens, trace=trace)
            elif condition == "hard_shuffled_canonical16": result = generate(model, tokenizer, canonical_reader, row, memory_to(wrong, device), args.max_new_tokens)
            else: result = generate(model, tokenizer, canonical_reader, row, memory_to(canonical_payload, device, oracle_support=True), args.max_new_tokens)
            em, f1 = answer_scores(result["prediction"], row["answer"]); outputs[condition] = result["prediction"]
            item = {"id": row["id"], "type": row["type"], "answer": row["answer"], "condition": condition, "em": em, "f1": f1, "output": result}
            if trace is not None: item["support_attention_mass"] = trace_support_mass(trace, torch.as_tensor(canonical_payload["metadata"]["support_token_mask"]))
            if condition == "hard_shuffled_canonical16": item.update({"source_id": wrong["row"]["id"], "source_answer": wrong["row"]["answer"], "source_answer_scores": answer_scores(result["prediction"], wrong["row"]["answer"])})
            records.append(item)
        paired.append({"id": row["id"], "canonical_vs_shuffled_prediction_switch": float(normalize_answer(outputs["correct_canonical16"]) != normalize_answer(outputs["hard_shuffled_canonical16"])),
                       "question_only_equals_reader_off": float(outputs["question_only"] == outputs["reader_off"])})
    write_jsonl(output / "per_sample_generation.jsonl", records); metrics = {condition: summarize(records, condition) for condition in conditions}
    q, native_score, canonical_score, shuffled = (metrics[name]["f1"] for name in ("question_only", "correct_native_projected16", "correct_canonical16", "hard_shuffled_canonical16"))
    recovery = (canonical_score - q) / (native_score - q) if abs(native_score - q) > 1e-8 else None
    result = {"status": "complete", "validation_evaluated_once_after_training": True, "conditions": metrics,
              "correct_shuffled_f1_gap": canonical_score - shuffled, "canonical_relative_native_projected_recovery": recovery,
              "prediction_switch_rate": sum(row["canonical_vs_shuffled_prediction_switch"] for row in paired) / len(paired),
              "reader_off_exact_output_consistency": sum(row["question_only_equals_reader_off"] for row in paired) / len(paired),
              "phenomena": {"correct_improves_question_only": canonical_score > q, "wrong_memory_hurts": canonical_score > shuffled,
                            "canonical_close_to_native_projected": canonical_score >= native_score - 0.05},
              "n": len(canonical), "native_checkpoint": args.native_checkpoint, "canonical_checkpoint": args.canonical_checkpoint}
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__": main()
