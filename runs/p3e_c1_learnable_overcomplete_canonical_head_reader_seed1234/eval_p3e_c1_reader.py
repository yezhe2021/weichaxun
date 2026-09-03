import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import (answer_scores, apply_chat, evidence_block, extract_prediction, generate, hard_negative_mapping,
                         load_receiver, normalize_answer, question_prompt, seed_everything, write_json, write_jsonl)
from p3e_c1_common import DuplicateHeadwiseCache, LearnableCanonicalHeadReader, duplicate_memory_to


def full_text_prompt(tokenizer, row):
    system = "Answer the question using the supplied gold evidence. Give a short answer. End with exactly FINAL: <answer>."
    return apply_chat(tokenizer, system, f"QUESTION\n{row['question']}\n\nGOLD SUPPORTING EVIDENCE\n{evidence_block(row)}") + "FINAL:"


@torch.inference_mode()
def plain_generate(model, tokenizer, prompt, max_new_tokens):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False); encoded = {name: value.to(model.device) for name, value in encoded.items()}
    output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist(); text = tokenizer.decode(tokens, skip_special_tokens=True); prediction, method = extract_prediction(text)
    return {"text": text, "prediction": prediction, "parse_method": method, "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens}


def trace_diagnostics(trace, support_mask):
    support = support_mask.float(); result = {}
    for layer, calls in trace.items():
        support_values, route_values = [], []
        for call in calls:
            attention = call["attention"].detach().float().cpu()
            support_values.append((attention * support[None, None, None, None, :]).sum(-1).mean(dim=(0, 1, 2)))
            route_values.append(call["route"].detach().float().cpu())
        route = torch.stack(route_values).mean(0)
        result[str(layer)] = {"canonical_head_support_attention_mass": torch.stack(support_values).mean(0).tolist(),
                              "route": route.tolist(), "selected_top2": route.topk(2, dim=-1).indices.tolist(),
                              "canonical_head_usage": route.mean(0).tolist()}
    return result


def summarize(records, condition):
    selected = [row for row in records if row["condition"] == condition]
    result = {"n": len(selected), "em": sum(row["em"] for row in selected) / len(selected), "f1": sum(row["f1"] for row in selected) / len(selected),
              "eos_rate": sum(row["output"]["eos_reached"] for row in selected) / len(selected), "by_type": {}}
    for kind in ("bridge", "comparison"):
        group = [row for row in selected if row["type"] == kind]
        if group: result["by_type"][kind] = {"n": len(group), "em": sum(row["em"] for row in group) / len(group), "f1": sum(row["f1"] for row in group) / len(group)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int); parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); output = Path(args.out); output.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    cache = DuplicateHeadwiseCache(args.memory); count = min(args.max_samples or len(cache), len(cache)); negatives = hard_negative_mapping(cache)
    model, tokenizer = load_receiver(args.model, device); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); metadata = checkpoint["reader_metadata"]
    reader = LearnableCanonicalHeadReader(model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"], metadata["top_k"], 0.25).to(device)
    reader.load_state_dict(checkpoint["reader"]); reader.eval()
    conditions = ["question_only", "gold_full_text", "reader_off", "correct_learnable_duplicate16", "hard_shuffled_learnable_duplicate16", "oracle_support_learnable_duplicate16"]
    records, pairs, trace_rows = [], [], []
    for index in tqdm(range(count), desc="p3e_c1_free_running"):
        payload, wrong = cache.load(index), cache.load(negatives[index]); row = payload["row"]; predictions = {}
        for condition in conditions:
            trace = {} if condition == "correct_learnable_duplicate16" else None
            if condition == "question_only": result = plain_generate(model, tokenizer, question_prompt(tokenizer, row), args.max_new_tokens)
            elif condition == "gold_full_text": result = plain_generate(model, tokenizer, full_text_prompt(tokenizer, row), args.max_new_tokens)
            elif condition == "reader_off": result = generate(model, tokenizer, reader, row, duplicate_memory_to(payload, device), args.max_new_tokens, enabled=False)
            elif condition == "correct_learnable_duplicate16": result = generate(model, tokenizer, reader, row, duplicate_memory_to(payload, device), args.max_new_tokens, trace=trace)
            elif condition == "hard_shuffled_learnable_duplicate16": result = generate(model, tokenizer, reader, row, duplicate_memory_to(wrong, device), args.max_new_tokens)
            else: result = generate(model, tokenizer, reader, row, duplicate_memory_to(payload, device, oracle_support=True), args.max_new_tokens)
            em, f1 = answer_scores(result["prediction"], row["answer"]); predictions[condition] = result["prediction"]
            item = {"id": row["id"], "type": row["type"], "answer": row["answer"], "condition": condition, "em": em, "f1": f1, "output": result}
            if trace is not None:
                item["routing_and_support"] = trace_diagnostics(trace, torch.as_tensor(payload["metadata"]["support_token_mask"])); trace_rows.append({"id": row["id"], "layers": item["routing_and_support"]})
            if condition == "hard_shuffled_learnable_duplicate16": item.update({"source_id": wrong["row"]["id"], "source_answer": wrong["row"]["answer"]})
            records.append(item)
        pairs.append({"id": row["id"], "correct_shuffled_switch": float(normalize_answer(predictions["correct_learnable_duplicate16"]) != normalize_answer(predictions["hard_shuffled_learnable_duplicate16"])),
                      "question_only_equals_reader_off": float(predictions["question_only"] == predictions["reader_off"])})
    write_jsonl(output / "per_sample_generation.jsonl", records); write_jsonl(output / "routing_and_support.jsonl", trace_rows)
    metrics = {condition: summarize(records, condition) for condition in conditions}; correct = metrics["correct_learnable_duplicate16"]["f1"]
    shuffled, question = metrics["hard_shuffled_learnable_duplicate16"]["f1"], metrics["question_only"]["f1"]
    routes = reader.routes().detach().cpu()
    write_json(output / "SUCCESS.json", {"status": "complete", "experiment": "P3-E-C1 learnable overcomplete Canonical Head Reader", "samples": count,
        "conditions": metrics, "correct_shuffled_f1_gap": correct - shuffled, "correct_question_only_f1_gain": correct - question,
        "prediction_switch_rate": sum(row["correct_shuffled_switch"] for row in pairs) / len(pairs),
        "reader_off_exact_output_consistency": sum(row["question_only_equals_reader_off"] for row in pairs) / len(pairs),
        "gates": reader.gates().detach().cpu().tolist(), "routes": routes.tolist(),
        "selected_top2": routes.topk(2, dim=-1).indices.tolist(), "canonical_head_usage": routes.mean(dim=1).tolist(),
        "path": {"writer": "fixed duplicate_writer16", "memory": "[16,T,16,128]", "query": "native pre-RoPE [B,S,32,128] plus rank-32 residual",
                 "routing": "per-layer 32x16 hard top-2", "output": "[B,S,4096] through frozen native o_proj", "writer_trainable_parameters": 0},
        "checkpoint": args.checkpoint, "data": args.memory})


if __name__ == "__main__": main()
