import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import (answer_scores, apply_chat, evidence_block, extract_prediction, hard_negative_mapping,
                         normalize_answer, question_prompt, seed_everything, write_json, write_jsonl)
from p3e_c2_common import SenderNativeHeadwiseCache, load_writer, writer_memory
from p3e_c4_common import Qwen35CanonicalReader, load_qwen35


def full_text_prompt(tokenizer, row):
    system = "Answer the question using the supplied gold evidence. Give a short answer. End with exactly FINAL: <answer>."
    return apply_chat(tokenizer, system, f"QUESTION\n{row['question']}\n\nGOLD SUPPORTING EVIDENCE\n{evidence_block(row)}") + "FINAL:"


@torch.inference_mode()
def plain_generate(model, tokenizer, prompt, max_new_tokens):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False); encoded = {name: value.to(model.device) for name, value in encoded.items()}
    output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist(); text = tokenizer.decode(tokens, skip_special_tokens=True); prediction, method = extract_prediction(text)
    return {"text": text, "prediction": prediction, "parse_method": method, "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens}


@torch.inference_mode()
def reader_generate(model, tokenizer, reader, row, memory, max_new_tokens, enabled=True, trace=None):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False); encoded = {name: value.to(model.device) for name, value in encoded.items()}
    kwargs = dict(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if enabled:
        with reader.inject(model, memory, trace): output = model.generate(**kwargs)
    else: output = model.generate(**kwargs)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist(); text = tokenizer.decode(tokens, skip_special_tokens=True); prediction, method = extract_prediction(text)
    return {"text": text, "prediction": prediction, "parse_method": method, "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens}


def summarize(records, condition):
    selected = [row for row in records if row["condition"] == condition]
    result = {"n": len(selected), "em": sum(row["em"] for row in selected) / len(selected), "f1": sum(row["f1"] for row in selected) / len(selected),
              "eos_rate": sum(row["output"]["eos_reached"] for row in selected) / len(selected), "by_type": {}}
    for kind in ("bridge", "comparison"):
        group = [row for row in selected if row["type"] == kind]
        if group: result["by_type"][kind] = {"n": len(group), "em": sum(row["em"] for row in group) / len(group), "f1": sum(row["f1"] for row in group) / len(group)}
    return result


def trace_summary(trace, support_mask):
    support = support_mask.float(); result = {}
    for layer, calls in trace.items():
        masses = []
        for call in calls:
            attention = call["token_attention"].detach().float().cpu()
            masses.append((attention * support[None, None, None, None, None, :]).sum(-1).mean(dim=(0, 1, 2)))
        first = calls[0]
        result[str(layer)] = {"group_canonical_head_support_mass": torch.stack(masses).mean(0).tolist(),
                              "head_route": first["head_route"].detach().float().cpu().tolist(),
                              "group_route": first["group_route"].detach().float().cpu().tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--writer", required=True); parser.add_argument("--reader", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--max-samples", type=int, default=64); parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    cache = SenderNativeHeadwiseCache(args.memory); count = min(args.max_samples, len(cache)); negatives = hard_negative_mapping(cache)
    model, tokenizer = load_qwen35(args.model, device); writer, _ = load_writer(args.writer, device); writer.requires_grad_(False); writer.eval()
    checkpoint = torch.load(args.reader, map_location="cpu", weights_only=False); metadata = checkpoint["reader_metadata"]
    reader = Qwen35CanonicalReader(model, metadata["rank"], metadata["gate_init"], metadata["top_k"], metadata["seed"]).to(device); reader.load_state_dict(checkpoint["reader"]); reader.requires_grad_(False); reader.eval()
    conditions = ("question_only", "gold_full_text", "reader_off", "correct_canonical16", "hard_shuffled_canonical16", "oracle_support_canonical16")
    records, traces, pairs = [], [], []
    for index in tqdm(range(count), desc=f"c4_qwen35_eval_seed{args.seed}"):
        payload, wrong = cache.load(index), cache.load(negatives[index]); row = payload["row"]; predictions = {}
        for condition in conditions:
            trace = {} if condition == "correct_canonical16" else None
            if condition == "question_only": result = plain_generate(model, tokenizer, question_prompt(tokenizer, row), args.max_new_tokens)
            elif condition == "gold_full_text": result = plain_generate(model, tokenizer, full_text_prompt(tokenizer, row), args.max_new_tokens)
            elif condition == "reader_off": result = reader_generate(model, tokenizer, reader, row, writer_memory(writer, payload, device, no_grad=True), args.max_new_tokens, enabled=False)
            elif condition == "correct_canonical16": result = reader_generate(model, tokenizer, reader, row, writer_memory(writer, payload, device, no_grad=True), args.max_new_tokens, trace=trace)
            elif condition == "hard_shuffled_canonical16": result = reader_generate(model, tokenizer, reader, row, writer_memory(writer, wrong, device, no_grad=True), args.max_new_tokens)
            else: result = reader_generate(model, tokenizer, reader, row, writer_memory(writer, payload, device, oracle_support=True, no_grad=True), args.max_new_tokens)
            em, f1 = answer_scores(result["prediction"], row["answer"]); predictions[condition] = result["prediction"]
            item = {"id": row["id"], "type": row["type"], "answer": row["answer"], "condition": condition, "em": em, "f1": f1, "output": result}
            if trace is not None: traces.append({"id": row["id"], "layers": trace_summary(trace, torch.as_tensor(payload["metadata"]["support_token_mask"]))})
            if condition == "hard_shuffled_canonical16": item.update({"source_id": wrong["row"]["id"], "source_answer": wrong["row"]["answer"]})
            records.append(item)
        pairs.append({"id": row["id"], "correct_shuffled_switch": float(normalize_answer(predictions["correct_canonical16"]) != normalize_answer(predictions["hard_shuffled_canonical16"])),
                      "reader_off_matches_question_only": float(predictions["reader_off"] == predictions["question_only"])})
    write_jsonl(output / "per_sample_generation.jsonl", records); write_jsonl(output / "routing_and_support.jsonl", traces)
    metrics = {condition: summarize(records, condition) for condition in conditions}; correct, shuffled, question = metrics["correct_canonical16"]["f1"], metrics["hard_shuffled_canonical16"]["f1"], metrics["question_only"]["f1"]
    write_json(output / "SUCCESS.json", {"status": "complete", "experiment": "C4 same frozen C2 Writer to Qwen3.5-4B", "seed": args.seed, "samples": count,
        "conditions": metrics, "correct_shuffled_f1_gap": correct - shuffled, "correct_question_only_f1_gain": correct - question,
        "prediction_switch_rate": sum(row["correct_shuffled_switch"] for row in pairs) / len(pairs), "reader_off_consistency": sum(row["reader_off_matches_question_only"] for row in pairs) / len(pairs),
        "reader_metadata": metadata, "writer": args.writer, "reader": args.reader, "writer_frozen": True})


if __name__ == "__main__": main()
