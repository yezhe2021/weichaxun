import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import (answer_scores, extract_prediction, generate, hard_negative_mapping,
                         load_receiver, normalize_answer, question_prompt, seed_everything)
from p3e_f_common import CanonicalCache, memory_to, write_json, write_jsonl
from p3e_h_common import load_c1_reader
from p3e_i_common import (install_lora, load_lora_state, lora_diagnostics,
                          lora_enabled)


@torch.inference_mode()
def plain_generate(model, tokenizer, modules, row, max_new_tokens):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    with lora_enabled(modules, False):
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    prediction, method = extract_prediction(text)
    return {"text": text, "prediction": prediction, "parse_method": method,
            "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-reader", required=True)
    parser.add_argument("--lora-reader", required=True)
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
    reader = load_c1_reader(model, c1_checkpoint)
    checkpoint = torch.load(args.lora_reader, map_location="cpu", weights_only=False)
    metadata = checkpoint["lora_metadata"]
    modules = install_lora(
        model, metadata["receiver_layers"], metadata["rank"], metadata["alpha"],
        metadata["dropout"]
    )
    load_lora_state(modules, checkpoint["lora_state"])
    conditions = ["question_only", "current_c1_reader", "c1_reader_plus_lora_qa_only",
                  "hard_shuffled_lora", "oracle_support_lora", "reader_off"]
    records, pairs = [], []
    for index in tqdm(range(64), desc="p3e_i_eval64"):
        payload, wrong = cache.load(index), cache.load(negatives[index])
        row = payload["row"]
        predictions, tokens = {}, {}
        for condition in conditions:
            if condition == "question_only":
                result = plain_generate(model, tokenizer, modules, row, args.max_new_tokens)
            elif condition == "current_c1_reader":
                with lora_enabled(modules, False):
                    result = generate(model, tokenizer, reader, row, memory_to(payload, device),
                                      args.max_new_tokens)
            elif condition == "c1_reader_plus_lora_qa_only":
                with lora_enabled(modules, True):
                    result = generate(model, tokenizer, reader, row, memory_to(payload, device),
                                      args.max_new_tokens)
            elif condition == "hard_shuffled_lora":
                with lora_enabled(modules, True):
                    result = generate(model, tokenizer, reader, row, memory_to(wrong, device),
                                      args.max_new_tokens)
            elif condition == "oracle_support_lora":
                with lora_enabled(modules, True):
                    result = generate(model, tokenizer, reader, row,
                                      memory_to(payload, device, True), args.max_new_tokens)
            else:
                with lora_enabled(modules, False):
                    result = generate(model, tokenizer, reader, row, memory_to(payload, device),
                                      args.max_new_tokens, enabled=False)
            em, f1 = answer_scores(result["prediction"], row["answer"])
            item = {"id": row["id"], "type": row["type"], "answer": row["answer"],
                    "condition": condition, "em": em, "f1": f1, "output": result}
            if condition == "hard_shuffled_lora":
                item.update({"source_id": wrong["row"]["id"], "source_answer": wrong["row"]["answer"]})
            records.append(item)
            predictions[condition], tokens[condition] = result["prediction"], result["token_ids"]
        pairs.append({
            "id": row["id"],
            "prediction_switch": float(
                normalize_answer(predictions["c1_reader_plus_lora_qa_only"]) !=
                normalize_answer(predictions["hard_shuffled_lora"])
            ),
            "reader_off_exact": float(tokens["question_only"] == tokens["reader_off"]),
        })
    write_jsonl(output / "per_sample_generation.jsonl", records)
    metrics = {condition: summarize(records, condition) for condition in conditions}
    write_json(output / "SUCCESS.json", {
        "status": "complete", "experiment": "P3-E-I Adapter-Augmented Reader QA-only",
        "samples": 64, "conditions": metrics,
        "lora_minus_c1_f1": (
            metrics["c1_reader_plus_lora_qa_only"]["f1"] -
            metrics["current_c1_reader"]["f1"]
        ),
        "correct_shuffled_f1_gap": (
            metrics["c1_reader_plus_lora_qa_only"]["f1"] -
            metrics["hard_shuffled_lora"]["f1"]
        ),
        "prediction_switch_rate": sum(row["prediction_switch"] for row in pairs) / 64,
        "reader_off_exact_output_consistency": sum(row["reader_off_exact"] for row in pairs) / 64,
        "lora_metadata": metadata, "lora_diagnostics": lora_diagnostics(modules),
        "manual_semantic_evaluation": "pending_blinded_CPW_review",
        "evidence_reconstruction_stage": "not_run_pending_QA_only_result",
        "base_reader": args.base_reader, "lora_reader": args.lora_reader,
    })


if __name__ == "__main__":
    main()
