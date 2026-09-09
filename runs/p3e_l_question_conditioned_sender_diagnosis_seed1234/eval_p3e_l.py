import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import (
    answer_scores,
    apply_chat,
    evidence_block,
    extract_prediction,
    generate,
    load_receiver,
    normalize_answer,
    question_prompt,
    write_json,
    write_jsonl,
)
from p3e_c1_common import LearnableCanonicalHeadReader
from p3e_c2_common import SenderNativeHeadwiseCache, load_writer, writer_memory
from p3e_l_common import ConditionedNativeCache, condition_payload


CONDITIONS = (
    "question_only",
    "evidence_only_canonical",
    "neutral_prefix_canonical",
    "wrong_question_canonical",
    "correct_question_canonical",
    "correct_question_hard_shuffled_evidence",
    "full_evidence_text",
)


def full_text_prompt(tokenizer, row):
    system = "Answer the question with a short answer. End with exactly FINAL: <answer>."
    user = f"{evidence_block(row)}\n\nQUESTION\n{row['question']}"
    return apply_chat(tokenizer, system, user) + "FINAL:"


@torch.inference_mode()
def plain_generate(model, tokenizer, prompt, max_new_tokens):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    tokens = output[0, encoded["input_ids"].shape[1] :].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    prediction, method = extract_prediction(text)
    return {
        "text": text,
        "prediction": prediction,
        "parse_method": method,
        "token_ids": tokens,
        "eos_reached": tokenizer.eos_token_id in tokens,
    }


def summarize(rows, condition):
    subset = [row for row in rows if row["condition"] == condition]
    result = {
        "n": len(subset),
        "em": sum(row["em"] for row in subset) / len(subset),
        "f1": sum(row["f1"] for row in subset) / len(subset),
        "eos_rate": sum(row["eos_reached"] for row in subset) / len(subset),
        "average_output_tokens": sum(row["output_tokens"] for row in subset) / len(subset),
    }
    for kind in ("bridge", "comparison"):
        typed = [row for row in subset if row["type"] == kind]
        if typed:
            result[kind] = {
                "n": len(typed),
                "em": sum(row["em"] for row in typed) / len(typed),
                "f1": sum(row["f1"] for row in typed) / len(typed),
            }
    return result


def load_reader(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["reader_metadata"]
    reader = LearnableCanonicalHeadReader(
        model,
        metadata["selected_layers"],
        metadata["rank"],
        metadata["gate_init"],
        metadata["top_k"],
        0.25,
    ).to(device)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    reader.eval()
    return reader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-memory", required=True)
    parser.add_argument("--conditioned-memory", required=True)
    parser.add_argument("--baseline-writer", required=True)
    parser.add_argument("--conditioned-writer", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    base = SenderNativeHeadwiseCache(args.base_memory)
    conditioned = ConditionedNativeCache(args.conditioned_memory)
    count = min(args.max_samples, len(base), len(conditioned))
    model, tokenizer = load_receiver(args.model, device)
    reader = load_reader(model, args.reader, device)
    baseline_writer, _ = load_writer(args.baseline_writer, device)
    conditioned_writer, _ = load_writer(args.conditioned_writer, device)
    baseline_writer.requires_grad_(False).eval()
    conditioned_writer.requires_grad_(False).eval()

    rows = []
    pair_rows = []
    for index in tqdm(range(count), desc="p3e_l_free_running"):
        base_payload = base.load(index)
        bundle = conditioned.load(index)
        row = base_payload["row"]
        predictions = {}
        for condition in CONDITIONS:
            if condition == "question_only":
                result = plain_generate(model, tokenizer, question_prompt(tokenizer, row), args.max_new_tokens)
                source_id, source_answer = row["id"], row["answer"]
            elif condition == "full_evidence_text":
                result = plain_generate(model, tokenizer, full_text_prompt(tokenizer, row), args.max_new_tokens)
                source_id, source_answer = row["id"], row["answer"]
            elif condition == "evidence_only_canonical":
                memory = writer_memory(baseline_writer, base_payload, device, no_grad=True)
                result = generate(model, tokenizer, reader, row, memory, args.max_new_tokens)
                source_id, source_answer = row["id"], row["answer"]
            else:
                native_condition = condition.replace("_canonical", "")
                payload = condition_payload(bundle, native_condition)
                memory = writer_memory(conditioned_writer, payload, device, no_grad=True)
                result = generate(model, tokenizer, reader, row, memory, args.max_new_tokens)
                source_id, source_answer = payload["source_id"], payload["source_answer"]
            em, f1 = answer_scores(result["prediction"], row["answer"])
            source_em, source_f1 = answer_scores(result["prediction"], source_answer)
            predictions[condition] = result["prediction"]
            rows.append(
                {
                    "id": row["id"],
                    "type": row.get("type"),
                    "condition": condition,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "source_id": source_id,
                    "source_answer": source_answer,
                    "prediction": result["prediction"],
                    "generation": result["text"],
                    "parse_method": result["parse_method"],
                    "token_ids": result["token_ids"],
                    "output_tokens": len(result["token_ids"]),
                    "eos_reached": result["eos_reached"],
                    "em": em,
                    "f1": f1,
                    "source_em": source_em,
                    "source_f1": source_f1,
                }
            )
        pair_rows.append(
            {
                "id": row["id"],
                "question_conditioning_switch": float(
                    normalize_answer(predictions["correct_question_canonical"])
                    != normalize_answer(predictions["evidence_only_canonical"])
                ),
                "correct_vs_wrong_question_switch": float(
                    normalize_answer(predictions["correct_question_canonical"])
                    != normalize_answer(predictions["wrong_question_canonical"])
                ),
            }
        )

    metrics = {condition: summarize(rows, condition) for condition in CONDITIONS}
    delta_q = (
        metrics["correct_question_canonical"]["f1"]
        - metrics["evidence_only_canonical"]["f1"]
    )
    write_jsonl(output / "per_example.jsonl", rows)
    write_jsonl(output / "pair_controls.jsonl", pair_rows)
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-L Question-Conditioned Sender Diagnosis",
            "samples": count,
            "metrics": metrics,
            "delta_q_automatic_f1": delta_q,
            "prediction_switch": {
                key: sum(row[key] for row in pair_rows) / len(pair_rows)
                for key in pair_rows[0]
                if key != "id"
            },
            "manual_cpw_required": True,
            "baseline_writer": args.baseline_writer,
            "conditioned_writer": args.conditioned_writer,
            "reader": args.reader,
        },
    )
    with (output / "manual_cpw_blind.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "condition", "question", "gold_answer", "generation", "C_P_W", "strict_correct", "lenient_correct"])
        for row in rows:
            writer.writerow([row["id"], row["condition"], row["question"], row["gold_answer"], row["generation"], "", "", ""])


if __name__ == "__main__":
    main()
