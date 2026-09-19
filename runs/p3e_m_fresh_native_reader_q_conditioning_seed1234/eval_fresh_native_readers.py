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
from p3e_a_common import NativeHeadwiseReader, native_memory_to
from p3e_m_common import FreshReaderMemory


CONDITIONS = (
    "question_only",
    "reader_off",
    "full_evidence_text",
    "evidence_only_native_reader_a",
    "evidence_only_hard_shuffled_reader_a",
    "question_conditioned_native_reader_b",
    "question_conditioned_hard_shuffled_reader_b",
)


def full_text_prompt(tokenizer, row):
    system = "Answer the question using the supplied gold evidence. Give a short answer. End with exactly FINAL: <answer>."
    return (
        apply_chat(
            tokenizer,
            system,
            f"QUESTION\n{row['question']}\n\nGOLD SUPPORTING EVIDENCE\n{evidence_block(row)}",
        )
        + "FINAL:"
    )


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


def load_reader(model, path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["reader_metadata"]
    reader = NativeHeadwiseReader(
        model,
        metadata["selected_layers"],
        metadata["rank"],
        metadata["gate_init"],
    ).to(device)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    reader.eval()
    return reader, checkpoint


def summarize(records, condition):
    rows = [row for row in records if row["condition"] == condition]
    result = {
        "n": len(rows),
        "em": sum(row["em"] for row in rows) / len(rows),
        "f1": sum(row["f1"] for row in rows) / len(rows),
        "eos_rate": sum(row["eos_reached"] for row in rows) / len(rows),
        "average_output_tokens": sum(row["output_tokens"] for row in rows) / len(rows),
        "by_type": {},
    }
    for kind in ("bridge", "comparison"):
        typed = [row for row in rows if row["type"] == kind]
        if typed:
            result["by_type"][kind] = {
                "n": len(typed),
                "em": sum(row["em"] for row in typed) / len(typed),
                "f1": sum(row["f1"] for row in typed) / len(typed),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-memory", required=True)
    parser.add_argument("--conditioned-memory", required=True)
    parser.add_argument("--reader-a", required=True)
    parser.add_argument("--reader-b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = FreshReaderMemory(args.base_memory, args.conditioned_memory)
    count = min(args.max_samples, len(cache))
    model, tokenizer = load_receiver(args.model, device)
    reader_a, checkpoint_a = load_reader(model, args.reader_a, device)
    reader_b, checkpoint_b = load_reader(model, args.reader_b, device)
    if checkpoint_a["initialization"] != checkpoint_b["initialization"]:
        raise RuntimeError("Readers do not share the same initialization checkpoint")
    if sum(p.numel() for p in reader_a.parameters()) != sum(
        p.numel() for p in reader_b.parameters()
    ):
        raise RuntimeError("Reader parameter counts differ")

    records = []
    pairs = []
    for index in tqdm(range(count), desc="p3e_m_free_running"):
        a = cache.evidence_only(index)
        a_wrong = cache.evidence_only_hard(index)
        b = cache.question_conditioned(index)
        b_wrong = cache.question_conditioned_hard(index)
        row = a["row"]
        predictions = {}
        for condition in CONDITIONS:
            if condition == "question_only":
                result = plain_generate(
                    model, tokenizer, question_prompt(tokenizer, row), args.max_new_tokens
                )
                source_id, source_answer = row["id"], row["answer"]
            elif condition == "reader_off":
                result = generate(
                    model,
                    tokenizer,
                    reader_a,
                    row,
                    native_memory_to(a, device),
                    args.max_new_tokens,
                    enabled=False,
                )
                source_id, source_answer = row["id"], row["answer"]
            elif condition == "full_evidence_text":
                result = plain_generate(
                    model, tokenizer, full_text_prompt(tokenizer, row), args.max_new_tokens
                )
                source_id, source_answer = row["id"], row["answer"]
            elif condition == "evidence_only_native_reader_a":
                result = generate(
                    model,
                    tokenizer,
                    reader_a,
                    row,
                    native_memory_to(a, device),
                    args.max_new_tokens,
                )
                source_id, source_answer = row["id"], row["answer"]
            elif condition == "evidence_only_hard_shuffled_reader_a":
                result = generate(
                    model,
                    tokenizer,
                    reader_a,
                    row,
                    native_memory_to(a_wrong, device),
                    args.max_new_tokens,
                )
                source_id, source_answer = a_wrong["row"]["id"], a_wrong["row"]["answer"]
            elif condition == "question_conditioned_native_reader_b":
                result = generate(
                    model,
                    tokenizer,
                    reader_b,
                    row,
                    native_memory_to(b, device),
                    args.max_new_tokens,
                )
                source_id, source_answer = row["id"], row["answer"]
            else:
                result = generate(
                    model,
                    tokenizer,
                    reader_b,
                    row,
                    native_memory_to(b_wrong, device),
                    args.max_new_tokens,
                )
                source_id, source_answer = b_wrong["source_id"], b_wrong["source_answer"]
            em, f1 = answer_scores(result["prediction"], row["answer"])
            source_em, source_f1 = answer_scores(result["prediction"], source_answer)
            predictions[condition] = result["prediction"]
            records.append(
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
        pairs.append(
            {
                "id": row["id"],
                "a_correct_shuffled_switch": float(
                    normalize_answer(predictions["evidence_only_native_reader_a"])
                    != normalize_answer(
                        predictions["evidence_only_hard_shuffled_reader_a"]
                    )
                ),
                "b_correct_shuffled_switch": float(
                    normalize_answer(
                        predictions["question_conditioned_native_reader_b"]
                    )
                    != normalize_answer(
                        predictions[
                            "question_conditioned_hard_shuffled_reader_b"
                        ]
                    )
                ),
                "a_b_switch": float(
                    normalize_answer(predictions["evidence_only_native_reader_a"])
                    != normalize_answer(
                        predictions["question_conditioned_native_reader_b"]
                    )
                ),
                "question_only_equals_reader_off": float(
                    predictions["question_only"] == predictions["reader_off"]
                ),
            }
        )

    metrics = {condition: summarize(records, condition) for condition in CONDITIONS}
    delta_q = (
        metrics["question_conditioned_native_reader_b"]["f1"]
        - metrics["evidence_only_native_reader_a"]["f1"]
    )
    gap_a = (
        metrics["evidence_only_native_reader_a"]["f1"]
        - metrics["evidence_only_hard_shuffled_reader_a"]["f1"]
    )
    gap_b = (
        metrics["question_conditioned_native_reader_b"]["f1"]
        - metrics["question_conditioned_hard_shuffled_reader_b"]["f1"]
    )
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_jsonl(output / "pair_controls.jsonl", pairs)
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-M Fresh Native Reader Q-Conditioning Diagnosis",
            "samples": count,
            "conditions": metrics,
            "delta_q_automatic_f1": delta_q,
            "correct_shuffled_f1_gap_a": gap_a,
            "correct_shuffled_f1_gap_b": gap_b,
            "switch_rates": {
                key: sum(row[key] for row in pairs) / len(pairs)
                for key in pairs[0]
                if key != "id"
            },
            "reader_a_gates": reader_a.gates().detach().cpu().tolist(),
            "reader_b_gates": reader_b.gates().detach().cpu().tolist(),
            "reader_a": args.reader_a,
            "reader_b": args.reader_b,
            "same_initialization": checkpoint_a["initialization"],
            "same_architecture": True,
            "same_parameter_count": True,
            "writer_loaded": False,
            "canonical_projection_used": False,
            "manual_cpw_required": True,
        },
    )
    with (output / "manual_cpw_blind.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "condition",
                "question",
                "gold_answer",
                "generation",
                "C_P_W",
                "strict_correct",
                "lenient_correct",
            ]
        )
        for row in records:
            writer.writerow(
                [
                    row["id"],
                    row["condition"],
                    row["question"],
                    row["gold_answer"],
                    row["generation"],
                    "",
                    "",
                    "",
                ]
            )


if __name__ == "__main__":
    main()
