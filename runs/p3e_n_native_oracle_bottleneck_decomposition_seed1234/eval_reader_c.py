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
from p3e_n_common import ReceiverNativeConditionedCache


CONDITIONS = (
    "question_only",
    "reader_off",
    "full_evidence_text",
    "qwen3_4b_qconditioned_native_reader_c",
    "qwen3_4b_qconditioned_hard_shuffled_reader_c",
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


def summarize(records, condition):
    rows = [row for row in records if row["condition"] == condition]
    result = {
        "n": len(rows),
        "em": sum(row["em"] for row in rows) / len(rows),
        "f1": sum(row["f1"] for row in rows) / len(rows),
        "eos_rate": sum(row["eos_reached"] for row in rows) / len(rows),
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
    parser.add_argument("--memory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--p3em-evaluation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = ReceiverNativeConditionedCache(args.memory)
    count = min(args.max_samples, len(cache))
    model, tokenizer = load_receiver(args.model, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    records, pairs = [], []
    for index in tqdm(range(count), desc="p3e_n_reader_c_free_running"):
        correct = cache.correct(index)
        shuffled = cache.shuffled(index)
        row = correct["row"]
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
                    reader,
                    row,
                    native_memory_to(correct, device),
                    args.max_new_tokens,
                    enabled=False,
                )
                source_id, source_answer = row["id"], row["answer"]
            elif condition == "full_evidence_text":
                result = plain_generate(
                    model, tokenizer, full_text_prompt(tokenizer, row), args.max_new_tokens
                )
                source_id, source_answer = row["id"], row["answer"]
            elif condition == "qwen3_4b_qconditioned_native_reader_c":
                result = generate(
                    model,
                    tokenizer,
                    reader,
                    row,
                    native_memory_to(correct, device),
                    args.max_new_tokens,
                )
                source_id, source_answer = row["id"], row["answer"]
            else:
                result = generate(
                    model,
                    tokenizer,
                    reader,
                    row,
                    native_memory_to(shuffled, device),
                    args.max_new_tokens,
                )
                source_id, source_answer = (
                    shuffled["source_id"],
                    shuffled["source_answer"],
                )
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
                "correct_shuffled_switch": float(
                    normalize_answer(
                        predictions["qwen3_4b_qconditioned_native_reader_c"]
                    )
                    != normalize_answer(
                        predictions[
                            "qwen3_4b_qconditioned_hard_shuffled_reader_c"
                        ]
                    )
                ),
                "question_only_equals_reader_off": float(
                    predictions["question_only"] == predictions["reader_off"]
                ),
            }
        )

    metrics = {condition: summarize(records, condition) for condition in CONDITIONS}
    correct_f1 = metrics["qwen3_4b_qconditioned_native_reader_c"]["f1"]
    shuffled_f1 = metrics[
        "qwen3_4b_qconditioned_hard_shuffled_reader_c"
    ]["f1"]
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_jsonl(output / "pair_controls.jsonl", pairs)
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-N Native Oracle Bottleneck Decomposition",
            "samples": count,
            "conditions": metrics,
            "dependency_4b_automatic_f1": correct_f1 - shuffled_f1,
            "prediction_switch_rate": sum(
                row["correct_shuffled_switch"] for row in pairs
            )
            / len(pairs),
            "reader_off_exact_output_consistency": sum(
                row["question_only_equals_reader_off"] for row in pairs
            )
            / len(pairs),
            "reader_c_gates": reader.gates().detach().cpu().tolist(),
            "reader_c": args.checkpoint,
            "p3em_heterogeneous_reference": args.p3em_evaluation,
            "manual_cpw_required": True,
            "writer_loaded": False,
            "canonical_projection_used": False,
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
