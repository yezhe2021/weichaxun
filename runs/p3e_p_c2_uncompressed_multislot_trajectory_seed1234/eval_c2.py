import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import answer_scores, load_receiver, write_json, write_jsonl
from p3e_n_common import ReceiverNativeConditionedCache
from p3e_p_common import (
    SELECTED_LAYERS,
    TeacherTrajectoryCache,
    generate_student,
    native_memory,
    pack_student,
    stack_trace,
    state_diagnostics,
)
from p3e_p_c2_common import load_c2_reader


def summarize(records, condition):
    rows = [row for row in records if row["condition"] == condition]
    result = {
        "n": len(rows),
        "em": sum(row["em"] for row in rows) / max(1, len(rows)),
        "f1": sum(row["f1"] for row in rows) / max(1, len(rows)),
        "eos_rate": sum(row["eos_reached"] for row in rows)
        / max(1, len(rows)),
        "average_output_tokens": sum(len(row["token_ids"]) for row in rows)
        / max(1, len(rows)),
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


def slot_metrics(slots, slot_attention):
    # slots: [L,J,K,D], slot_attention: [L,J,K]
    normalized = F.normalize(slots.float(), dim=-1)
    similarity = torch.einsum(
        "ljkd,ljmd->ljkm", normalized, normalized
    )
    count = similarity.shape[-1]
    off_diagonal = ~torch.eye(
        count, dtype=torch.bool, device=similarity.device
    )
    average_cosine = similarity[..., off_diagonal].mean()
    probabilities = slot_attention.float().clamp_min(1e-9)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    return {
        "slot_off_diagonal_cosine": float(average_cosine),
        "state_query_slot_attention_entropy": float(entropy.mean()),
        "state_query_slot_attention_max": float(
            probabilities.max(dim=-1).values.mean()
        ),
    }


@torch.inference_mode()
def diagnostic_forward(model, tokenizer, reader, row, memory, teacher):
    packed = pack_student(tokenizer, row, model.device)
    trace = {}
    with reader.inject(
        model, memory, packed["prediction_positions"], trace
    ):
        model(
            input_ids=packed["input_ids"],
            attention_mask=packed["attention_mask"],
            position_ids=packed["position_ids"],
            use_cache=False,
            return_dict=True,
        )
    original = stack_trace(trace, SELECTED_LAYERS, "original")
    corrected = stack_trace(trace, SELECTED_LAYERS, "corrected")
    slots = stack_trace(trace, SELECTED_LAYERS, "slots")
    slot_attention = stack_trace(
        trace, SELECTED_LAYERS, "slot_attention"
    )
    target = teacher["teacher_states"].float().to(model.device)
    result = state_diagnostics(original, corrected, target)
    result.update(slot_metrics(slots, slot_attention))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = ReceiverNativeConditionedCache(args.memory)
    teacher_cache = TeacherTrajectoryCache(args.teacher_cache)
    count = min(args.max_samples, len(cache), len(teacher_cache))
    model, tokenizer = load_receiver(args.model, device)
    reader, checkpoint = load_c2_reader(model, args.checkpoint, device)
    reader.requires_grad_(False)
    records = []
    diagnostics = []

    for index in tqdm(range(count), desc="p3e_p_c2_eval"):
        correct = cache.correct(index)
        shuffled = cache.shuffled(index)
        row = correct["row"]
        correct_memory = native_memory(correct, device)
        for condition, memory, source_answer in (
            ("c2_correct_kv", correct_memory, row["answer"]),
            (
                "c2_hard_shuffled",
                native_memory(shuffled, device),
                shuffled["source_answer"],
            ),
        ):
            result = generate_student(
                model,
                tokenizer,
                reader,
                row,
                memory,
                args.max_new_tokens,
                enabled=True,
            )
            em, f1 = answer_scores(result["prediction"], row["answer"])
            source_em, source_f1 = answer_scores(
                result["prediction"], source_answer
            )
            records.append(
                {
                    "id": row["id"],
                    "type": row.get("type"),
                    "condition": condition,
                    "question": row["question"],
                    "gold_answer": row["answer"],
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
        diagnostics.append(
            diagnostic_forward(
                model,
                tokenizer,
                reader,
                row,
                correct_memory,
                teacher_cache.load(index),
            )
        )

    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-P-C2 Uncompressed Multi-Slot Trajectory Reader",
            "samples": count,
            "conditions": {
                condition: summarize(records, condition)
                for condition in ("c2_correct_kv", "c2_hard_shuffled")
            },
            "state_and_slot_diagnostics": {
                key: sum(row[key] for row in diagnostics)
                / len(diagnostics)
                for key in diagnostics[0]
            },
            "checkpoint": args.checkpoint,
            "reader_metadata": checkpoint["reader_metadata"],
            "teacher_branch_used_during_generation": False,
            "teacher_states_read_only_for_offline_diagnostics": True,
        },
    )


if __name__ == "__main__":
    main()

