import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import answer_scores, load_receiver, write_json, write_jsonl
from p3e_n_common import ReceiverNativeConditionedCache
from p3e_p_common import (
    SELECTED_LAYERS,
    TeacherTrajectoryCache,
    generate_student,
    load_reader,
    native_memory,
    pack_student,
    stack_trace,
    state_diagnostics,
)


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
    target = teacher["teacher_states"].float().to(model.device)
    return state_diagnostics(original, corrected, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--a0", required=True)
    parser.add_argument("--a1", required=True)
    parser.add_argument("--c1", required=True)
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
    checkpoints = {"a0": args.a0, "a1": args.a1, "c1": args.c1}
    records = []
    diagnostics = {}

    for method, checkpoint_path in checkpoints.items():
        reader, checkpoint = load_reader(model, checkpoint_path, device)
        reader.requires_grad_(False)
        method_diagnostics = []
        for index in tqdm(range(count), desc=f"p3e_p_eval_{method}"):
            correct = cache.correct(index)
            shuffled = cache.shuffled(index)
            row = correct["row"]
            teacher = teacher_cache.load(index)
            correct_memory = native_memory(correct, device)
            shuffled_memory = native_memory(shuffled, device)
            for suffix, memory, source_answer in (
                ("correct_kv", correct_memory, row["answer"]),
                ("hard_shuffled", shuffled_memory, shuffled["source_answer"]),
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
                        "condition": f"{method}_{suffix}",
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
            method_diagnostics.append(
                diagnostic_forward(
                    model,
                    tokenizer,
                    reader,
                    row,
                    correct_memory,
                    teacher,
                )
            )
        diagnostics[method] = {
            key: sum(row[key] for row in method_diagnostics)
            / len(method_diagnostics)
            for key in method_diagnostics[0]
        }
        del reader
        torch.cuda.empty_cache()

    # Position-aligned Reader-off control uses the A0 object only as a disabled
    # context and therefore applies no learned parameter.
    reader_off, _ = load_reader(model, args.a0, device)
    reader_off.requires_grad_(False)
    for index in tqdm(range(count), desc="p3e_p_eval_reader_off"):
        correct = cache.correct(index)
        row = correct["row"]
        result = generate_student(
            model,
            tokenizer,
            reader_off,
            row,
            native_memory(correct, device),
            args.max_new_tokens,
            enabled=False,
        )
        em, f1 = answer_scores(result["prediction"], row["answer"])
        records.append(
            {
                "id": row["id"],
                "type": row.get("type"),
                "condition": "reader_off_position_aligned",
                "question": row["question"],
                "gold_answer": row["answer"],
                "source_answer": row["answer"],
                "prediction": result["prediction"],
                "generation": result["text"],
                "parse_method": result["parse_method"],
                "token_ids": result["token_ids"],
                "eos_reached": result["eos_reached"],
                "em": em,
                "f1": f1,
                "source_em": em,
                "source_f1": f1,
            }
        )

    conditions = [
        "reader_off_position_aligned",
        "a0_correct_kv",
        "a0_hard_shuffled",
        "a1_correct_kv",
        "a1_hard_shuffled",
        "c1_correct_kv",
        "c1_hard_shuffled",
    ]
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-P Trajectory-Supervised Memory Interface Diagnosis",
            "samples": count,
            "conditions": {
                condition: summarize(records, condition)
                for condition in conditions
            },
            "state_diagnostics": diagnostics,
            "teacher_branch_used_during_generation": False,
            "teacher_states_read_only_for_offline_diagnostics": True,
            "position_aligned": True,
            "checkpoints": checkpoints,
        },
    )


if __name__ == "__main__":
    main()

