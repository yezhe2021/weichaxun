import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from p3d3_common import write_json
from p3e_n_common import ReceiverNativeConditionedCache
from p3e_p_common import TeacherTrajectoryCache, build_position_ids, answer_suffix, full_text_prompt, question_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    args = parser.parse_args()

    memory = ReceiverNativeConditionedCache(args.memory)
    teacher = TeacherTrajectoryCache(args.teacher_cache)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    count = min(args.max_samples, len(memory), len(teacher))
    checked_tokens = 0
    for index in range(count):
        row = memory.correct(index)["row"]
        cached = teacher.load(index)
        if cached["id"] != row["id"]:
            raise RuntimeError("Teacher/sample ID mismatch")
        suffix = answer_suffix(tokenizer, row["answer"])
        if suffix != cached["answer_token_ids"]:
            raise RuntimeError("Teacher/student labels differ")
        target_prompt = tokenizer(
            question_prompt(tokenizer, row), add_special_tokens=False
        ).input_ids
        full_prompt = tokenizer(
            full_text_prompt(tokenizer, row), add_special_tokens=False
        ).input_ids
        position_ids, _ = build_position_ids(
            target_prompt, full_prompt, len(suffix), torch.device("cpu")
        )
        for step, token in enumerate(suffix):
            if cached["teacher_visible_answer_prefix_lengths"][step] != step:
                raise RuntimeError("Teacher answer-prefix leakage")
            student_index = len(target_prompt) - 1 + step
            teacher_position = len(full_prompt) - 1 + step
            if int(position_ids[0, student_index]) != teacher_position:
                raise RuntimeError("Position alignment failed")
            if token != cached["answer_token_ids"][step]:
                raise RuntimeError("Teacher/student target token mismatch")
            checked_tokens += 1
        if cached["teacher_states"].shape[:2] != (16, len(suffix)):
            raise RuntimeError("Teacher state shape mismatch")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "samples": count,
            "answer_prediction_steps": checked_tokens,
            "teacher_student_prefix_equal": True,
            "teacher_student_label_equal": True,
            "position_ids_aligned": True,
            "current_and_future_gold_hidden": False,
        },
    )


if __name__ == "__main__":
    main()
