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
    load_receiver,
    write_json,
    write_jsonl,
)
from p3e_n_common import ReceiverNativeConditionedCache
from p3e_o_common import (
    NativeTrajectoryOracle,
    aligned_target_positions,
    longest_common_suffix,
    regular_positions,
)


CONDITIONS = {
    "o1_oracle_attention_output": ("attention_output", True),
    "o2_oracle_post_attention_state": ("post_attention_state", True),
    "o3_oracle_block_output": ("block_output", True),
    "o3_oracle_block_output_unaligned": ("block_output", False),
}


def question_prompt(tokenizer, row):
    system = (
        "Answer the question with a short answer. "
        "End with exactly FINAL: <answer>."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"QUESTION\n{row['question']}"},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt + "FINAL:"


def full_text_prompt(tokenizer, row):
    system = (
        "Answer the question using the supplied gold evidence. "
        "Give a short answer. End with exactly FINAL: <answer>."
    )
    user = (
        f"QUESTION\n{row['question']}\n\n"
        f"GOLD SUPPORTING EVIDENCE\n{evidence_block(row)}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt + "FINAL:"


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
def synchronous_oracle_generate(
    model,
    tokenizer,
    patcher,
    row,
    mode,
    position_aligned,
    max_new_tokens,
):
    device = model.device
    full_prompt_ids = tokenizer(
        full_text_prompt(tokenizer, row), add_special_tokens=False
    ).input_ids
    target_prompt_ids = tokenizer(
        question_prompt(tokenizer, row), add_special_tokens=False
    ).input_ids
    if len(full_prompt_ids) <= len(target_prompt_ids):
        raise RuntimeError("Full-text prompt must be longer than target prompt")
    common_suffix = longest_common_suffix(full_prompt_ids, target_prompt_ids)
    if common_suffix == 0:
        raise RuntimeError("Prompts do not share an answer-prefix suffix")

    generated = []
    for _ in range(max_new_tokens):
        full_ids = torch.tensor(
            [full_prompt_ids + generated], dtype=torch.long, device=device
        )
        target_ids = torch.tensor(
            [target_prompt_ids + generated], dtype=torch.long, device=device
        )
        full_positions = regular_positions(full_ids.shape[1], device)
        if position_aligned:
            target_positions = aligned_target_positions(
                len(target_prompt_ids),
                len(full_prompt_ids),
                len(generated),
                common_suffix,
                device,
            )
        else:
            target_positions = regular_positions(target_ids.shape[1], device)

        oracle_states = {}
        with patcher.capture(oracle_states):
            model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                position_ids=full_positions,
                use_cache=False,
                return_dict=True,
            )
        with patcher.patch(oracle_states, mode):
            target_output = model(
                input_ids=target_ids,
                attention_mask=torch.ones_like(target_ids),
                position_ids=target_positions,
                use_cache=False,
                return_dict=True,
            )
        next_token = int(target_output.logits[0, -1].argmax().item())
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    prediction, parse_method = extract_prediction(text)
    return {
        "text": text,
        "prediction": prediction,
        "parse_method": parse_method,
        "token_ids": generated,
        "eos_reached": tokenizer.eos_token_id in generated,
        "position_aligned": position_aligned,
        "full_prompt_tokens": len(full_prompt_ids),
        "target_prompt_tokens": len(target_prompt_ids),
        "position_gap": len(full_prompt_ids) - len(target_prompt_ids),
        "common_suffix_tokens": common_suffix,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = ReceiverNativeConditionedCache(args.memory)
    count = min(args.max_samples, len(cache))
    model, tokenizer = load_receiver(args.model, device)
    checkpoint = torch.load(
        args.reader_checkpoint, map_location="cpu", weights_only=False
    )
    selected_layers = checkpoint["reader_metadata"]["selected_layers"]
    if len(selected_layers) != 16:
        raise RuntimeError(
            f"Expected Reader C's 16 injection layers, got {selected_layers}"
        )
    patcher = NativeTrajectoryOracle(model, selected_layers)
    records = []

    for index in tqdm(range(count), desc="p3e_o_oracle_patching"):
        row = cache.correct(index)["row"]
        for condition, (mode, aligned) in CONDITIONS.items():
            result = synchronous_oracle_generate(
                model,
                tokenizer,
                patcher,
                row,
                mode,
                aligned,
                args.max_new_tokens,
            )
            em, f1 = answer_scores(result["prediction"], row["answer"])
            records.append(
                {
                    "id": row["id"],
                    "type": row.get("type"),
                    "condition": condition,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "prediction": result["prediction"],
                    "generation": result["text"],
                    "parse_method": result["parse_method"],
                    "token_ids": result["token_ids"],
                    "eos_reached": result["eos_reached"],
                    "em": em,
                    "f1": f1,
                    "position_aligned": result["position_aligned"],
                    "full_prompt_tokens": result["full_prompt_tokens"],
                    "target_prompt_tokens": result["target_prompt_tokens"],
                    "position_gap": result["position_gap"],
                    "common_suffix_tokens": result["common_suffix_tokens"],
                }
            )

    write_jsonl(output / "per_sample_generation.jsonl", records)
    summary = {
        "status": "complete",
        "experiment": "P3-E-O Native Trajectory Oracle Patching",
        "samples": count,
        "model": args.model,
        "selected_layers": selected_layers,
        "free_running": True,
        "gold_answer_prefix_used": False,
        "current_token_only_patch": True,
        "writer_canonical_reader_used": False,
        "conditions": {
            condition: summarize(records, condition)
            for condition in CONDITIONS
        },
    }
    write_json(output / "SUCCESS.json", summary)

    with (output / "manual_cpw_blind.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "type",
                "condition",
                "question",
                "gold_answer",
                "generation",
                "prediction",
                "C_P_W",
                "strict_correct",
                "lenient_correct",
                "notes",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: row.get(key, "")
                    for key in writer.fieldnames
                }
            )


if __name__ == "__main__":
    main()

