import argparse
import csv
import json
import random
import re
from collections import OrderedDict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


INSUFFICIENT = "INSUFFICIENT"


def normalize_answer(text):
    return re.sub(r"^[\s`*\"']+|[\s`*\"'.,;:!?]+$", "", str(text)).casefold()


def extract_answer(text, allowed_answers):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.IGNORECASE | re.DOTALL)
    values = list(dict.fromkeys([*allowed_answers, INSUFFICIENT]))
    mapping = {normalize_answer(value): value for value in values}
    pattern = re.compile(
        r"(?<![\w-])(" + "|".join(sorted(map(re.escape, values), key=len, reverse=True)) + r")(?![\w-])",
        re.IGNORECASE,
    )
    anchored = re.findall(r"(?:FINAL|ANSWER|答案)\s*[:：]\s*([^\n\r]+)", clean, flags=re.IGNORECASE)
    for region in reversed(anchored):
        matches = pattern.findall(region)
        if matches:
            return mapping[normalize_answer(matches[-1])], "final_anchor"
    matches = pattern.findall(clean)
    if matches:
        return mapping[normalize_answer(matches[-1])], "last_valid_answer"
    return "", "not_found"


def load_complete_pairs(path, max_pairs, seed):
    grouped = OrderedDict()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped.setdefault(row["pair_id"], {})[row["variant"]] = row
    pairs = [pair for pair in grouped.values() if {"base", "counterfactual"} <= set(pair)]
    random.Random(seed).shuffle(pairs)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise RuntimeError("No complete base/counterfactual pairs were found")
    for pair in pairs:
        base = pair["base"]
        counterfactual = pair["counterfactual"]
        if base["question"] != counterfactual["question"] or base["evidence_a"] != counterfactual["evidence_a"]:
            raise ValueError(f"Malformed counterfactual pair: {base['pair_id']}")
        if normalize_answer(base["answer"]) == normalize_answer(counterfactual["answer"]):
            raise ValueError(f"Counterfactual answer did not change: {base['pair_id']}")
    return pairs


def compatible_negative(pair, candidate):
    base = pair["base"]
    other = candidate["base"]
    current_answers = {
        normalize_answer(pair["base"]["answer"]),
        normalize_answer(pair["counterfactual"]["answer"]),
    }
    other_answers = {
        normalize_answer(candidate["base"]["answer"]),
        normalize_answer(candidate["counterfactual"]["answer"]),
    }
    return (
        pair is not candidate
        and base["target_person"].casefold() not in other["evidence_a"].casefold()
        and base["target_organization"].casefold() not in other["evidence_b"].casefold()
        and current_answers.isdisjoint(other_answers)
    )


def negative_mapping(pairs):
    mapping = []
    for index, pair in enumerate(pairs):
        selected = None
        for offset in range(1, len(pairs)):
            candidate = pairs[(index + offset) % len(pairs)]
            if compatible_negative(pair, candidate):
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(f"No compatible negative pair for {pair['base']['pair_id']}")
        mapping.append(selected)
    return mapping


def apply_chat_template(tokenizer, system, user):
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{system}\n\n{user}\n\n"
    return prompt + "FINAL:"


def full_text_prompt(tokenizer, row, evidence_a=None, evidence_b=None):
    system = (
        "Use only the supplied evidence. Follow the relation from the person to the organization, "
        "then from that organization to its location. Ignore distractors. "
        "If either relation is missing, answer INSUFFICIENT. End with exactly: FINAL: <answer>."
    )
    user = (
        f"QUESTION\n{row['question']}\n\n"
        f"EVIDENCE A\n{evidence_a if evidence_a is not None else row['evidence_a']}\n\n"
        f"EVIDENCE B\n{evidence_b if evidence_b is not None else row['evidence_b']}"
    )
    return apply_chat_template(tokenizer, system, user)


def partial_prompt(tokenizer, row, evidence_label=None, evidence=None):
    system = (
        "Use only the supplied evidence. If the evidence is insufficient to connect the person in the "
        "question to an organization and then to a location, answer INSUFFICIENT. "
        "End with exactly: FINAL: <answer>."
    )
    user = f"QUESTION\n{row['question']}"
    if evidence_label is not None:
        user += f"\n\n{evidence_label}\n{evidence}"
    return apply_chat_template(tokenizer, system, user)


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens, device):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_length = encoded["input_ids"].shape[1]
    token_ids = generated[0, input_length:].tolist()
    eos_ids = tokenizer.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    return {
        "generated_token_ids": token_ids,
        "generated_text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "generated_tokens": len(token_ids),
        "eos_reached": bool(token_ids and token_ids[-1] in eos_ids),
    }


def condition_rows(tokenizer, pair, negative_pair):
    base = pair["base"]
    counterfactual = pair["counterfactual"]
    other = negative_pair["base"]
    return [
        {
            "condition": "question_only",
            "prompt": partial_prompt(tokenizer, base),
            "target": INSUFFICIENT,
            "leak_target": base["answer"],
        },
        {
            "condition": "a_only",
            "prompt": partial_prompt(tokenizer, base, "EVIDENCE A", base["evidence_a"]),
            "target": INSUFFICIENT,
            "leak_target": base["answer"],
        },
        {
            "condition": "b_only_base",
            "prompt": partial_prompt(tokenizer, base, "EVIDENCE B", base["evidence_b"]),
            "target": INSUFFICIENT,
            "leak_target": base["answer"],
        },
        {
            "condition": "b_only_counterfactual",
            "prompt": partial_prompt(
                tokenizer, counterfactual, "EVIDENCE B", counterfactual["evidence_b"]
            ),
            "target": INSUFFICIENT,
            "leak_target": counterfactual["answer"],
        },
        {
            "condition": "full_text_base",
            "prompt": full_text_prompt(tokenizer, base),
            "target": base["answer"],
            "leak_target": base["answer"],
        },
        {
            "condition": "full_text_counterfactual",
            "prompt": full_text_prompt(tokenizer, counterfactual),
            "target": counterfactual["answer"],
            "leak_target": counterfactual["answer"],
        },
        {
            "condition": "mismatched_b",
            "prompt": full_text_prompt(tokenizer, base, evidence_b=other["evidence_b"]),
            "target": INSUFFICIENT,
            "leak_target": base["answer"],
        },
        {
            "condition": "shuffled_full_text",
            "prompt": full_text_prompt(
                tokenizer,
                base,
                evidence_a=other["evidence_a"],
                evidence_b=other["evidence_b"],
            ),
            "target": INSUFFICIENT,
            "leak_target": base["answer"],
        },
    ]


def summarize_conditions(records):
    rows = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "n": len(selected),
                "exact_match": sum(row["exact_match"] for row in selected) / len(selected),
                "insufficient_rate": sum(
                    normalize_answer(row["prediction"]) == normalize_answer(INSUFFICIENT)
                    for row in selected
                )
                / len(selected),
                "leak_target_hit_rate": sum(row["leak_target_hit"] for row in selected) / len(selected),
                "answer_found_rate": sum(bool(row["prediction"]) for row in selected) / len(selected),
                "eos_rate": sum(row["eos_reached"] for row in selected) / len(selected),
                "mean_generated_tokens": sum(row["generated_tokens"] for row in selected) / len(selected),
            }
        )
    return rows


def paired_metrics(records):
    base = {row["pair_id"]: row for row in records if row["condition"] == "full_text_base"}
    counterfactual = {
        row["pair_id"]: row
        for row in records
        if row["condition"] == "full_text_counterfactual"
    }
    pair_ids = sorted(base.keys() & counterfactual.keys())
    paired_correct = [base[pair_id]["exact_match"] * counterfactual[pair_id]["exact_match"] for pair_id in pair_ids]
    switched = [
        normalize_answer(base[pair_id]["prediction"])
        != normalize_answer(counterfactual[pair_id]["prediction"])
        for pair_id in pair_ids
    ]
    return {
        "n": len(pair_ids),
        "paired_counterfactual_consistency": sum(paired_correct) / len(pair_ids),
        "paired_correct_count": int(sum(paired_correct)),
        "prediction_switch_rate": sum(switched) / len(pair_ids),
    }


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Direct free-running answerability demo for the 1.7B sender")
    parser.add_argument("--model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument(
        "--data",
        default="/home/yezhe/伪查询/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234/data/test.jsonl",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.dtype == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU execution requires float32")

    pairs = load_complete_pairs(args.data, args.max_pairs, args.seed)
    negatives = negative_mapping(pairs)
    allowed_answers = sorted(
        {
            answer
            for pair in pairs
            for variant in ("base", "counterfactual")
            for answer in (
                pair[variant]["answer"],
                pair[variant].get("counterpart_answer", ""),
                *pair[variant].get("candidate_answers", []),
            )
            if answer
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    records = []
    for pair_index, pair in enumerate(tqdm(pairs, desc="sender_answerability")):
        pair_id = pair["base"]["pair_id"]
        for specification in condition_rows(tokenizer, pair, negatives[pair_index]):
            result = generate(model, tokenizer, specification["prompt"], args.max_new_tokens, device)
            prediction, extraction_method = extract_answer(result["generated_text"], allowed_answers)
            records.append(
                {
                    "pair_id": pair_id,
                    "condition": specification["condition"],
                    "target": specification["target"],
                    "leak_target": specification["leak_target"],
                    "prediction": prediction,
                    "exact_match": float(
                        normalize_answer(prediction) == normalize_answer(specification["target"])
                    ),
                    "leak_target_hit": float(
                        normalize_answer(prediction) == normalize_answer(specification["leak_target"])
                    ),
                    "extraction_method": extraction_method,
                    "prompt": specification["prompt"],
                    **result,
                }
            )

    conditions = summarize_conditions(records)
    paired = paired_metrics(records)
    condition_map = {row["condition"]: row for row in conditions}
    base_em = condition_map["full_text_base"]["exact_match"]
    counterfactual_em = condition_map["full_text_counterfactual"]["exact_match"]
    passed = (
        base_em >= 0.75
        and counterfactual_em >= 0.75
        and paired["paired_counterfactual_consistency"] >= 0.625
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_csv(output / "condition_summary.csv", conditions)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "args": vars(args),
                "model": args.model,
                "pairs": len(pairs),
                "conditions": conditions,
                "paired_metrics": paired,
                "demo_gate": {
                    "base_em_threshold": 0.75,
                    "counterfactual_em_threshold": 0.75,
                    "paired_consistency_threshold": 0.625,
                    "passed": passed,
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
