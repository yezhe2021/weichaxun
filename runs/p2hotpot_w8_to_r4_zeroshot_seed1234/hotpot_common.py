import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_chat(tokenizer, system, user):
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def question_prompt(tokenizer, row):
    system = (
        "Answer the question using the external evidence available to you. Do not use unstated facts. "
        "If the evidence is insufficient, answer INSUFFICIENT. End with exactly: FINAL: <answer>."
    )
    return apply_chat(tokenizer, system, f"QUESTION\n{row['question']}") + "FINAL:"


def full_text_prompt(tokenizer, row):
    system = (
        "Answer the question using only the supplied evidence. Give only a short answer and end with exactly: "
        "FINAL: <answer>."
    )
    user = (
        f"QUESTION\n{row['question']}\n\nEVIDENCE A\n{row['evidence_a']}\n\n"
        f"EVIDENCE B\n{row['evidence_b']}"
    )
    return apply_chat(tokenizer, system, user) + "FINAL:"


def sender_text(row):
    text = (
        f"QUESTION\n{row['question']}\n\nEVIDENCE A\n{row['evidence_a']}\n\n"
        f"EVIDENCE B\n{row['evidence_b']}"
    )
    a_start = text.index(row["evidence_a"]); a_end = a_start + len(row["evidence_a"])
    b_start = text.index(row["evidence_b"], a_end); b_end = b_start + len(row["evidence_b"])
    return text, ((a_start, a_end), (b_start, b_end))


def extract_prediction(text):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.I | re.S).strip()
    anchored = re.findall(r"(?:FINAL|ANSWER)\s*:\s*([^\n\r]+)", clean, flags=re.I)
    candidate = anchored[-1] if anchored else (next((line for line in clean.splitlines() if line.strip()), ""))
    candidate = re.sub(r"^[\s`*:\-]+|[\s`*]+$", "", candidate)
    candidate = re.split(r"\s+(?:because|since|based on)\s+", candidate, maxsplit=1, flags=re.I)[0]
    return candidate.strip(), "final_anchor" if anchored else ("first_line" if candidate else "not_found")


def normalize_answer(value):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def remove_punctuation(text):
        return "".join(character for character in text if character not in string.punctuation)
    return " ".join(remove_articles(remove_punctuation(str(value).lower())).split())


def answer_scores(prediction, answer):
    predicted = normalize_answer(prediction); target = normalize_answer(answer)
    exact = float(predicted == target)
    predicted_tokens, target_tokens = predicted.split(), target.split()
    if not predicted_tokens or not target_tokens:
        return exact, float(predicted_tokens == target_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(target_tokens)).values())
    if overlap == 0:
        return exact, 0.0
    precision = overlap / len(predicted_tokens); recall = overlap / len(target_tokens)
    return exact, 2 * precision * recall / (precision + recall)


@torch.inference_mode()
def greedy_generate(model, tokenizer, prompt, max_new_tokens, reader=None, memory=None, enabled=False):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    kwargs = dict(
        **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    if enabled:
        with reader.inject(model, memory):
            output = model.generate(**kwargs)
    else:
        output = model.generate(**kwargs)
    generated = output[0, encoded["input_ids"].shape[1]:]
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "token_ids": generated.tolist(),
        "eos_reached": bool(tokenizer.eos_token_id in generated.tolist()),
    }


def summarize(records):
    summaries = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        by_type = {}
        for question_type in sorted({row["type"] for row in selected}):
            typed = [row for row in selected if row["type"] == question_type]
            by_type[question_type] = {
                "n": len(typed), "em": float(np.mean([row["em"] for row in typed])),
                "f1": float(np.mean([row["f1"] for row in typed])),
            }
        summaries.append({
            "condition": condition, "n": len(selected),
            "em": float(np.mean([row["em"] for row in selected])),
            "f1": float(np.mean([row["f1"] for row in selected])),
            "source_em": float(np.mean([row.get("source_em", 0.0) for row in selected])),
            "eos_rate": float(np.mean([row["eos_reached"] for row in selected])),
            "parse_rate": float(np.mean([row["extraction_method"] != "not_found" for row in selected])),
            "mean_generated_tokens": float(np.mean([len(row["token_ids"]) for row in selected])),
            "by_type": by_type,
        })
    return summaries
