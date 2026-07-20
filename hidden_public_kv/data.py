import json
import random
import re
import string
from collections import Counter
from pathlib import Path

import torch


SYSTEM_PROMPT = "Answer using the available evidence. Give only a short answer and end with exactly: FINAL: <answer>."


def normalize_answer(value):
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction, answer):
    predicted, target = normalize_answer(prediction), normalize_answer(answer)
    exact = float(predicted == target)
    predicted_tokens, target_tokens = predicted.split(), target.split()
    if not predicted_tokens or not target_tokens:
        return exact, float(predicted_tokens == target_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(target_tokens)).values())
    if not overlap:
        return exact, 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(target_tokens)
    return exact, 2 * precision * recall / (precision + recall)


def extract_prediction(text):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.I | re.S).strip()
    anchored = re.findall(r"(?:FINAL|ANSWER)\s*:\s*([^\n\r]+)", clean, flags=re.I)
    candidate = anchored[-1] if anchored else next((line for line in clean.splitlines() if line.strip()), "")
    return re.sub(r"^[\s`*:\-]+|[\s`*]+$", "", candidate).strip()


def load_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def gold_evidence(raw_row):
    context = {title: sentences for title, sentences in raw_row["context"]}
    support = {(title, int(index)) for title, index in raw_row["supporting_facts"]}
    records = []
    for title, sentences in raw_row["context"]:
        for index, sentence in enumerate(sentences):
            if (title, index) in support and sentence.strip():
                records.append({"title": title, "sentence_id": index, "text": sentence.strip()})
    if not records:
        raise RuntimeError(f"No gold supporting sentences for {raw_row['_id']}")
    return records


def prepare_row(raw_row):
    records = gold_evidence(raw_row)
    answer = str(raw_row["answer"])
    normalized_answer = normalize_answer(answer)
    removable = [i for i, item in enumerate(records) if normalized_answer and normalized_answer in normalize_answer(item["text"])]
    kept = [item["text"] for i, item in enumerate(records) if i not in removable]
    return {
        "id": raw_row["_id"], "question": raw_row["question"], "answer": answer,
        "type": raw_row.get("type", "unknown"), "level": raw_row.get("level"),
        "supporting_sentences": records,
        "evidence_text": "\n".join(item["text"] for item in records),
        "removed_evidence_text": "\n".join(kept) if removable and kept else None,
        "removed_sentence_indices": removable,
    }


def prepare_dataset(raw_path, output_path, limit=0, seed=1234):
    with open(raw_path, encoding="utf-8") as handle:
        source = json.load(handle)
    if limit:
        random.Random(seed).shuffle(source)
        source = source[:limit]
    rows = [prepare_row(row) for row in source]
    write_jsonl(output_path, rows)
    return rows


def apply_chat(tokenizer, question, include_evidence=None):
    content = f"QUESTION\n{question}"
    if include_evidence is not None:
        content = f"SUPPORTING EVIDENCE\n{include_evidence}\n\n{content}"
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = SYSTEM_PROMPT + "\n\n" + content + "\n\n"
    return prompt + "FINAL:"


def pack_answer(tokenizer, row, device, max_length=256):
    prompt_ids = tokenizer(apply_chat(tokenizer, row["question"]), add_special_tokens=False).input_ids
    suffix_ids = tokenizer(" " + row["answer"] + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    if len(prompt_ids) + len(suffix_ids) > max_length:
        raise ValueError(f"Receiver sequence exceeds max_length for {row['id']}")
    ids = torch.tensor([prompt_ids + suffix_ids], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, :len(prompt_ids)] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def per_sequence_nll(logits, labels):
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    losses = torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]), shifted_labels.reshape(-1),
        ignore_index=-100, reduction="none"
    ).view(shifted_labels.shape)
    valid = shifted_labels.ne(-100)
    return (losses * valid).sum(-1) / valid.sum(-1).clamp_min(1)
