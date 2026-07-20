import hashlib
import json
import random
import re
import string
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle: json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def apply_chat(tokenizer, system, user):
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def question_prompt(tokenizer, row):
    system = "Answer using the external evidence. Give only a short answer and end with exactly: FINAL: <answer>."
    return apply_chat(tokenizer, system, f"QUESTION\n{row['question']}") + "FINAL:"


def full_text_prompt(tokenizer, row):
    system = "Answer using only the supplied evidence. Give only a short answer and end with exactly: FINAL: <answer>."
    user = f"QUESTION\n{row['question']}\n\nEVIDENCE A\n{row['evidence_a']}\n\nEVIDENCE B\n{row['evidence_b']}"
    return apply_chat(tokenizer, system, user) + "FINAL:"


def sender_text(row):
    text = f"QUESTION\n{row['question']}\n\nEVIDENCE A\n{row['evidence_a']}\n\nEVIDENCE B\n{row['evidence_b']}"
    a0 = text.index(row["evidence_a"]); a1 = a0 + len(row["evidence_a"])
    b0 = text.index(row["evidence_b"], a1); b1 = b0 + len(row["evidence_b"])
    return text, ((a0, a1), (b0, b1))


def extract_prediction(text):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.I | re.S)
    clean = re.sub(r"</?answer>", "", clean, flags=re.I).strip()
    anchored = re.findall(r"(?:FINAL|ANSWER)\s*:\s*([^\n\r]+)", clean, flags=re.I)
    candidate = anchored[-1] if anchored else next((line for line in clean.splitlines() if line.strip()), "")
    candidate = re.sub(r"^[\s`*:\-]+|[\s`*]+$", "", candidate)
    candidate = re.split(r"\s+(?:because|since|based on)\s+", candidate, maxsplit=1, flags=re.I)[0]
    return candidate.strip(), "final_anchor" if anchored else ("first_line" if candidate else "not_found")


def normalize_answer(value):
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction, answer):
    predicted, target = normalize_answer(prediction), normalize_answer(answer)
    exact = float(predicted == target); p_tokens, t_tokens = predicted.split(), target.split()
    if not p_tokens or not t_tokens: return exact, float(p_tokens == t_tokens)
    overlap = sum((Counter(p_tokens) & Counter(t_tokens)).values())
    if not overlap: return exact, 0.0
    precision, recall = overlap / len(p_tokens), overlap / len(t_tokens)
    return exact, 2 * precision * recall / (precision + recall)


class MemoryCache:
    def __init__(self, index_path, capacity=4):
        self.index_path = Path(index_path)
        with self.index_path.open(encoding="utf-8") as handle: self.index = json.load(handle)
        self.entries = self.index["entries"]; self.root = self.index_path.parent
        self.capacity = capacity; self.loaded = OrderedDict()

    def __len__(self): return len(self.entries)

    def load(self, index):
        if index not in self.loaded:
            self.loaded[index] = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
            while len(self.loaded) > self.capacity: self.loaded.popitem(last=False)
        self.loaded.move_to_end(index); return self.loaded[index]


def memory_for(payload, source, device):
    memory = payload["memories"][source]
    keys, values = memory["keys"].float(), memory["values"].float()
    if source == "raw_kv":
        if keys.shape[-1] != 1024: raise RuntimeError("Raw KV geometry drift")
        keys = keys.reshape(len(keys), 256, 4).mean(-1) * 2.0
        values = values.reshape(len(values), 256, 4).mean(-1) * 2.0
        keys = F.layer_norm(keys, (256,)); values = F.layer_norm(values, (256,))
    return {
        "keys": keys.to(device), "values": values.to(device),
        "mask": memory["mask"].to(device), "answer_token_mask": memory["answer_token_mask"].to(device),
    }


def zero_memory(memory):
    return {**memory, "keys": torch.zeros_like(memory["keys"]), "values": torch.zeros_like(memory["values"]), "answer_token_mask": torch.zeros_like(memory["answer_token_mask"])}


def resize_rows(value, target):
    if len(value) == target: return value
    indices = torch.linspace(0, len(value) - 1, target, device=value.device).round().long()
    return value.index_select(0, indices)


def mismatch_memory(current, other):
    return {**current, "values": resize_rows(other["values"], len(current["keys"])), "answer_token_mask": torch.zeros_like(current["answer_token_mask"])}


def pack_answer(tokenizer, row, answer, max_length, device):
    prompt_ids = tokenizer(question_prompt(tokenizer, row), add_special_tokens=False).input_ids
    suffix = tokenizer(" " + answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    if len(prompt_ids) + len(suffix) > max_length: suffix = suffix[:max(1, max_length - len(prompt_ids))]
    ids = torch.tensor([prompt_ids + suffix], dtype=torch.long, device=device)
    labels = ids.clone(); labels[:, :len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels


def sequence_nll(model, tokenizer, reader, row, memory, answer, max_length, device):
    ids, mask, labels = pack_answer(tokenizer, row, answer, max_length, device)
    with reader.inject(model, memory):
        return model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True).loss.float()


@torch.inference_mode()
def generate(model, tokenizer, reader, row, memory, max_new_tokens, enabled=True, full_text=False):
    prompt = full_text_prompt(tokenizer, row) if full_text else question_prompt(tokenizer, row)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    kwargs = dict(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if enabled:
        with reader.inject(model, memory): output = model.generate(**kwargs)
    else: output = model.generate(**kwargs)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
    return {"text": tokenizer.decode(tokens, skip_special_tokens=True), "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens}


def summarize_records(records):
    output = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        by_type = {}
        for kind in ("bridge", "comparison"):
            rows = [row for row in selected if row["type"] == kind]
            if rows: by_type[kind] = {"n": len(rows), "em": float(np.mean([r["em"] for r in rows])), "f1": float(np.mean([r["f1"] for r in rows]))}
        yes_no = [row for row in selected if row["answer_type"] == "yes_no"]
        output.append({
            "condition": condition, "n": len(selected), "em": float(np.mean([r["em"] for r in selected])),
            "f1": float(np.mean([r["f1"] for r in selected])),
            "yes_no_accuracy": float(np.mean([r["em"] for r in yes_no])) if yes_no else None,
            "source_em": float(np.mean([r.get("source_em", 0.0) for r in selected])),
            "eos_rate": float(np.mean([r["eos_reached"] for r in selected])),
            "parse_rate": float(np.mean([r["extraction_method"] != "not_found" for r in selected])), "by_type": by_type,
        })
    return output
