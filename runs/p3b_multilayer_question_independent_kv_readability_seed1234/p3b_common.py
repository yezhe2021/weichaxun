import hashlib
import json
import math
import random
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


LAYER_SETS = {
    "last1": [35],
    "last4": list(range(32, 36)),
    "last8": list(range(28, 36)),
    "uniform16": [int(round(x)) for x in np.linspace(0, 35, 16)],
    "all36": list(range(36)),
}
SENDER_MODES = ("evidence_only", "question_evidence")
SOURCES = ("hidden", "native_kv", "pca", "random", "trainable")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_block(row):
    return f"EVIDENCE A\n{row['evidence_a']}\n\nEVIDENCE B\n{row['evidence_b']}"


def sender_text(row, mode):
    evidence = evidence_block(row)
    if mode == "evidence_only":
        return evidence, 0, len(evidence)
    if mode == "question_evidence":
        text = f"QUESTION\n{row['question']}\n\n{evidence}"
        start = text.index(evidence)
        return text, start, start + len(evidence)
    raise ValueError(mode)


def all_occurrences(text, answer):
    haystack, needle = text.casefold(), answer.casefold()
    if not needle:
        return []
    found, cursor = [], 0
    while True:
        start = haystack.find(needle, cursor)
        if start < 0:
            break
        found.append((start, start + len(needle)))
        cursor = start + max(1, len(needle))
    return found


def normalize_answer(value):
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction, answer):
    predicted, target = normalize_answer(prediction), normalize_answer(answer)
    exact = float(predicted == target)
    p_tokens, t_tokens = predicted.split(), target.split()
    if not p_tokens or not t_tokens:
        return exact, float(p_tokens == t_tokens)
    overlap = sum((Counter(p_tokens) & Counter(t_tokens)).values())
    if not overlap:
        return exact, 0.0
    precision, recall = overlap / len(p_tokens), overlap / len(t_tokens)
    return exact, 2 * precision * recall / (precision + recall)


def token_indices(offsets, left, right):
    return [i for i, (start, end) in enumerate(offsets) if start != end and start < right and end > left]


def token_span_targets(offsets, answer_ranges):
    targets = []
    for left, right in answer_ranges:
        hits = [i for i, (start, end) in enumerate(offsets) if start != end and start < right and end > left]
        if hits:
            targets.append((hits[0], hits[-1]))
    return sorted(set(targets))


def marginal_span_loss(start_logits, end_logits, spans):
    if not spans:
        raise RuntimeError("No valid answer spans")
    starts = torch.tensor(sorted({s for s, _ in spans}), device=start_logits.device)
    ends = torch.tensor(sorted({e for _, e in spans}), device=end_logits.device)
    start_loss = -torch.logsumexp(F.log_softmax(start_logits, dim=-1).index_select(0, starts), dim=0)
    end_loss = -torch.logsumexp(F.log_softmax(end_logits, dim=-1).index_select(0, ends), dim=0)
    return start_loss + end_loss


def best_span(start_logits, end_logits, max_answer_tokens=16):
    length = int(start_logits.numel())
    best_score, best = -float("inf"), (0, 0)
    for start in range(length):
        upper = min(length, start + max_answer_tokens)
        values = start_logits[start] + end_logits[start:upper]
        score, offset = values.max(dim=0)
        if float(score) > best_score:
            best_score, best = float(score), (start, start + int(offset))
    return best


def resize_tokens(tensor, target):
    if tensor.shape[1] == target:
        return tensor
    indices = torch.linspace(0, tensor.shape[1] - 1, target, device=tensor.device).round().long()
    return tensor.index_select(1, indices)


def decode_span(evidence, offsets, start, end):
    if not offsets or start >= len(offsets) or end >= len(offsets):
        return ""
    left, right = offsets[start][0], offsets[end][1]
    if right <= left:
        return ""
    return evidence[left:right].strip()


def layer_set(name):
    if name not in LAYER_SETS:
        raise ValueError(f"Unknown layer set: {name}")
    values = LAYER_SETS[name]
    if len(values) != len(set(values)):
        raise RuntimeError(f"Layer set contains duplicates: {name}")
    return values


def stable_permutation(length, seed, device=None):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(length, generator=generator)
    return permutation.to(device) if device is not None else permutation


def mean(values):
    return float(np.mean(values)) if values else 0.0

