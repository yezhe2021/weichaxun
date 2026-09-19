import hashlib
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import torch


CONDITIONS = [
    "question_only",
    "full_evidence_text",
    "supporting_text",
    "sender_summary_text",
    "native_headwise_kv",
    "learned_canonical_kv",
    "hard_shuffled_canonical_kv",
    "reader_off",
]


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(cwd):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return None


def tensor_bytes(*tensors):
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))


def supporting_text(row):
    evidence = f"EVIDENCE A\n{row['evidence_a']}\n\nEVIDENCE B\n{row['evidence_b']}"
    pieces = []
    for left, right in row.get("support_char_spans", []):
        piece = evidence[int(left):int(right)].strip()
        if piece:
            pieces.append(piece)
    if not pieces:
        raise RuntimeError(f"No supporting text recovered for {row['id']}")
    return "\n".join(pieces)


def apply_chat(tokenizer, system, user):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def evidence_prompt(tokenizer, row, evidence, label="EVIDENCE"):
    system = "Answer the question using the supplied evidence. Give a short answer. End with exactly FINAL: <answer>."
    user = f"QUESTION\n{row['question']}\n\n{label}\n{evidence}"
    return apply_chat(tokenizer, system, user) + "FINAL:"


def summary_prompt(tokenizer, row, evidence):
    system = "Produce a faithful evidence summary for downstream question answering."
    instruction = (
        "Summarize the evidence needed to answer the question.\n"
        "Include the relevant entities, relations, comparison facts,\n"
        "and intermediate multi-hop facts. Do not provide an unsupported answer."
    )
    user = f"{instruction}\n\nQUESTION\n{row['question']}\n\nEVIDENCE\n{evidence}"
    return apply_chat(tokenizer, system, user)


def answer_in_text(answer, text, normalize_answer):
    target = normalize_answer(answer)
    source = normalize_answer(text)
    return bool(target and target in source)


def model_context_limit(model):
    candidates = []
    for name in ("max_position_embeddings", "max_sequence_length", "seq_length"):
        value = getattr(model.config, name, None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            candidates.append(value)
    return max(candidates) if candidates else 32768


class CudaStageTimer:
    def __init__(self, enabled=True):
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.events = []

    @contextmanager
    def stage(self, name):
        if self.enabled:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self.events.append((name, start, end))
        else:
            start = time.perf_counter()
            yield
            self.events.append((name, start, time.perf_counter()))

    def totals(self):
        if self.enabled:
            torch.cuda.synchronize()
            result = {}
            for name, start, end in self.events:
                result[name] = result.get(name, 0.0) + float(start.elapsed_time(end))
            return result
        result = {}
        for name, start, end in self.events:
            result[name] = result.get(name, 0.0) + 1000.0 * (end - start)
        return result


@contextmanager
def timed_reader(reader, timer):
    originals = []
    for branch in reader.branches:
        original = branch.forward

        def wrapped(*args, __original=original, **kwargs):
            with timer.stage("reader"):
                return __original(*args, **kwargs)

        originals.append((branch, original))
        branch.forward = wrapped
    try:
        yield
    finally:
        for branch, original in originals:
            branch.forward = original


@contextmanager
def timed_model_forward(model, timer):
    original = model.forward
    call_index = [0]

    def wrapped(*args, **kwargs):
        name = "receiver_prefill" if call_index[0] == 0 else "receiver_decode_forward"
        call_index[0] += 1
        with timer.stage(name):
            return original(*args, **kwargs)

    model.forward = wrapped
    try:
        yield
    finally:
        model.forward = original


def aggregate_mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def safe_ratio(numerator, denominator, epsilon=1e-8):
    return None if abs(denominator) < epsilon else numerator / denominator


def strip_summary(text):
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.I | re.S)
    return text.strip()

