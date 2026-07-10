import csv
import json
import math
import random
import re
import string
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


CONDITIONS = (
    "question_only",
    "a_only",
    "b_only",
    "a_plus_b",
    "mismatched_a_plus_b",
    "shuffled_a_plus_b",
)


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    table = str.maketrans("", "", string.punctuation)
    return " ".join(remove_articles(str(text).lower().translate(table)).split())


def exact_match(prediction, gold):
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1(prediction, gold):
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = sum(
        min(prediction_tokens.count(token), gold_tokens.count(token))
        for token in set(prediction_tokens) & set(gold_tokens)
    )
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_answer(text, answer):
    target = normalize_answer(answer)
    return float(bool(target) and target in normalize_answer(text))


def extract_short_answer(text):
    text = str(text).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    for marker in ("\n\n", "\nQuestion:", "\nEvidence:", "<|im_end|>", "</s>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    for prefix in ("Answer:", "The answer is", "answer is", "It is", "It was"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return text.strip(" .\t\n\r\"'")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_dtype(name, device):
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    values = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = values[name]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU requires float32 or auto dtype")
    return dtype


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def get_layers(model):
    base = getattr(model, "model", model)
    if not hasattr(base, "layers"):
        raise ValueError("Expected a Qwen/Llama-style model.layers stack")
    return base.layers


def render_question_prompt(tokenizer, question, prompt_style="chat"):
    user_text = (
        f"Question:\n{question}\n\n"
        "Evidence:\nNo external evidence is provided.\n\n"
        "Return only the short answer, without explanation."
    )
    if prompt_style == "chat" and tokenizer.chat_template:
        messages = [
            {
                "role": "system",
                "content": "Answer the question from the supplied evidence. Be concise and do not explain.",
            },
            {"role": "user", "content": user_text},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        "Answer the question from the supplied evidence. Give only the short answer.\n\n"
        f"{user_text}\n\nAnswer:"
    )


def pack_prompt_answer(tokenizer, question, answer, max_length, device, prompt_style="chat"):
    prompt = render_question_prompt(tokenizer, question, prompt_style)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_text = " " + str(answer).strip()
    if tokenizer.eos_token:
        answer_text += tokenizer.eos_token
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    if len(prompt_ids) + len(answer_ids) > max_length:
        answer_ids = answer_ids[: max(max_length - len(prompt_ids), 1)]
    ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels, len(prompt_ids)


class OracleCacheDataset(Dataset):
    def __init__(self, path):
        self.payload = torch.load(path, map_location="cpu", weights_only=False)
        if self.payload.get("format_version") != 1:
            raise ValueError("Unsupported oracle cache format")
        self.examples = self.payload["examples"]

    @property
    def raw_dim(self):
        return int(self.payload["hidden_size"])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class ExternalEvidenceReader(nn.Module):
    def __init__(self, receiver_dim, shared_dim, num_heads):
        super().__init__()
        if shared_dim % num_heads:
            raise ValueError("shared_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = shared_dim // num_heads
        self.query_norm = nn.LayerNorm(receiver_dim)
        self.query_proj = nn.Linear(receiver_dim, shared_dim)
        self.output_proj = nn.Linear(shared_dim, receiver_dim)
        self.gate = nn.Linear(receiver_dim, 1)
        nn.init.normal_(self.output_proj.weight, std=0.01)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, hidden, evidence_key, evidence_value, evidence_mask):
        batch, query_tokens, _ = hidden.shape
        slots = evidence_key.shape[1]
        query_state = self.query_norm(hidden.float())
        query = self.query_proj(query_state).view(batch, query_tokens, self.num_heads, self.head_dim)
        key = evidence_key.view(batch, slots, self.num_heads, self.head_dim)
        value = evidence_value.view(batch, slots, self.num_heads, self.head_dim)
        scores = torch.einsum("bthd,bshd->bhts", query, key) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~evidence_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        readout = torch.einsum("bhts,bshd->bthd", attention, value).reshape(batch, query_tokens, -1)
        patch = self.output_proj(readout)
        gate = torch.sigmoid(self.gate(query_state))
        return patch * gate, attention, gate


class OracleEvidenceAdapter(nn.Module):
    def __init__(self, raw_dim, receiver_dim, shared_dim, num_heads, reader_layers):
        super().__init__()
        self.raw_dim = raw_dim
        self.shared_dim = shared_dim
        self.reader_layers = tuple(int(layer) for layer in reader_layers)
        self.slot_norm = nn.LayerNorm(raw_dim)
        self.key_proj = nn.Linear(raw_dim, shared_dim)
        self.value_proj = nn.Linear(raw_dim, shared_dim)
        self.key_norm = nn.LayerNorm(shared_dim)
        self.value_norm = nn.LayerNorm(shared_dim)
        self.readers = nn.ModuleDict(
            {
                str(layer): ExternalEvidenceReader(receiver_dim, shared_dim, num_heads)
                for layer in self.reader_layers
            }
        )

    def project_memory(self, raw_slots, slot_mask):
        normalized = self.slot_norm(raw_slots.float())
        key = self.key_norm(self.key_proj(normalized))
        value = self.value_norm(self.value_proj(normalized))
        return key, value, slot_mask.bool()


def flatten_condition_memory(example, condition, device, mismatch_example=None, shuffle=False):
    source = mismatch_example if mismatch_example is not None else example
    slots = source["slots"].to(device=device, dtype=torch.float32).unsqueeze(0)
    mask = source["slot_mask"].to(device=device, dtype=torch.bool).unsqueeze(0)
    if condition == "a_only":
        mask[:, 1, :] = False
    elif condition == "b_only":
        mask[:, 0, :] = False
    elif condition not in {"a_plus_b", "mismatched_a_plus_b", "shuffled_a_plus_b"}:
        raise ValueError(condition)
    slots = slots.flatten(1, 2)
    mask = mask.flatten(1, 2)
    if shuffle:
        permutation = torch.randperm(slots.shape[1], device=device)
        slots = slots[:, permutation]
        mask = mask[:, permutation]
    if not bool(mask.any()):
        raise RuntimeError("Evidence condition selected no valid oracle slots")
    return slots, mask


@contextmanager
def inject_evidence(receiver, adapter, memory, target_selector, diagnostics=None):
    evidence_key, evidence_value, evidence_mask = memory
    handles = []
    diagnostics = diagnostics if diagnostics is not None else {}

    for layer_index in adapter.reader_layers:
        layer = get_layers(receiver)[layer_index]
        reader = adapter.readers[str(layer_index)]

        def hook(module, args, kwargs, output, layer_index=layer_index, reader=reader):
            hidden = output[0] if isinstance(output, tuple) else output
            start, end = target_selector(hidden)
            if end <= start:
                return output
            target = hidden[:, start:end, :]
            patch, attention, gate = reader(target, evidence_key, evidence_value, evidence_mask)
            mixed = hidden.clone()
            mixed[:, start:end, :] = target + patch.to(target.dtype)
            diagnostics[str(layer_index)] = {
                "gate_mean": float(gate.detach().mean().cpu()),
                "attention_entropy": float(
                    (-(attention.float().clamp_min(1e-9).log() * attention.float()).sum(dim=-1).mean()).detach().cpu()
                ),
            }
            if isinstance(output, tuple):
                return (mixed,) + output[1:]
            return mixed

        handles.append(layer.register_forward_hook(hook, with_kwargs=True))
    try:
        yield diagnostics
    finally:
        for handle in handles:
            handle.remove()


def run_teacher_forced(receiver, tokenizer, adapter, example, condition, device, max_length, prompt_style, mismatch_example=None):
    ids, attention_mask, labels, prompt_length = pack_prompt_answer(
        tokenizer,
        example["question"],
        example["answer"],
        max_length,
        device,
        prompt_style,
    )
    slots, slot_mask = flatten_condition_memory(
        example,
        condition,
        device,
        mismatch_example=mismatch_example,
        shuffle=False,
    )
    memory = adapter.project_memory(slots, slot_mask)

    def selector(hidden):
        return prompt_length - 1, hidden.shape[1] - 1

    diagnostics = {}
    with inject_evidence(receiver, adapter, memory, selector, diagnostics):
        output = receiver(
            input_ids=ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
    return output, diagnostics


@torch.no_grad()
def greedy_generate(
    receiver,
    tokenizer,
    adapter,
    example,
    condition,
    device,
    max_new_tokens,
    prompt_style,
    mismatch_example=None,
):
    prompt = render_question_prompt(tokenizer, example["question"], prompt_style)
    current = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    past = None
    generated = []
    diagnostics = {}
    memory = None
    if condition != "question_only":
        shuffle = condition == "shuffled_a_plus_b"
        slots, slot_mask = flatten_condition_memory(
            example,
            condition,
            device,
            mismatch_example=mismatch_example,
            shuffle=shuffle,
        )
        memory = adapter.project_memory(slots, slot_mask)

    eos_ids = tokenizer.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    for _ in range(max_new_tokens):
        if memory is None:
            output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        else:
            def selector(hidden):
                return hidden.shape[1] - 1, hidden.shape[1]

            with inject_evidence(receiver, adapter, memory, selector, diagnostics):
                output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        next_token = int(output.logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_token)
        if next_token in eos_ids:
            break
        past = output.past_key_values
        current = torch.tensor([[next_token]], dtype=torch.long, device=device)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text, diagnostics


def summarize_generation(records, examples, conditions, p0_summary=None):
    condition_rows = []
    for condition in conditions:
        selected = [row for row in records if row["condition"] == condition]
        condition_rows.append(
            {
                "condition": condition,
                "n": len(selected),
                "exact_match": float(np.mean([row["exact_match"] for row in selected])),
                "answer_f1": float(np.mean([row["answer_f1"] for row in selected])),
                "contains_gold": float(np.mean([row["contains_gold"] for row in selected])),
                "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in selected])),
            }
        )

    by_sample = {}
    for row in records:
        by_sample.setdefault(row["sample"], {})[row["condition"]] = row
    paired = {}
    required = {"question_only", "a_only", "b_only", "a_plus_b"}
    complete = [values for values in by_sample.values() if required.issubset(values)]
    if complete:
        gains = [
            values["a_plus_b"]["answer_f1"]
            - max(values["a_only"]["answer_f1"], values["b_only"]["answer_f1"])
            for values in complete
        ]
        paired.update(
            {
                "n": len(complete),
                "mean_ab_f1_gain_over_best_single": float(np.mean(gains)),
                "ab_beats_both_single_f1_rate": float(np.mean([gain > 0 for gain in gains])),
                "compositional_exact_match_rate": float(
                    np.mean(
                        [
                            values["a_plus_b"]["exact_match"] == 1
                            and values["a_only"]["exact_match"] == 0
                            and values["b_only"]["exact_match"] == 0
                            for values in complete
                        ]
                    )
                ),
                "ab_f1_gain_over_question_only": float(
                    np.mean(
                        [
                            values["a_plus_b"]["answer_f1"]
                            - values["question_only"]["answer_f1"]
                            for values in complete
                        ]
                    )
                ),
            }
        )
    mismatch_complete = [
        values for values in by_sample.values() if {"a_plus_b", "mismatched_a_plus_b"}.issubset(values)
    ]
    if mismatch_complete:
        paired["correct_vs_mismatched_ab_f1_gap"] = float(
            np.mean(
                [
                    values["a_plus_b"]["answer_f1"]
                    - values["mismatched_a_plus_b"]["answer_f1"]
                    for values in mismatch_complete
                ]
            )
        )
    shuffle_complete = [
        values for values in by_sample.values() if {"a_plus_b", "shuffled_a_plus_b"}.issubset(values)
    ]
    if shuffle_complete:
        paired["slot_permutation_prediction_match_rate"] = float(
            np.mean(
                [
                    values["a_plus_b"]["generated_token_ids"]
                    == values["shuffled_a_plus_b"]["generated_token_ids"]
                    for values in shuffle_complete
                ]
            )
        )
    if p0_summary and complete:
        p0_conditions = {row["condition"]: row for row in p0_summary["conditions"]}
        p1_conditions = {row["condition"]: row for row in condition_rows}
        denominator = p0_conditions["a_plus_b"]["answer_f1"] - p0_conditions["question_only"]["answer_f1"]
        numerator = p1_conditions["a_plus_b"]["answer_f1"] - p1_conditions["question_only"]["answer_f1"]
        paired["p0_full_text_f1_gap_closure"] = float(numerator / denominator) if denominator else None
    return condition_rows, paired
