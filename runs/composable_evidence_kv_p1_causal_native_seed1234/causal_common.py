import csv
import json
import math
import re
import string
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


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


def normalize_answer(text):
    table = str.maketrans("", "", string.punctuation)
    return " ".join(str(text).lower().translate(table).split())


def exact_match(prediction, answer):
    return float(normalize_answer(prediction) == normalize_answer(answer))


def extract_answer(text):
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    text = text.splitlines()[0].strip() if text.splitlines() else text
    for prefix in ("Answer:", "The answer is", "answer is"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return text.strip(" .\t\n\r\"'")


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def parse_dtype(name, device):
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    result = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]
    if device.type == "cpu" and result != torch.float32:
        raise ValueError("CPU requires float32")
    return result


def get_layers(model):
    base = getattr(model, "model", model)
    if not hasattr(base, "layers"):
        raise ValueError("Expected model.layers")
    return base.layers


def render_prompt(tokenizer, row, condition, prompt_style="chat"):
    evidence = []
    if condition in {"a_only", "a_plus_b", "full_text"}:
        evidence.append("Source A:\n" + row["evidence_a"])
    if condition in {"b_only", "a_plus_b", "full_text"}:
        evidence.append("Source B:\n" + row["evidence_b"])
    evidence_text = "\n\n".join(evidence) if evidence else "No external evidence is provided."
    user = (
        f"Question:\n{row['question']}\n\nEvidence:\n{evidence_text}\n\n"
        "Return only the answer identifier. If the evidence is insufficient, return INSUFFICIENT."
    )
    if prompt_style == "chat" and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "Follow the evidence exactly. Do not use unstated knowledge."},
            {"role": "user", "content": user},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user + "\n\nAnswer:"


def pack_question_answer(tokenizer, row, target, max_length, device, prompt_style="chat"):
    prompt = render_prompt(tokenizer, row, "question_only", prompt_style)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    suffix = " " + target + (tokenizer.eos_token or "")
    answer_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    answer_ids = answer_ids[: max(1, max_length - len(prompt_ids))]
    input_ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return input_ids, torch.ones_like(input_ids), labels, len(prompt_ids)


def token_span_masks(text, entities, tokenizer):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.offset_mapping
    masks = []
    for entity in entities:
        start = text.find(entity)
        if start < 0:
            raise ValueError(f"Entity {entity} is absent from evidence")
        end = start + len(entity)
        masks.append([offset_start < end and offset_end > start for offset_start, offset_end in offsets])
    return encoded.input_ids, torch.tensor(masks, dtype=torch.bool)


def pool_spans(memory, masks):
    weights = masks.float()
    return torch.einsum("ct,td->cd", weights, memory) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)


class TwoHopReader(nn.Module):
    def __init__(self, memory_dim, state_dim, heads):
        super().__init__()
        if state_dim % heads:
            raise ValueError("state_dim must be divisible by heads")
        self.state_dim = state_dim
        self.heads = heads
        self.head_dim = state_dim // heads
        self.q_in = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, state_dim))
        self.mem_norm = nn.LayerNorm(memory_dim)
        self.key_a = nn.Linear(memory_dim, state_dim)
        self.value_a = nn.Linear(memory_dim, state_dim)
        self.key_b = nn.Linear(memory_dim, state_dim)
        self.value_b = nn.Linear(memory_dim, state_dim)
        self.answer_key = nn.Linear(memory_dim, state_dim)
        self.update_a = nn.GRUCell(state_dim, state_dim)
        self.update_b = nn.GRUCell(state_dim, state_dim)
        self.null_a = nn.Parameter(torch.zeros(1, state_dim))
        self.null_b = nn.Parameter(torch.zeros(1, state_dim))
        self.state_norm = nn.LayerNorm(state_dim)

    def attend(self, state, memory, key_proj, value_proj, null_value):
        if memory is None:
            return null_value.expand(state.shape[0], -1), None
        normalized = self.mem_norm(memory.float())
        key = key_proj(normalized).view(state.shape[0], memory.shape[1], self.heads, self.head_dim)
        value = value_proj(normalized).view(state.shape[0], memory.shape[1], self.heads, self.head_dim)
        query = state.view(state.shape[0], self.heads, self.head_dim)
        scores = torch.einsum("bhd,bthd->bht", query, key) / math.sqrt(self.head_dim)
        attention = scores.softmax(dim=-1)
        readout = torch.einsum("bht,bthd->bhd", attention, value).reshape(state.shape[0], -1)
        return readout, attention

    def forward(self, question_state, memory_a, memory_b):
        s0 = self.q_in(question_state.float())
        read_a, attention_a = self.attend(s0, memory_a, self.key_a, self.value_a, self.null_a)
        s1 = self.state_norm(self.update_a(read_a, s0))
        read_b, attention_b = self.attend(s1, memory_b, self.key_b, self.value_b, self.null_b)
        s2 = self.state_norm(self.update_b(read_b, s1))
        return s0, s1, s2, attention_a, attention_b

    def bridge_candidates(self, memory_b, span_masks):
        pooled = pool_spans(memory_b[0].float(), span_masks)
        return self.key_b(self.mem_norm(pooled))

    def answer_candidates(self, memory_b, span_masks):
        pooled = pool_spans(memory_b[0].float(), span_masks)
        return self.answer_key(self.mem_norm(pooled))


class ResidualWriter(nn.Module):
    def __init__(self, receiver_dim, state_dim, bottleneck, max_gate):
        super().__init__()
        self.max_gate = max_gate
        self.h_norm = nn.LayerNorm(receiver_dim)
        self.s_norm = nn.LayerNorm(state_dim)
        self.h_down = nn.Linear(receiver_dim, bottleneck)
        self.s_down = nn.Linear(state_dim, bottleneck)
        self.mix = nn.Sequential(nn.Linear(bottleneck * 3, bottleneck), nn.SiLU(), nn.Linear(bottleneck, receiver_dim))
        self.gate = nn.Linear(bottleneck * 2, 1)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, hidden, state):
        normalized_h = self.h_norm(hidden.float())
        projected_h = self.h_down(normalized_h)
        projected_s = self.s_down(self.s_norm(state.float())).unsqueeze(1).expand(-1, hidden.shape[1], -1)
        delta = self.mix(torch.cat([projected_h, projected_s, projected_h * projected_s], dim=-1))
        hidden_rms = hidden.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        delta_rms = delta.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        delta = delta * hidden_rms / delta_rms
        gate = self.max_gate * torch.sigmoid(self.gate(torch.cat([projected_h, projected_s], dim=-1)))
        scale = self.log_scale.exp().clamp_max(1.0)
        return delta * gate * scale, gate


class CausalEvidenceAdapter(nn.Module):
    def __init__(self, memory_dim, receiver_dim, state_dim, heads, writer_layers, writer_bottleneck=256, max_gate=0.5):
        super().__init__()
        self.writer_layers = tuple(int(value) for value in writer_layers)
        self.reader = TwoHopReader(memory_dim, state_dim, heads)
        self.writers = nn.ModuleDict(
            {str(layer): ResidualWriter(receiver_dim, state_dim, writer_bottleneck, max_gate) for layer in self.writer_layers}
        )


@contextmanager
def inject_state(receiver, adapter, state, selector, diagnostics=None):
    diagnostics = diagnostics if diagnostics is not None else {}
    handles = []
    for layer_index in adapter.writer_layers:
        writer = adapter.writers[str(layer_index)]
        layer = get_layers(receiver)[layer_index]

        def hook(module, args, kwargs, output, writer=writer, layer_index=layer_index):
            hidden = output[0] if isinstance(output, tuple) else output
            start, end = selector(hidden)
            if end <= start:
                return output
            patch, gate = writer(hidden[:, start:end, :], state)
            mixed = hidden.clone()
            mixed[:, start:end, :] = hidden[:, start:end, :] + patch.to(hidden.dtype)
            diagnostics[str(layer_index)] = float(gate.detach().mean().cpu())
            return (mixed,) + output[1:] if isinstance(output, tuple) else mixed

        handles.append(layer.register_forward_hook(hook, with_kwargs=True))
    try:
        yield diagnostics
    finally:
        for handle in handles:
            handle.remove()


def iter_cache(index_path, shuffle=False, seed=0):
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    shards = list(index["shards"])
    if shuffle:
        import random
        random.Random(seed).shuffle(shards)
    for shard in shards:
        rows = torch.load(Path(index_path).parent / shard, map_location="cpu", weights_only=False)["examples"]
        if shuffle:
            import random
            random.Random(f"{seed}:{shard}").shuffle(rows)
        for row in rows:
            yield row


def aggregate(rows, key="condition"):
    output = []
    for value in sorted({row[key] for row in rows}):
        selected = [row for row in rows if row[key] == value]
        output.append({key: value, "n": len(selected), "exact_match": float(np.mean([r["exact_match"] for r in selected]))})
    return output
