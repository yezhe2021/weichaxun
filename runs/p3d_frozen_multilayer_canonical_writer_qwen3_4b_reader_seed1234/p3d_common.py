import hashlib
import json
import math
import random
import re
import string
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


INSUFFICIENT = "INSUFFICIENT"


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


def read_json(path):
    with open(path, encoding="utf-8") as handle: return json.load(handle)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


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


def extract_prediction(text):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.I | re.S)
    clean = re.sub(r"</?answer>", "", clean, flags=re.I).strip()
    anchored = re.findall(r"(?:FINAL|ANSWER)\s*:\s*([^\n\r]+)", clean, flags=re.I)
    candidate = anchored[-1] if anchored else next((line for line in clean.splitlines() if line.strip()), "")
    candidate = re.sub(r"^[\s`*:\-]+|[\s`*]+$", "", candidate)
    candidate = re.split(r"\s+(?:because|since|based on)\s+", candidate, maxsplit=1, flags=re.I)[0]
    return candidate.strip(), "final_anchor" if anchored else ("first_line" if candidate else "not_found")


def apply_chat(tokenizer, system, user):
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def question_prompt(tokenizer, row):
    system = "Answer using the external evidence. If it is insufficient, answer INSUFFICIENT. Give a short answer ending with exactly FINAL: <answer>."
    return apply_chat(tokenizer, system, f"QUESTION\n{row['question']}") + "FINAL:"


def full_text_prompt(tokenizer, row):
    system = "Answer using only the supplied evidence. Give a short answer ending with exactly FINAL: <answer>."
    user = f"QUESTION\n{row['question']}\n\nEVIDENCE A\n{row['evidence_a']}\n\nEVIDENCE B\n{row['evidence_b']}"
    return apply_chat(tokenizer, system, user) + "FINAL:"


def load_receiver(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, tokenizer


def decoder_layers(model):
    return model.model.layers


def sinusoidal_layer_embedding(groups, dimension):
    positions = torch.arange(groups, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension))
    embedding = torch.zeros(groups, dimension)
    embedding[:, 0::2] = torch.sin(positions * frequencies)
    embedding[:, 1::2] = torch.cos(positions * frequencies[: embedding[:, 1::2].shape[1]])
    return embedding


class FactorizedProjection(nn.Module):
    def __init__(self, input_dim, output_dim, rank, zero_output=False):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)
        nn.init.orthogonal_(self.down.weight)
        if zero_output: nn.init.normal_(self.up.weight, std=1e-3)
        else: nn.init.orthogonal_(self.up.weight)

    def forward(self, value):
        return self.up(F.silu(self.down(value)))


class ReceiverLayerReader(nn.Module):
    def __init__(self, hidden_size, memory_dim, rank, adapter_rank, gate_init):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.query = FactorizedProjection(hidden_size, memory_dim, rank)
        self.router_query = FactorizedProjection(hidden_size, memory_dim, rank)
        self.output = FactorizedProjection(memory_dim, hidden_size, rank, zero_output=True)
        self.adapter_down = nn.Linear(hidden_size, adapter_rank, bias=False)
        self.adapter_up = nn.Linear(adapter_rank, hidden_size, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        gate_init = min(0.1, max(1e-6, gate_init))
        self.gate_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init))))

    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, hidden, memory, layer_embedding):
        normalized = self.hidden_norm(hidden.float())
        query = F.layer_norm(self.query(normalized), (memory["keys"].shape[-1],))
        router_query = F.layer_norm(self.router_query(normalized), (memory["keys"].shape[-1],))
        keys = F.layer_norm(memory["keys"].float(), (memory["keys"].shape[-1],))
        values = F.layer_norm(memory["values"].float(), (memory["values"].shape[-1],))
        positioned_keys = F.layer_norm(keys + 0.1 * layer_embedding[:, None, :], (keys.shape[-1],))
        scores = torch.einsum("bsd,gtd->bsgt", query, positioned_keys) / math.sqrt(keys.shape[-1])
        mask = memory["mask"].bool()
        scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        per_group = torch.einsum("bsgt,gtd->bsgd", attention, values)
        router_features = F.layer_norm(per_group + 0.1 * layer_embedding[None, None, :, :], (keys.shape[-1],))
        router_logits = torch.einsum("bsd,bsgd->bsg", router_query, router_features) / math.sqrt(keys.shape[-1])
        router = router_logits.softmax(dim=-1)
        combined = torch.einsum("bsg,bsgd->bsd", router, per_group)
        projected = self.output(combined)
        projected = projected + self.adapter_up(F.silu(self.adapter_down(projected)))
        return (self.gate() * projected).to(hidden.dtype), attention, router


class MultiLayerEvidenceReader(nn.Module):
    def __init__(self, model, groups, memory_dim, rank=64, adapter_rank=32, gate_init=0.005, active_layers=None):
        super().__init__()
        self.groups = int(groups); self.memory_dim = int(memory_dim); self.rank = int(rank); self.adapter_rank = int(adapter_rank)
        self.hidden_size = int(model.config.hidden_size)
        self.active_layers = list(range(len(decoder_layers(model)))) if active_layers is None else list(active_layers)
        self.layer_to_local = {layer: local for local, layer in enumerate(self.active_layers)}
        self.readers = nn.ModuleList([
            ReceiverLayerReader(self.hidden_size, memory_dim, rank, adapter_rank, gate_init) for _ in self.active_layers
        ])
        self.register_buffer("canonical_layer_embedding", sinusoidal_layer_embedding(groups, memory_dim), persistent=True)
        self._memory = None; self._pending = {}; self._diagnostics = None

    def gates(self):
        return torch.stack([reader.gate() for reader in self.readers])

    def _read(self, hidden, receiver_layer):
        local = self.layer_to_local[receiver_layer]
        delta, attention, router = self.readers[local](hidden, self._memory, self.canonical_layer_embedding)
        if self._diagnostics is not None:
            slot = self._diagnostics.setdefault(str(receiver_layer), {"calls": 0, "delta_norm": 0.0, "gate": 0.0, "router": torch.zeros(self.groups), "attention_entropy": 0.0})
            slot["calls"] += 1
            slot["delta_norm"] += float(delta.detach().float().norm(dim=-1).mean().cpu())
            slot["gate"] = float(self.readers[local].gate().detach().cpu())
            slot["router"] += router.detach().float().mean(dim=(0, 1)).cpu()
            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean() / math.log(max(2, attention.shape[-1]))
            slot["attention_entropy"] += float(entropy.detach().cpu())
        return delta

    @contextmanager
    def inject(self, model, memory, diagnostics=None):
        if memory["keys"].ndim != 3 or memory["keys"].shape != memory["values"].shape:
            raise ValueError("Memory K/V must have matching [G,T,D] shape")
        if memory["keys"].shape[0] != self.groups or memory["keys"].shape[-1] != self.memory_dim:
            raise ValueError(f"Expected [{self.groups},T,{self.memory_dim}] memory")
        if memory["mask"].shape != memory["keys"].shape[:2]:
            raise ValueError("Memory mask must have [G,T] shape")
        self._memory, self._diagnostics = memory, diagnostics
        handles = []
        for layer_index in self.active_layers:
            layer = decoder_layers(model)[layer_index]
            def pre_hook(module, args, kwargs, layer_index=layer_index):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                self._pending[layer_index] = self._read(hidden, layer_index)
            def post_hook(module, args, kwargs, output, layer_index=layer_index):
                delta = self._pending.pop(layer_index)
                return (output[0] + delta,) + output[1:] if isinstance(output, tuple) else output + delta
            handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(layer.register_forward_hook(post_hook, with_kwargs=True))
        try: yield diagnostics
        finally:
            for handle in handles: handle.remove()
            self._pending.clear(); self._memory = None; self._diagnostics = None

    def metadata(self):
        return {"groups": self.groups, "memory_dim": self.memory_dim, "rank": self.rank, "adapter_rank": self.adapter_rank, "hidden_size": self.hidden_size, "active_layers": self.active_layers}


class EvidenceMemoryCache:
    def __init__(self, index_path, source, original_layers=None, capacity=3):
        self.path = Path(index_path); self.root = self.path.parent; self.index = read_json(index_path)
        self.entries = self.index["entries"]; self.source = source; self.original_layers = list(original_layers or [])
        self.capacity = capacity; self.loaded = OrderedDict()

    def __len__(self): return len(self.entries)

    def load(self, index):
        if index not in self.loaded:
            raw = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
            if self.source == "native16":
                states = raw["modes"]["evidence_only"]
                selected = torch.tensor(self.original_layers)
                keys = states["keys"].index_select(0, selected)
                values = states["values"].index_select(0, selected)
                metadata = {name: states[name] for name in ("offsets", "answer_token_spans", "token_ids", "valid_mask", "support_token_mask")}
                payload = {"row": raw["row"], "evidence": raw["evidence"], "keys": keys, "values": values, "metadata": metadata}
            else:
                payload = raw
            self.loaded[index] = payload
            while len(self.loaded) > self.capacity: self.loaded.popitem(last=False)
        self.loaded.move_to_end(index); return self.loaded[index]


def memory_to(payload, device):
    keys, values = payload["keys"].to(device).float(), payload["values"].to(device).float()
    groups, tokens = keys.shape[:2]
    return {"keys": keys, "values": values, "mask": torch.ones(groups, tokens, dtype=torch.bool, device=device), "answer_token_mask": torch.tensor([any(start <= i <= end for start, end in payload["metadata"]["answer_token_spans"]) for i in range(tokens)], dtype=torch.bool, device=device)}


def resize_memory(memory, target_tokens):
    if memory["keys"].shape[1] == target_tokens: return memory
    indices = torch.linspace(0, memory["keys"].shape[1] - 1, target_tokens, device=memory["keys"].device).round().long()
    return {"keys": memory["keys"].index_select(1, indices), "values": memory["values"].index_select(1, indices), "mask": memory["mask"].index_select(1, indices), "answer_token_mask": memory["answer_token_mask"].index_select(0, indices)}


def zero_memory(memory):
    return {**memory, "keys": torch.zeros_like(memory["keys"]), "values": torch.zeros_like(memory["values"]), "answer_token_mask": torch.zeros_like(memory["answer_token_mask"])}


def compose_memory(key_memory, value_memory, target_tokens=None):
    target_tokens = target_tokens or key_memory["keys"].shape[1]
    key_memory, value_memory = resize_memory(key_memory, target_tokens), resize_memory(value_memory, target_tokens)
    return {"keys": key_memory["keys"], "values": value_memory["values"], "mask": key_memory["mask"] & value_memory["mask"], "answer_token_mask": value_memory["answer_token_mask"]}


def permute_tokens(memory, seed):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(memory["keys"].shape[1], generator=generator).to(memory["keys"].device)
    return {"keys": memory["keys"].index_select(1, permutation), "values": memory["values"].index_select(1, permutation), "mask": memory["mask"].index_select(1, permutation), "answer_token_mask": memory["answer_token_mask"].index_select(0, permutation)}


def permute_layers(memory):
    permutation = torch.arange(memory["keys"].shape[0] - 1, -1, -1, device=memory["keys"].device)
    return {"keys": memory["keys"].index_select(0, permutation), "values": memory["values"].index_select(0, permutation), "mask": memory["mask"].index_select(0, permutation), "answer_token_mask": memory["answer_token_mask"]}


def pack_answer(tokenizer, prompt, answer, max_length, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    suffix = tokenizer(" " + answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    suffix = suffix[: max(1, max_length - len(prompt_ids))]
    ids = torch.tensor([prompt_ids + suffix], dtype=torch.long, device=device)
    labels = ids.clone(); labels[:, : len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels


def answer_logits(output, labels):
    shifted_labels = labels[:, 1:]
    return output.logits[:, :-1, :][shifted_labels != -100]


def forward_answer(model, tokenizer, reader, row, memory, answer, max_length, device, enabled=True, full_text=False):
    prompt = full_text_prompt(tokenizer, row) if full_text else question_prompt(tokenizer, row)
    ids, mask, labels = pack_answer(tokenizer, prompt, answer, max_length, device)
    if enabled:
        with reader.inject(model, memory): output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True)
    else:
        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True)
    return output.loss.float(), answer_logits(output, labels)


@torch.inference_mode()
def generate(model, tokenizer, reader, row, memory, max_new_tokens, enabled=True):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    kwargs = dict(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    diagnostics = {}
    start = time.perf_counter()
    if enabled:
        with reader.inject(model, memory, diagnostics): output = model.generate(**kwargs)
    else: output = model.generate(**kwargs)
    elapsed = time.perf_counter() - start
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
    return {"text": tokenizer.decode(tokens, skip_special_tokens=True), "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens, "elapsed_seconds": elapsed, "diagnostics": summarize_diagnostics(diagnostics)}


def summarize_diagnostics(diagnostics):
    rows = []
    for layer, values in diagnostics.items():
        calls = max(1, values["calls"])
        rows.append({"receiver_layer": int(layer), "gate": values["gate"], "delta_norm": values["delta_norm"] / calls, "attention_entropy": values["attention_entropy"] / calls, "canonical_router": (values["router"] / calls).tolist()})
    return sorted(rows, key=lambda row: row["receiver_layer"])
