import hashlib
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU requires float32")
    return dtype


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PairCache:
    def __init__(self, index_path, capacity=8):
        self.index_path = Path(index_path)
        with open(self.index_path, encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.entries = self.index.get("pair_files", self.index.get("pairs_index"))
        if not isinstance(self.entries, list):
            raise ValueError(f"No pair index in {index_path}")
        self.root = self.index_path.parent
        self.capacity = int(capacity)
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index in self.loaded:
            self.loaded.move_to_end(index)
            return self.loaded[index]
        payload = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
        rows = payload.get("examples", payload.get("variants"))
        pair = {row["variant"]: row for row in rows}
        if "stable_alignment" in payload:
            pair["_stable_alignment"] = payload["stable_alignment"]
        self.loaded[index] = pair
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
        return pair


def label_vocabulary(*caches):
    labels = sorted({
        answer for cache in caches for entry in cache.entries
        for answer in (entry["base_answer"], entry["counterfactual_answer"])
    })
    if len(labels) != 40:
        raise ValueError(f"Expected 40 city labels, found {len(labels)}")
    return labels


def deterministic_negative(entries, index):
    own = {entries[index]["base_answer"], entries[index]["counterfactual_answer"]}
    for offset in range(1, len(entries)):
        candidate = (index + offset) % len(entries)
        other = {entries[candidate]["base_answer"], entries[candidate]["counterfactual_answer"]}
        if own.isdisjoint(other):
            return candidate
    raise RuntimeError(f"No answer-disjoint negative for {index}")


def projection(x, state, whiten=False):
    result = (x.float() - state["mean"].to(x.device)) @ state["components"].to(x.device)
    if whiten:
        result = result / state["scale"].to(x.device).clamp_min(1e-4)
    return result


class LowRankResidual(nn.Module):
    def __init__(self, dim=256, rank=64):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(F.silu(self.down(x)))


class TokenCanonicalWriter(nn.Module):
    def __init__(self, projections, dim=256, rank=64, freeze_base=True):
        super().__init__()
        self.dim = int(dim)
        self.key_base = nn.Linear(1024, dim, bias=True)
        self.value_base = nn.Linear(1024, dim, bias=True)
        with torch.no_grad():
            self.key_base.weight.copy_(projections["key"]["components"].T)
            self.key_base.bias.copy_(-projections["key"]["mean"] @ projections["key"]["components"])
            self.value_base.weight.copy_(projections["value"]["components"].T)
            self.value_base.bias.copy_(-projections["value"]["mean"] @ projections["value"]["components"])
        if freeze_base:
            for parameter in list(self.key_base.parameters()) + list(self.value_base.parameters()):
                parameter.requires_grad_(False)
        self.key_input_norm = nn.LayerNorm(dim)
        self.value_input_norm = nn.LayerNorm(dim)
        self.fusion = nn.Linear(2 * dim, dim)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.weight[:, :dim].copy_(0.5 * torch.eye(dim))
            self.fusion.weight[:, dim:].copy_(0.5 * torch.eye(dim))
            self.fusion.bias.zero_()
        self.shared_adapter = LowRankResidual(dim, rank)
        self.shared_norm = nn.LayerNorm(dim)
        self.key_adapter = LowRankResidual(dim, rank)
        self.value_adapter = LowRankResidual(dim, rank)
        self.key_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)
        self.key_log_scale = nn.Parameter(torch.zeros(dim))
        self.value_log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, key_flat, value_flat):
        key0 = self.key_input_norm(self.key_base(key_flat.float()))
        value0 = self.value_input_norm(self.value_base(value_flat.float()))
        shared0 = self.fusion(torch.cat((key0, value0), dim=-1))
        shared = self.shared_norm(shared0 + self.shared_adapter(shared0))
        keys = self.key_norm(key0 + self.key_adapter(shared)) * self.key_log_scale.exp()
        values = self.value_norm(value0 + self.value_adapter(shared)) * self.value_log_scale.exp()
        return {"keys": keys, "values": values, "shared": shared}


class VariableAttentionProbe(nn.Module):
    def __init__(self, classes, dim=256, queries=4):
        super().__init__()
        self.key_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)
        self.key_projection = nn.Linear(dim, 128, bias=False)
        self.value_projection = nn.Linear(dim, 256, bias=False)
        self.queries = nn.Parameter(torch.empty(queries, 128))
        self.classifier = nn.Sequential(
            nn.Linear(queries * 256, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, classes)
        )
        nn.init.normal_(self.queries, std=1.0 / math.sqrt(128))

    def forward(self, keys, values, mask, return_attention=False):
        key = self.key_projection(self.key_norm(keys.float()))
        value = self.value_projection(self.value_norm(values.float()))
        score = torch.einsum("qd,btd->bqt", self.queries, key) / math.sqrt(128)
        score = score.masked_fill(~mask[:, None, :], torch.finfo(score.dtype).min)
        attention = score.softmax(dim=-1)
        pooled = torch.einsum("bqt,btv->bqv", attention, value).flatten(1)
        logits = self.classifier(pooled)
        return (logits, attention) if return_attention else logits


def pad_memories(rows, device):
    length = max(row["keys"].shape[0] for row in rows)
    dim = rows[0]["keys"].shape[-1]
    keys = torch.zeros(len(rows), length, dim, device=device)
    values = torch.zeros_like(keys)
    mask = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        size = row["keys"].shape[0]
        keys[index, :size] = row["keys"].to(device)
        values[index, :size] = row["values"].to(device)
        mask[index, :size] = True
    return keys, values, mask


def effective_rank(matrix):
    singular = torch.linalg.svdvals(matrix.float())
    probability = singular / singular.sum().clamp_min(1e-12)
    return float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()).cpu())


def linear_cka(left, right):
    left = left.float() - left.float().mean(0, keepdim=True)
    right = right.float() - right.float().mean(0, keepdim=True)
    numerator = (left.T @ right).square().sum()
    denominator = ((left.T @ left).square().sum() * (right.T @ right).square().sum()).sqrt()
    return float((numerator / denominator.clamp_min(1e-12)).cpu())


def paired_consistency(records, condition="correct"):
    grouped = {}
    for row in records:
        if row["condition"] == condition:
            grouped.setdefault(row["pair_id"], {})[row["variant"]] = row
    complete = [row for row in grouped.values() if {"base", "counterfactual"}.issubset(row)]
    return float(np.mean([
        pair["base"]["original_target_correct"] * pair["counterfactual"]["original_target_correct"]
        for pair in complete
    ])) if complete else 0.0
