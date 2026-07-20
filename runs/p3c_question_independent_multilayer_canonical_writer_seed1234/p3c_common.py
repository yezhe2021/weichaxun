import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from p3b_common import answer_scores, best_span, decode_span, marginal_span_loss, mean, resize_tokens, stable_permutation


LAYER_CONFIGS = {
    "uniform16": [int(round(x)) for x in torch.linspace(0, 35, 16).tolist()],
    "all36": list(range(36)),
}
SEEDS = (1234, 2345, 3456)


class CanonicalLayer(nn.Module):
    def __init__(self, key_mean, value_mean, key_projection, value_projection, rank=32):
        super().__init__()
        input_dim, output_dim = key_projection.shape
        self.key_base = nn.Linear(input_dim, output_dim)
        self.value_base = nn.Linear(input_dim, output_dim)
        with torch.no_grad():
            self.key_base.weight.copy_(key_projection.T)
            self.value_base.weight.copy_(value_projection.T)
            self.key_base.bias.copy_(-(key_mean @ key_projection))
            self.value_base.bias.copy_(-(value_mean @ value_projection))
        self.key_down = nn.Linear(output_dim, rank, bias=False)
        self.key_up = nn.Linear(rank, output_dim, bias=False)
        self.value_down = nn.Linear(output_dim, rank, bias=False)
        self.value_up = nn.Linear(rank, output_dim, bias=False)
        nn.init.normal_(self.key_down.weight, std=0.02)
        nn.init.normal_(self.value_down.weight, std=0.02)
        nn.init.zeros_(self.key_up.weight)
        nn.init.zeros_(self.value_up.weight)
        self.key_norm = nn.LayerNorm(output_dim)
        self.value_norm = nn.LayerNorm(output_dim)
        self.key_scale = nn.Parameter(torch.ones(output_dim))
        self.value_scale = nn.Parameter(torch.ones(output_dim))
        self.register_buffer("key_mean", key_mean.clone(), persistent=False)
        self.register_buffer("value_mean", value_mean.clone(), persistent=False)
        self.register_buffer("key_projection", key_projection.clone(), persistent=False)
        self.register_buffer("value_projection", value_projection.clone(), persistent=False)
        self.register_buffer("initial_key_weight", key_projection.T.clone(), persistent=False)
        self.register_buffer("initial_value_weight", value_projection.T.clone(), persistent=False)

    def forward(self, keys, values):
        key_base = self.key_base(keys)
        value_base = self.value_base(values)
        canonical_key = self.key_norm(key_base + self.key_up(F.gelu(self.key_down(key_base)))) * self.key_scale
        canonical_value = self.value_norm(value_base + self.value_up(F.gelu(self.value_down(value_base)))) * self.value_scale
        return canonical_key, canonical_value

    def decode(self, keys, values):
        return keys @ self.key_projection.T + self.key_mean, values @ self.value_projection.T + self.value_mean

    def weight_regularization(self):
        return F.mse_loss(self.key_base.weight, self.initial_key_weight) + F.mse_loss(self.value_base.weight, self.initial_value_weight)


class MultiLayerCanonicalWriter(nn.Module):
    def __init__(self, projections, selected_layers, rank=32):
        super().__init__()
        self.selected_layers = list(selected_layers)
        pca = projections["pca"]
        self.layers = nn.ModuleList(
            [
                CanonicalLayer(
                    pca["key_mean"][layer].float(),
                    pca["value_mean"][layer].float(),
                    pca["key_projection"][layer].float(),
                    pca["value_projection"][layer].float(),
                    rank,
                )
                for layer in self.selected_layers
            ]
        )
        self.rank = rank

    def forward(self, native_keys, native_values):
        keys, values = [], []
        for module, layer in zip(self.layers, self.selected_layers):
            key, value = module(native_keys[layer], native_values[layer])
            keys.append(key)
            values.append(value)
        return torch.stack(keys), torch.stack(values)

    def decode(self, canonical_keys, canonical_values):
        keys, values = [], []
        for index, module in enumerate(self.layers):
            key, value = module.decode(canonical_keys[index], canonical_values[index])
            keys.append(key)
            values.append(value)
        return torch.stack(keys), torch.stack(values)

    def weight_regularization(self):
        return torch.stack([module.weight_regularization() for module in self.layers]).mean()

    def config(self):
        return {"selected_layers": self.selected_layers, "rank": self.rank, "canonical_dim": 256}


def teacher_trace(teacher, selected_keys, selected_values, question, memory_enabled=True):
    layer_outputs = []
    for reader, keys, values in zip(teacher.readers, selected_keys, selected_values):
        keys = reader.k_proj(reader.k_norm(keys))
        values = reader.v_proj(reader.v_norm(values))
        q_key = F.normalize(reader.q_key(question), dim=-1)
        q_value = reader.q_value(question)
        if memory_enabled:
            route_logits = keys @ q_key / math.sqrt(keys.shape[-1])
            attention = route_logits.softmax(dim=0)
            readout = torch.sum(attention[:, None] * values, dim=0)
        else:
            route_logits = torch.zeros(keys.shape[0], device=keys.device)
            attention = torch.zeros_like(route_logits)
            readout = torch.zeros_like(q_value)
        positions = reader.position(torch.arange(keys.shape[0], device=keys.device))
        token_state = torch.tanh(values + q_value[None] + readout[None] + positions)
        start = reader.start_head(token_state).squeeze(-1) + route_logits
        end = reader.end_head(token_state).squeeze(-1) + route_logits
        support = reader.support_head(token_state).squeeze(-1)
        router = reader.router(torch.cat((q_value, readout), dim=-1)).squeeze(-1)
        layer_outputs.append((start, end, support, router, attention, readout))
    layer_weights = torch.stack([output[3] for output in layer_outputs]).softmax(dim=0)
    return {
        "start": sum(weight * output[0] for weight, output in zip(layer_weights, layer_outputs)),
        "end": sum(weight * output[1] for weight, output in zip(layer_weights, layer_outputs)),
        "support": sum(weight * output[2] for weight, output in zip(layer_weights, layer_outputs)),
        "layer_weights": layer_weights,
        "attention": torch.stack([output[4] for output in layer_outputs]),
        "readouts": torch.stack([output[5] for output in layer_outputs]),
    }


def symmetric_kl(student, teacher):
    student_log = F.log_softmax(student, dim=-1)
    teacher_log = F.log_softmax(teacher, dim=-1)
    return 0.5 * (
        F.kl_div(student_log, teacher_log.exp(), reduction="batchmean")
        + F.kl_div(teacher_log, student_log.exp(), reduction="batchmean")
    )


def teacher_distillation(student, target):
    readout = 1.0 - F.cosine_similarity(student["readouts"], target["readouts"], dim=-1).mean()
    readout = readout + F.mse_loss(F.layer_norm(student["readouts"], (student["readouts"].shape[-1],)), F.layer_norm(target["readouts"], (target["readouts"].shape[-1],)))
    router = symmetric_kl(student["layer_weights"], target["layer_weights"])
    span = symmetric_kl(student["start"], target["start"]) + symmetric_kl(student["end"], target["end"])
    attention = symmetric_kl(student["attention"], target["attention"])
    return readout + 0.5 * router + 0.5 * span + 0.1 * attention, {
        "readout": readout.detach(), "router": router.detach(), "span": span.detach(), "attention": attention.detach()
    }


def relation_matrix(states):
    states = F.normalize(states, dim=-1)
    return states @ states.T


def structure_loss(canonical_keys, canonical_values, native_keys, native_values, max_tokens=32):
    total = canonical_keys.new_tensor(0.0)
    count = 0
    for layer in range(canonical_keys.shape[0]):
        length = canonical_keys.shape[1]
        indices = torch.linspace(0, length - 1, min(length, max_tokens), device=canonical_keys.device).round().long()
        ck = canonical_keys[layer].index_select(0, indices)
        cv = canonical_values[layer].index_select(0, indices)
        nk = native_keys[layer].index_select(0, indices)
        nv = native_values[layer].index_select(0, indices)
        total = total + F.mse_loss(relation_matrix(ck), relation_matrix(nk))
        total = total + F.mse_loss(relation_matrix(cv), relation_matrix(nv))
        canonical_binding = F.normalize(ck, dim=-1) @ F.normalize(cv, dim=-1).T
        native_binding = F.normalize(nk, dim=-1) @ F.normalize(nv, dim=-1).T
        total = total + 0.5 * F.mse_loss(canonical_binding, native_binding)
        diagonal = canonical_binding.diagonal().mean()
        off_diagonal = (canonical_binding.sum() - canonical_binding.diagonal().sum()) / max(1, canonical_binding.numel() - len(canonical_binding))
        total = total + 0.1 * F.relu(0.05 + off_diagonal - diagonal)
        count += 1
    return total / max(1, count)


def variance_floor(memories, floor=0.10):
    pooled_keys = torch.stack([keys.mean(dim=(0, 1)) for keys, _ in memories])
    pooled_values = torch.stack([values.mean(dim=(0, 1)) for _, values in memories])
    return F.relu(floor - pooled_keys.std(dim=0, unbiased=False)).mean() + F.relu(floor - pooled_values.std(dim=0, unbiased=False)).mean()


def temporary_span_loss(output, metadata, support_weight=0.05):
    loss = marginal_span_loss(output["start"], output["end"], metadata["answer_token_spans"])
    support = metadata["support_token_mask"].float().to(output["support"].device)
    return loss + support_weight * F.binary_cross_entropy_with_logits(output["support"], support)


class CanonicalCache:
    def __init__(self, index_path, capacity=3):
        self.path = Path(index_path)
        with self.path.open(encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.root = self.path.parent
        self.entries = self.index["entries"]
        self.capacity = capacity
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index not in self.loaded:
            self.loaded[index] = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]


def support_recall(logits, mask):
    mask = mask.bool().to(logits.device)
    count = int(mask.sum())
    if count == 0:
        return 0.0
    selected = logits.topk(min(count, len(logits))).indices
    return float(mask.index_select(0, selected).float().mean())


def summarize_records(records):
    output = []
    for condition in sorted({row["condition"] for row in records}):
        rows = [row for row in records if row["condition"] == condition]
        output.append({
            "condition": condition,
            "n": len(rows),
            "current_answer_em": mean([row["current_answer_em"] for row in rows]),
            "current_answer_f1": mean([row["current_answer_f1"] for row in rows]),
            "source_memory_em": mean([row["source_memory_em"] for row in rows]),
            "source_memory_f1": mean([row["source_memory_f1"] for row in rows]),
            "start_accuracy": mean([row["start_accuracy"] for row in rows]),
            "end_accuracy": mean([row["end_accuracy"] for row in rows]),
            "supporting_sentence_recall": mean([row["supporting_sentence_recall"] for row in rows]),
            "loss": mean([row["loss"] for row in rows]),
        })
    return output

