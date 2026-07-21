import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from p3d_common import (
    EvidenceMemoryCache, FactorizedProjection, answer_logits, decoder_layers, extract_prediction,
    full_text_prompt, memory_to, pack_answer, question_prompt, read_json, resize_memory,
    sinusoidal_layer_embedding, write_json, write_jsonl,
)


CONFIGURATIONS = {
    "uniform8": [0, 5, 10, 15, 20, 25, 30, 35],
    "midlate8": [14, 17, 20, 23, 26, 29, 32, 35],
    "key4": [12, 20, 28, 35],
    "all36": list(range(36)),
}


def canonical_spec(protocol, split):
    item = protocol["canonical16"]
    return item["canonical_cache"][split], item["groups"], item["memory_dim"]


def canonical_cache(protocol, split, capacity=3):
    path, _, _ = canonical_spec(protocol, split)
    return EvidenceMemoryCache(path, "canonical16", capacity=capacity)


def hard_negative_mapping(cache):
    answers = [entry.get("answer", "") for entry in cache.entries]
    rows = [cache.load(index)["row"] for index in range(len(cache))]
    lengths = [int(cache.load(index)["keys"].shape[1]) for index in range(len(cache))]
    mapping = []
    for index, row in enumerate(rows):
        candidates = [candidate for candidate, other in enumerate(rows) if candidate != index and str(other.get("answer", "")).casefold() != str(row.get("answer", "")).casefold()]
        def cost(candidate):
            other = rows[candidate]
            return (
                int(other.get("type") != row.get("type")),
                int(other.get("answer_type") != row.get("answer_type")),
                abs(len(str(answers[candidate]).split()) - len(str(answers[index]).split())),
                abs(lengths[candidate] - lengths[index]),
                candidate,
            )
        mapping.append(min(candidates, key=cost))
    return mapping


def oracle_token_mask(payload, device):
    tokens = int(payload["keys"].shape[1])
    selected = torch.zeros(tokens, dtype=torch.bool, device=device)
    for start, end in payload["metadata"].get("answer_token_spans", []):
        selected[max(0, int(start)): min(tokens, int(end) + 1)] = True
    if not selected.any():
        support = torch.as_tensor(payload["metadata"].get("support_token_mask", []), dtype=torch.bool, device=device)
        if support.numel() == tokens: selected |= support
    if not selected.any(): selected[:] = True
    return selected


def memory_with_oracle(payload, device, token_oracle=False, group_mask=None):
    memory = memory_to(payload, device)
    if token_oracle:
        selected = oracle_token_mask(payload, device)
        memory["mask"] = memory["mask"] & selected[None, :]
    if group_mask is not None: memory["oracle_group_mask"] = group_mask.bool().to(device)
    return memory


def install_oracle_forward(reader):
    def oracle_forward(module, hidden, memory, layer_embedding):
        normalized = module.hidden_norm(hidden.float())
        dimension = memory["keys"].shape[-1]
        query = F.layer_norm(module.query(normalized), (dimension,))
        router_query = F.layer_norm(module.router_query(normalized), (dimension,))
        keys = F.layer_norm(memory["keys"].float(), (dimension,))
        values = F.layer_norm(memory["values"].float(), (dimension,))
        positioned_keys = F.layer_norm(keys + 0.1 * layer_embedding[:, None, :], (dimension,))
        scores = torch.einsum("bsd,gtd->bsgt", query, positioned_keys) / math.sqrt(dimension)
        scores = scores.masked_fill(~memory["mask"].bool()[None, None, :, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        per_group = torch.einsum("bsgt,gtd->bsgd", attention, values)
        router_features = F.layer_norm(per_group + 0.1 * layer_embedding[None, None, :, :], (dimension,))
        router_logits = torch.einsum("bsd,bsgd->bsg", router_query, router_features) / math.sqrt(dimension)
        if "oracle_group_mask" in memory:
            router_logits = router_logits.masked_fill(~memory["oracle_group_mask"][None, None, :], torch.finfo(router_logits.dtype).min)
        router = router_logits.softmax(dim=-1)
        combined = torch.einsum("bsg,bsgd->bsd", router, per_group)
        projected = module.output(combined)
        projected = projected + module.adapter_up(F.silu(module.adapter_down(projected)))
        return (module.gate() * projected).to(hidden.dtype), attention, router
    import types
    for module in reader.readers: module.forward = types.MethodType(oracle_forward, module)
    return reader


def load_span_probe(protocol, device):
    from train_eval_p3b_probe import MultiLayerSpanProbe
    writer = Path(protocol["canonical16"]["writer_checkpoint"])
    checkpoint_path = writer.parent.parent / "fresh_probe" / "fresh_probe_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    probe = MultiLayerSpanProbe("pca", list(range(int(protocol["canonical16"]["groups"])))).to(device)
    probe.load_state_dict(checkpoint["model"]); probe.eval()
    for parameter in probe.parameters(): parameter.requires_grad_(False)
    return probe, checkpoint_path


@torch.inference_mode()
def span_teacher(probe, payload, device, top_groups=4):
    keys = payload["keys"].float().to(device); values = payload["values"].float().to(device)
    question = payload["question_state"].float().to(device)
    output = probe(keys, values, question)
    layer_weights = output["layer_weights"].float()
    attention = output["attention"].float()
    joint = layer_weights[:, None] * attention
    joint = joint / joint.sum().clamp_min(1e-8)
    group_mask = torch.zeros_like(layer_weights, dtype=torch.bool)
    group_mask[layer_weights.topk(min(top_groups, len(layer_weights))).indices] = True
    return {"joint": joint, "layer_weights": layer_weights, "attention": attention, "group_mask": group_mask}


class SharedReaderBlock(nn.Module):
    def __init__(self, hidden_size, memory_dim, rank=64, adapter_rank=32):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.query = FactorizedProjection(hidden_size, memory_dim, rank)
        self.router_query = FactorizedProjection(hidden_size, memory_dim, rank)
        self.output = FactorizedProjection(memory_dim, hidden_size, rank, zero_output=True)
        self.adapter_down = nn.Linear(hidden_size, adapter_rank, bias=False)
        self.adapter_up = nn.Linear(adapter_rank, hidden_size, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        self.compat_scale = nn.Parameter(torch.tensor(1.0))
        self.compat_bias = nn.Parameter(torch.tensor(-1.0))

    def read(self, hidden, memory, layer_embedding):
        dimension = memory["keys"].shape[-1]
        normalized = self.hidden_norm(hidden.float())
        query = F.layer_norm(self.query(normalized), (dimension,))
        router_query = F.layer_norm(self.router_query(normalized), (dimension,))
        keys = F.layer_norm(memory["keys"].float(), (dimension,))
        values = F.layer_norm(memory["values"].float(), (dimension,))
        positioned_keys = F.layer_norm(keys + 0.1 * layer_embedding[:, None, :], (dimension,))
        scores = torch.einsum("bsd,gtd->bsgt", query, positioned_keys) / math.sqrt(dimension)
        scores = scores.masked_fill(~memory["mask"].bool()[None, None, :, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        per_group = torch.einsum("bsgt,gtd->bsgd", attention, values)
        router_features = F.layer_norm(per_group + 0.1 * layer_embedding[None, None, :, :], (dimension,))
        router_logits = torch.einsum("bsd,bsgd->bsg", router_query, router_features) / math.sqrt(dimension)
        router = router_logits.softmax(dim=-1)
        combined = torch.einsum("bsg,bsgd->bsd", router, per_group)
        projected = self.output(combined)
        projected = projected + self.adapter_up(F.silu(self.adapter_down(projected)))
        alignment = scores.max(dim=-1).values
        compatibility = self.compat_scale * torch.einsum("bsg,bsg->bs", router, alignment) + self.compat_bias
        return projected, attention, router, compatibility


class GroundedEvidenceReader(nn.Module):
    def __init__(self, model, groups, memory_dim, active_layers, shared_blocks=4, rank=64, adapter_rank=32, gate_init=0.005):
        super().__init__()
        self.groups = int(groups); self.memory_dim = int(memory_dim); self.hidden_size = int(model.config.hidden_size)
        self.active_layers = list(active_layers); self.shared_blocks = min(int(shared_blocks), len(self.active_layers))
        self.rank = int(rank); self.adapter_rank = int(adapter_rank)
        self.blocks = nn.ModuleList([SharedReaderBlock(self.hidden_size, memory_dim, rank, adapter_rank) for _ in range(self.shared_blocks)])
        self.block_assignment = [min(self.shared_blocks - 1, index * self.shared_blocks // len(self.active_layers)) for index in range(len(self.active_layers))]
        initial = math.log(gate_init / (1.0 - gate_init))
        self.gate_logits = nn.Parameter(torch.full((len(self.active_layers),), initial))
        self.register_buffer("canonical_layer_embedding", sinusoidal_layer_embedding(groups, memory_dim), persistent=True)
        self.layer_to_local = {layer: local for local, layer in enumerate(self.active_layers)}
        self._memory = None; self._pending = {}; self._trace = None

    def gates(self): return torch.sigmoid(self.gate_logits)

    def _read(self, hidden, receiver_layer):
        local = self.layer_to_local[receiver_layer]; block = self.blocks[self.block_assignment[local]]
        projected, attention, router, compatibility = block.read(hidden, self._memory, self.canonical_layer_embedding)
        validity = torch.sigmoid(compatibility).unsqueeze(-1)
        delta = (self.gates()[local] * validity * projected).to(hidden.dtype)
        if self._trace is not None:
            self._trace[receiver_layer] = {"delta": delta, "attention": attention, "router": router, "compatibility": compatibility, "validity": validity}
        return delta

    @contextmanager
    def inject(self, model, memory, trace=None):
        self._memory, self._trace = memory, trace
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
        try: yield trace
        finally:
            for handle in handles: handle.remove()
            self._pending.clear(); self._memory = None; self._trace = None

    def compatibility_from_hidden(self, hidden_by_layer, memory):
        scores = []
        for local, layer in enumerate(self.active_layers):
            hidden = hidden_by_layer[layer].reshape(1, 1, -1)
            block = self.blocks[self.block_assignment[local]]
            _, _, _, compatibility = block.read(hidden, memory, self.canonical_layer_embedding)
            scores.append(compatibility.mean())
        return torch.stack(scores).mean()

    def metadata(self):
        return {"groups": self.groups, "memory_dim": self.memory_dim, "hidden_size": self.hidden_size, "active_layers": self.active_layers, "shared_blocks": self.shared_blocks, "block_assignment": self.block_assignment, "rank": self.rank, "adapter_rank": self.adapter_rank}


def answer_positions(labels):
    return (labels[0] != -100).nonzero(as_tuple=False).flatten()


def forward_grounded(model, tokenizer, reader, row, memory, answer, max_length, device, enabled=True, trace=None):
    ids, mask, labels = pack_answer(tokenizer, question_prompt(tokenizer, row), answer, max_length, device)
    if enabled:
        with reader.inject(model, memory, trace): output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True)
    else:
        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True)
    return output.loss.float(), answer_logits(output, labels), labels


@torch.inference_mode()
def generate_grounded(model, tokenizer, reader, row, memory, max_new_tokens=32, enabled=True):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    trace = {}
    kwargs = dict(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if enabled:
        with reader.inject(model, memory, trace): output = model.generate(**kwargs)
    else: output = model.generate(**kwargs)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist(); text = tokenizer.decode(tokens, skip_special_tokens=True)
    prediction, parse_status = extract_prediction(text)
    diagnostics = []
    for layer, item in sorted(trace.items()):
        diagnostics.append({"receiver_layer": layer, "gate": float(reader.gates()[reader.layer_to_local[layer]].detach()), "validity": float(item["validity"].detach().mean()), "compatibility": float(item["compatibility"].detach().mean()), "router": item["router"].detach().float().mean(dim=(0, 1)).cpu().tolist(), "delta_rms": float(item["delta"].detach().float().square().mean().sqrt())})
    return {"text": text, "prediction": prediction, "parse_status": parse_status, "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens, "diagnostics": diagnostics}


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]
