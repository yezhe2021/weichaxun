import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


def decoder_layers(model):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    language_model = getattr(base, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model.layers
    raise RuntimeError("Could not locate decoder layers")


def text_config(model):
    return getattr(model.config, "text_config", model.config)


def full_attention_layers(model):
    layers = decoder_layers(model); kinds = getattr(text_config(model), "layer_types", None)
    if kinds is None:
        return list(range(len(layers)))
    selected = [index for index, kind in enumerate(kinds) if kind == "full_attention"]
    if not selected:
        raise RuntimeError("No full-attention layers found")
    return selected


class TokenCanonicalReader(nn.Module):
    def __init__(self, model, canonical_dim=256, rank=64, max_gate=1.0, gate_init=0.02, active_layers=None):
        super().__init__()
        self.hidden_size = int(text_config(model).hidden_size)
        self.canonical_dim = int(canonical_dim); self.rank = int(rank); self.max_gate = float(max_gate)
        self.active_layers = list(active_layers if active_layers is not None else full_attention_layers(model))
        self.layer_to_local = {layer: index for index, layer in enumerate(self.active_layers)}
        count = len(self.active_layers)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.shared_query = nn.Linear(self.hidden_size, canonical_dim, bias=False)
        self.shared_output = nn.Linear(canonical_dim, self.hidden_size, bias=False)
        self.query_down = nn.ModuleList([nn.Linear(self.hidden_size, rank, bias=False) for _ in range(count)])
        self.query_up = nn.ModuleList([nn.Linear(rank, canonical_dim, bias=False) for _ in range(count)])
        self.output_down = nn.ModuleList([nn.Linear(canonical_dim, rank, bias=False) for _ in range(count)])
        self.output_up = nn.ModuleList([nn.Linear(rank, self.hidden_size, bias=False) for _ in range(count)])
        self.query_scale = nn.Parameter(torch.ones(count))
        ratio = max(-0.999, min(0.999, gate_init / max(max_gate, 1e-8)))
        self.gate_logits = nn.Parameter(torch.full((count,), float(torch.atanh(torch.tensor(ratio)))))
        nn.init.orthogonal_(self.shared_query.weight); nn.init.orthogonal_(self.shared_output.weight)
        for module in list(self.query_up) + list(self.output_up):
            nn.init.zeros_(module.weight)
        self._memory = None; self._pending = {}; self._diagnostics = None

    def gates(self):
        return self.max_gate * torch.tanh(self.gate_logits)

    def _read(self, hidden, layer_index):
        local = self.layer_to_local[layer_index]
        normalized = self.input_norm(hidden.float())
        query = self.shared_query(normalized) + self.query_up[local](F.silu(self.query_down[local](normalized)))
        query = self.query_scale[local] * query
        keys, values = self._memory["keys"].float(), self._memory["values"].float()
        score = torch.matmul(query, keys.T) / math.sqrt(self.canonical_dim)
        mask = self._memory.get("mask")
        if mask is not None:
            score = score.masked_fill(~mask.bool()[None, None, :], torch.finfo(score.dtype).min)
        attention = score.softmax(-1)
        readout = torch.matmul(attention, values)
        projected = self.shared_output(readout) + self.output_up[local](F.silu(self.output_down[local](readout)))
        delta = self.gates()[local] * projected
        if self._diagnostics is not None:
            slot = self._diagnostics.setdefault(str(layer_index), {"calls": 0, "target_mass": 0.0, "gate": 0.0, "delta_norm": 0.0})
            slot["calls"] += 1; slot["gate"] = float(self.gates()[local].detach().cpu())
            slot["delta_norm"] += float(delta.detach().norm(dim=-1).mean().cpu())
            answer_mask = self._memory.get("answer_token_mask")
            if answer_mask is not None and answer_mask.any():
                slot["target_mass"] += float(attention[..., answer_mask.bool()].sum(-1).mean().detach().cpu())
        return delta.to(hidden.dtype)

    @contextmanager
    def inject(self, model, memory, diagnostics=None):
        if memory["keys"].ndim != 2 or memory["keys"].shape != memory["values"].shape or memory["keys"].shape[-1] != self.canonical_dim:
            raise ValueError("Canonical memory must contain matching [T,256] K/V")
        self._memory = memory; self._diagnostics = diagnostics
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
        try:
            yield diagnostics
        finally:
            for handle in handles:
                handle.remove()
            self._pending.clear(); self._memory = None; self._diagnostics = None

    def metadata(self):
        return {
            "hidden_size": self.hidden_size, "canonical_dim": self.canonical_dim, "rank": self.rank,
            "max_gate": self.max_gate, "active_layers": self.active_layers,
        }

