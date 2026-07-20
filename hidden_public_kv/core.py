import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


def decoder_backbone(model):
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise RuntimeError("Could not locate decoder layers")


def decoder_layers(model):
    return decoder_backbone(model).layers


def text_config(model):
    return getattr(model.config, "text_config", model.config)


class IndependentRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(dtype)


@dataclass
class PublicMemory:
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    mask: torch.Tensor

    def validate(self, layers: int, kv_heads: int, head_dim: int) -> "PublicMemory":
        if len(self.keys) != layers or len(self.values) != layers:
            raise ValueError(f"Expected {layers} public layers")
        if self.mask.ndim != 2:
            raise ValueError("Public mask must be [batch, memory_tokens]")
        expected = (self.mask.shape[0], kv_heads, self.mask.shape[1], head_dim)
        for key, value in zip(self.keys, self.values):
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(f"Expected public K/V shape {expected}")
        return self

    def to(self, device=None, dtype=None) -> "PublicMemory":
        def move(value):
            return value.to(device=device, dtype=dtype or value.dtype)
        return PublicMemory(tuple(move(x) for x in self.keys), tuple(move(x) for x in self.values), self.mask.to(device=device))

    def zero(self) -> "PublicMemory":
        return PublicMemory(tuple(torch.zeros_like(x) for x in self.keys), tuple(torch.zeros_like(x) for x in self.values), self.mask)


class PublicWriter(nn.Module):
    def __init__(self, hidden_size: int, layers: int = 8, kv_heads: int = 8, head_dim: int = 128, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layer_count = int(layers)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        width = self.kv_heads * self.head_dim
        self.norms = nn.ModuleList([IndependentRMSNorm(hidden_size, eps) for _ in range(layers)])
        self.key_projections = nn.ModuleList([nn.Linear(hidden_size, width, bias=False) for _ in range(layers)])
        self.value_projections = nn.ModuleList([nn.Linear(hidden_size, width, bias=False) for _ in range(layers)])

    def forward(self, hidden_taps: Sequence[torch.Tensor], mask: torch.Tensor) -> PublicMemory:
        if len(hidden_taps) != self.layer_count:
            raise ValueError(f"Expected {self.layer_count} hidden taps")
        keys, values = [], []
        for hidden, norm, key_projection, value_projection in zip(
            hidden_taps, self.norms, self.key_projections, self.value_projections
        ):
            if hidden.requires_grad or hidden.grad_fn is not None:
                raise RuntimeError("Sender hidden taps must be detached before Public Writer")
            batch, tokens, width = hidden.shape
            if width != self.hidden_size or tuple(mask.shape) != (batch, tokens):
                raise ValueError("Hidden tap and mask geometry mismatch")
            normalized = norm(hidden)
            key = key_projection(normalized).view(batch, tokens, self.kv_heads, self.head_dim).transpose(1, 2)
            value = value_projection(normalized).view(batch, tokens, self.kv_heads, self.head_dim).transpose(1, 2)
            keys.append(key.contiguous())
            values.append(value.contiguous())
        return PublicMemory(tuple(keys), tuple(values), mask.bool()).validate(self.layer_count, self.kv_heads, self.head_dim)

    def freeze_norms(self):
        for norm in self.norms:
            norm.requires_grad_(False)


class PublicReader(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        active_layers: Sequence[int],
        query_heads: int = 32,
        kv_heads: int = 8,
        head_dim: int = 128,
        max_gate: float = 1.0,
        gate_init: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        if query_heads % kv_heads:
            raise ValueError("Public query heads must be divisible by KV heads")
        self.hidden_size = int(hidden_size)
        self.active_layers = tuple(int(x) for x in active_layers)
        self.query_heads = int(query_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.groups = self.query_heads // self.kv_heads
        self.max_gate = float(max_gate)
        count = len(self.active_layers)
        query_width = self.query_heads * self.head_dim
        self.norms = nn.ModuleList([IndependentRMSNorm(hidden_size, eps) for _ in range(count)])
        self.query_projections = nn.ModuleList([nn.Linear(hidden_size, query_width, bias=False) for _ in range(count)])
        self.output_projections = nn.ModuleList([nn.Linear(query_width, hidden_size, bias=False) for _ in range(count)])
        ratio = max(-0.999, min(0.999, gate_init / max(max_gate, 1e-8)))
        self.gate_logits = nn.Parameter(torch.full((count,), float(torch.atanh(torch.tensor(ratio)))))
        for projection in self.output_projections:
            nn.init.normal_(projection.weight, mean=0.0, std=1e-3)
        self._memory = None
        self._pending = {}

    def gates(self):
        return self.max_gate * torch.tanh(self.gate_logits)

    def _read(self, hidden: torch.Tensor, local_layer: int) -> torch.Tensor:
        memory = self._memory
        key = memory.keys[local_layer]
        value = memory.values[local_layer]
        if key.shape[0] != hidden.shape[0]:
            if key.shape[0] == 1:
                key = key.expand(hidden.shape[0], -1, -1, -1)
                value = value.expand(hidden.shape[0], -1, -1, -1)
                mask = memory.mask.expand(hidden.shape[0], -1)
            else:
                raise ValueError("Public memory batch does not match receiver batch")
        else:
            mask = memory.mask
        batch, query_tokens, _ = hidden.shape
        query = self.query_projections[local_layer](self.norms[local_layer](hidden))
        query = query.view(batch, query_tokens, self.kv_heads, self.groups, self.head_dim)
        scores = torch.einsum("bthgd,bhmd->bthgm", query.float(), key.float()) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask[:, None, None, None, :], torch.finfo(scores.dtype).min)
        probability = scores.softmax(-1)
        readout = torch.einsum("bthgm,bhmd->bthgd", probability, value.float())
        readout = readout.reshape(batch, query_tokens, self.query_heads * self.head_dim).to(hidden.dtype)
        projected = self.output_projections[local_layer](readout)
        return projected * self.gates()[local_layer].to(projected.dtype)

    @contextmanager
    def inject(self, model, memory: PublicMemory):
        memory.validate(len(self.active_layers), self.kv_heads, self.head_dim)
        layers = decoder_layers(model)
        if max(self.active_layers) >= len(layers):
            raise ValueError("Public Reader layer exceeds receiver depth")
        self._memory = memory
        handles = []
        for local_layer, layer_index in enumerate(self.active_layers):
            layer = layers[layer_index]

            def pre_hook(module, args, kwargs, local_layer=local_layer):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                self._pending[local_layer] = self._read(hidden, local_layer)

            def post_hook(module, args, kwargs, output, local_layer=local_layer):
                delta = self._pending.pop(local_layer)
                if isinstance(output, tuple):
                    return (output[0] + delta,) + output[1:]
                return output + delta

            handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(layer.register_forward_hook(post_hook, with_kwargs=True))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._pending.clear()
            self._memory = None

    def metadata(self):
        return {
            "hidden_size": self.hidden_size,
            "active_layers": list(self.active_layers),
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "max_gate": self.max_gate,
        }


@contextmanager
def tap_collector(model, tap_layers: Iterable[int]):
    layers = decoder_layers(model)
    tap_layers = tuple(int(x) for x in tap_layers)
    captured = {}
    handles = []
    for layer_index in tap_layers:
        def hook(module, args, kwargs, output, layer_index=layer_index):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = hidden.detach()
        handles.append(layers[layer_index].register_forward_hook(hook, with_kwargs=True))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def capture_hidden_taps(model, model_inputs: dict, tap_layers: Sequence[int], use_cache: bool = False):
    with tap_collector(model, tap_layers) as captured:
        with torch.no_grad():
            output = model(**model_inputs, use_cache=use_cache, return_dict=True)
    missing = [layer for layer in tap_layers if layer not in captured]
    if missing:
        raise RuntimeError(f"Missing sender taps: {missing}")
    taps = tuple(captured[layer].detach() for layer in tap_layers)
    if any(tap.requires_grad or tap.grad_fn is not None for tap in taps):
        raise RuntimeError("Sender tap detachment failed")
    return taps, output
