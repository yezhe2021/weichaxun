from __future__ import annotations

from typing import Any

import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


class PreRoPECapture:
    """Capture Qwen3 K after k_norm but before RoPE, and native V after v_proj."""

    def __init__(self, model, expected_layers: int):
        self.expected_layers = expected_layers
        self.pre_key: dict[int, torch.Tensor] = {}
        self.value: dict[int, torch.Tensor] = {}
        self.handles = []
        if len(model.model.layers) != expected_layers:
            raise RuntimeError(f"model has {len(model.model.layers)} layers, expected {expected_layers}")
        for index, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.register_forward_pre_hook(self._hook(index), with_kwargs=True)
            )

    def _hook(self, index: int):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                raise RuntimeError("attention hook did not receive hidden_states")
            batch, tokens, _ = hidden.shape
            shape = (batch, tokens, -1, module.head_dim)
            key = module.k_norm(module.k_proj(hidden).view(shape))
            value = module.v_proj(hidden).view(shape)
            self.pre_key[index] = key[0].detach().to("cpu", dtype=torch.float16).clone()
            self.value[index] = value[0].detach().to("cpu", dtype=torch.float16).clone()

        return hook

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.pre_key) != self.expected_layers or len(self.value) != self.expected_layers:
            raise RuntimeError(
                f"incomplete capture: K={len(self.pre_key)} V={len(self.value)} expected={self.expected_layers}"
            )
        key = torch.stack([self.pre_key[index] for index in range(self.expected_layers)])
        value = torch.stack([self.value[index] for index in range(self.expected_layers)])
        return key, value

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def capture_full_native(model, token_ids: list[int], expected_layers: int, device: torch.device):
    capture = PreRoPECapture(model, expected_layers)
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    positions = torch.arange(len(token_ids), device=device).unsqueeze(0)
    try:
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=positions,
            use_cache=True,
        )
        pre_key, value = capture.result()
    finally:
        capture.close()
    return pre_key, value, output.past_key_values, output.logits[:, -1].detach().float().cpu()


def apply_receiver_rope(model, pre_key: torch.Tensor, positions: list[int] | None = None) -> torch.Tensor:
    """Apply the frozen Receiver's own Qwen3 RoPE without detaching Writer output."""
    if pre_key.ndim != 4:
        raise ValueError(f"pre_key must be [layers,tokens,heads,dim], got {tuple(pre_key.shape)}")
    token_count = pre_key.shape[1]
    positions = list(range(token_count)) if positions is None else list(positions)
    if len(positions) != token_count:
        raise ValueError("position count does not match KV token count")
    position_ids = torch.tensor([positions], dtype=torch.long, device=pre_key.device)
    dummy = torch.empty(
        1, token_count, model.config.hidden_size,
        dtype=pre_key.dtype, device=pre_key.device,
    )
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    key = pre_key.permute(0, 2, 1, 3)
    _, rotated = apply_rotary_pos_emb(key, key, cos, sin)
    return rotated.permute(0, 2, 1, 3)


def differentiable_cache(model, pre_key: torch.Tensor, value: torch.Tensor) -> DynamicCache:
    """Build a DynamicCache while preserving trajectory-loss → KV → Writer autograd."""
    if pre_key.shape != value.shape:
        raise ValueError(f"K/V shape mismatch: {tuple(pre_key.shape)} vs {tuple(value.shape)}")
    post_key = apply_receiver_rope(model, pre_key)
    items = [
        (
            post_key[layer].permute(1, 0, 2).unsqueeze(0),
            value[layer].permute(1, 0, 2).unsqueeze(0),
        )
        for layer in range(pre_key.shape[0])
    ]
    return DynamicCache(ddp_cache_data=items, config=model.config)


def official_cache_tensors(cache) -> tuple[torch.Tensor, torch.Tensor]:
    keys, values = [], []
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            keys.append(layer.keys[0].permute(1, 0, 2).detach().cpu())
            values.append(layer.values[0].permute(1, 0, 2).detach().cpu())
    else:
        for key, value in cache:
            keys.append(key[0].permute(1, 0, 2).detach().cpu())
            values.append(value[0].permute(1, 0, 2).detach().cpu())
    return torch.stack(keys), torch.stack(values)


def validate_native_shapes(key: torch.Tensor, value: torch.Tensor, layers: int, tokens: int, heads: int, dim: int):
    expected = (layers, tokens, heads, dim)
    if tuple(key.shape) != expected or tuple(value.shape) != expected:
        raise RuntimeError(f"native KV shape mismatch: K={tuple(key.shape)} V={tuple(value.shape)} expected={expected}")
