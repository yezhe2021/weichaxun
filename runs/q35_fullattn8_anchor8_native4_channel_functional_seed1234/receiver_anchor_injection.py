from __future__ import annotations

from types import MethodType

import torch
from transformers.models.qwen3 import modeling_qwen3


class AnchorInjection:
    """Static external memory is never inserted into the Receiver self-cache."""

    def __init__(self, model, anchor_layers):
        self.model = model
        self.allowed_layers = tuple(int(x) for x in anchor_layers)
        self.external = {}
        self.external_mask = None
        self.usage = {}
        self.original = {}
        for index, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            self.original[index] = attention.forward
            attention.forward = MethodType(self._forward, attention)

    def set_memory(self, memory, mask=None):
        unexpected = set(memory) - set(self.allowed_layers)
        if unexpected:
            raise RuntimeError(f"external memory at non-anchor layers: {sorted(unexpected)}")
        self.external = dict(memory)
        self.external_mask = mask
        self.usage = {layer: 0 for layer in memory}

    def clear(self):
        self.external = {}
        self.external_mask = None
        self.usage = {}

    def assert_usage(self, expected_layers):
        expected = set(expected_layers)
        if set(self.usage) != expected or any(v == 0 for v in self.usage.values()):
            raise RuntimeError(f"anchor usage mismatch: {self.usage}, expected={sorted(expected)}")
        if set(self.external) - set(self.allowed_layers):
            raise RuntimeError("non-anchor external memory exists")

    def _forward(
        controller,
        attention,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query_states = attention.q_norm(attention.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, attention.layer_idx
            )
        if attention.layer_idx in controller.external:
            external_k, external_v = controller.external[attention.layer_idx]
            external_k = external_k.to(device=key_states.device, dtype=key_states.dtype)
            external_v = external_v.to(device=value_states.device, dtype=value_states.dtype)
            prefix = external_k.shape[-2]
            key_states = torch.cat((external_k, key_states), dim=-2)
            value_states = torch.cat((external_v, value_states), dim=-2)
            if attention_mask is not None:
                q_len = attention_mask.shape[-2]
                if controller.external_mask is None:
                    prefix_mask = attention_mask.new_zeros((*attention_mask.shape[:-1], prefix))
                else:
                    valid = controller.external_mask.to(attention_mask.device).bool()
                    bias = torch.zeros(valid.shape, device=valid.device, dtype=attention_mask.dtype)
                    bias.masked_fill_(~valid, torch.finfo(attention_mask.dtype).min)
                    prefix_mask = bias[:, None, None, :].expand(-1, 1, q_len, -1)
                attention_mask = torch.cat((prefix_mask, attention_mask), dim=-1)
            controller.usage[attention.layer_idx] += 1
        interface = modeling_qwen3.ALL_ATTENTION_FUNCTIONS.get_interface(
            attention.config._attn_implementation,
            modeling_qwen3.eager_attention_forward,
        )
        attn_output, attn_weights = interface(
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attention.training else attention.attention_dropout,
            scaling=attention.scaling,
            sliding_window=attention.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attention.o_proj(attn_output), attn_weights


def receiver_rope(model, pre_key, positions):
    position_ids = torch.tensor([positions], dtype=torch.long, device=pre_key.device)
    dummy = torch.empty(
        (1, len(positions), model.config.hidden_size),
        dtype=pre_key.dtype,
        device=pre_key.device,
    )
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    cos, sin = cos[0][None, :, None, :], sin[0][None, :, None, :]
    half = pre_key.shape[-1] // 2
    rotated = torch.cat((-pre_key[..., half:], pre_key[..., :half]), dim=-1)
    return pre_key * cos + rotated * sin


def memory_dict(model, layers, pre_key, value, positions):
    post_key = receiver_rope(model, pre_key, positions)
    return {
        int(layer): (
            post_key[slot].permute(1, 0, 2).unsqueeze(0),
            value[slot].permute(1, 0, 2).unsqueeze(0),
        )
        for slot, layer in enumerate(layers)
    }
