from __future__ import annotations

from types import MethodType

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5


class Q35AnchorInjection:
    """External Anchor8 is static and separate from Qwen3.5 HybridCache."""

    def __init__(self, model, layers):
        self.model = model
        self.layers = tuple(int(x) for x in layers)
        self.external = {}
        self.external_mask = None
        self.usage = {}
        for index in self.layers:
            attention = model.model.layers[index].self_attn
            attention.forward = MethodType(self._forward, attention)

    def set_memory(self, memory, mask=None):
        if set(memory) - set(self.layers):
            raise RuntimeError(f"external memory at non-Full-Attention layers: {sorted(set(memory)-set(self.layers))}")
        self.external = dict(memory)
        self.external_mask = mask
        self.usage = {layer: 0 for layer in memory}

    def clear(self):
        self.external, self.external_mask, self.usage = {}, None, {}

    def assert_usage(self, expected):
        if set(self.usage) != set(expected) or any(value == 0 for value in self.usage.values()):
            raise RuntimeError(f"Qwen3.5 Anchor8 usage mismatch: {self.usage}")

    def _forward(
        controller, attention, hidden_states, position_embeddings,
        attention_mask, past_key_values=None, **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query_states, gate = torch.chunk(
            attention.q_proj(hidden_states).view(*input_shape, -1, attention.head_dim * 2),
            2, dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = attention.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen3_5.apply_rotary_pos_emb(
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
                attention_mask = torch.cat((prefix_mask, attention_mask), -1)
            controller.usage[attention.layer_idx] += 1
        interface = modeling_qwen3_5.ALL_ATTENTION_FUNCTIONS.get_interface(
            attention.config._attn_implementation,
            modeling_qwen3_5.eager_attention_forward,
        )
        attn_output, attn_weights = interface(
            attention, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not attention.training else attention.attention_dropout,
            scaling=attention.scaling, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return attention.o_proj(attn_output), attn_weights


def post_rope(model, pre_key, positions):
    length = len(positions)
    position_ids = torch.tensor([positions], device=pre_key.device)
    position_ids = position_ids[None].expand(4, 1, length)[1:]
    dummy = torch.empty(
        1, length, model.config.hidden_size,
        dtype=pre_key.dtype, device=pre_key.device,
    )
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    output = []
    for layer in range(pre_key.shape[0]):
        key = pre_key[layer].permute(1, 0, 2).unsqueeze(0)
        query = torch.zeros(1, 16, length, 256, dtype=key.dtype, device=key.device)
        _, rotated = modeling_qwen3_5.apply_rotary_pos_emb(query, key, cos, sin)
        output.append(rotated)
    return output


def memory_dict(model, layers, pre_key, value, positions):
    rotated = post_rope(model, pre_key, positions)
    return {
        int(layer): (
            rotated[slot], value[slot].permute(1, 0, 2).unsqueeze(0)
        )
        for slot, layer in enumerate(layers)
    }
