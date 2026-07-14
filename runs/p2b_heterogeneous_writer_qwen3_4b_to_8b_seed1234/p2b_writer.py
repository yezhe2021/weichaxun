import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_layer_candidates(sender_layers, receiver_layers, width=3):
    candidates = []
    for receiver_index in range(receiver_layers):
        position = receiver_index * max(1, sender_layers - 1) / max(1, receiver_layers - 1)
        center = int(round(position))
        values = []
        radius = max(1, width // 2)
        for offset in range(-radius, radius + 1):
            values.append(max(0, min(sender_layers - 1, center + offset)))
        while len(values) < width:
            values.append(center)
        candidates.append(values[:width])
    return torch.tensor(candidates, dtype=torch.long)


def initialize_head_logits(receiver_heads, sender_heads):
    logits = torch.full((receiver_heads, sender_heads), -6.0)
    for receiver_head in range(receiver_heads):
        source = int(round(receiver_head * max(1, sender_heads - 1) / max(1, receiver_heads - 1)))
        logits[receiver_head, source] = 6.0
    return logits


class HeterogeneousNativeKVWriter(nn.Module):
    def __init__(
        self,
        sender_layers,
        sender_heads,
        sender_head_dim,
        receiver_layers,
        receiver_heads,
        receiver_head_dim,
        layer_width=3,
    ):
        super().__init__()
        self.sender_layers = int(sender_layers)
        self.sender_heads = int(sender_heads)
        self.sender_head_dim = int(sender_head_dim)
        self.receiver_layers = int(receiver_layers)
        self.receiver_heads = int(receiver_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.layer_width = int(layer_width)
        self.register_buffer(
            "layer_candidates",
            normalized_layer_candidates(self.sender_layers, self.receiver_layers, self.layer_width),
            persistent=True,
        )
        layer_init = torch.full((self.receiver_layers, self.layer_width), -4.0)
        layer_init[:, self.layer_width // 2] = 4.0
        self.key_layer_logits = nn.Parameter(layer_init.clone())
        self.value_layer_logits = nn.Parameter(layer_init.clone())
        head_init = initialize_head_logits(self.receiver_heads, self.sender_heads)
        self.key_head_logits = nn.Parameter(head_init.unsqueeze(0).repeat(self.receiver_layers, 1, 1))
        self.value_head_logits = nn.Parameter(head_init.unsqueeze(0).repeat(self.receiver_layers, 1, 1))
        self.key_projection = nn.ModuleList(
            [nn.Linear(self.sender_head_dim, self.receiver_head_dim, bias=False) for _ in range(self.receiver_layers)]
        )
        self.value_projection = nn.ModuleList(
            [nn.Linear(self.sender_head_dim, self.receiver_head_dim, bias=False) for _ in range(self.receiver_layers)]
        )
        self.key_log_scale = nn.Parameter(torch.zeros(self.receiver_layers, self.receiver_heads))
        self.value_log_scale = nn.Parameter(torch.zeros(self.receiver_layers, self.receiver_heads))
        self._initialize_projections()

    def _initialize_projections(self):
        for key_projection, value_projection in zip(self.key_projection, self.value_projection):
            if self.sender_head_dim == self.receiver_head_dim:
                nn.init.eye_(key_projection.weight)
                nn.init.eye_(value_projection.weight)
            else:
                nn.init.orthogonal_(key_projection.weight)
                nn.init.orthogonal_(value_projection.weight)

    def _map_one(self, source, receiver_layer, layer_logits, head_logits, projection, log_scale):
        indices = self.layer_candidates[receiver_layer].tolist()
        layers = torch.stack([source[index].float() for index in indices], dim=0)
        layer_weights = layer_logits[receiver_layer].softmax(dim=-1)
        mixed_layer = torch.einsum("k,khtd->htd", layer_weights, layers)
        head_weights = head_logits[receiver_layer].softmax(dim=-1)
        mixed_heads = torch.einsum("rs,std->rtd", head_weights, mixed_layer)
        mapped = projection[receiver_layer](mixed_heads)
        return mapped * log_scale[receiver_layer].exp().view(-1, 1, 1)

    def forward(self, memory, output_dtype=None):
        if len(memory["keys"]) != self.sender_layers:
            raise ValueError(f"Expected {self.sender_layers} sender layers, got {len(memory['keys'])}")
        keys = []
        values = []
        for receiver_layer in range(self.receiver_layers):
            key = self._map_one(
                memory["keys"], receiver_layer, self.key_layer_logits, self.key_head_logits,
                self.key_projection, self.key_log_scale,
            )
            value = self._map_one(
                memory["values"], receiver_layer, self.value_layer_logits, self.value_head_logits,
                self.value_projection, self.value_log_scale,
            )
            keys.append(key.to(dtype=output_dtype or key.dtype))
            values.append(value.to(dtype=output_dtype or value.dtype))
        output = {"keys": keys, "values": values}
        if "answer_token_mask" in memory:
            output["answer_token_mask"] = memory["answer_token_mask"]
        return output

    def regularization(self):
        key_layer_entropy = -(self.key_layer_logits.softmax(-1) * self.key_layer_logits.log_softmax(-1)).sum(-1).mean()
        value_layer_entropy = -(self.value_layer_logits.softmax(-1) * self.value_layer_logits.log_softmax(-1)).sum(-1).mean()
        scale_penalty = self.key_log_scale.square().mean() + self.value_log_scale.square().mean()
        return 0.01 * (key_layer_entropy + value_layer_entropy) + scale_penalty


def shape_only_memory(memory, receiver_layers, receiver_heads, receiver_head_dim, output_dtype=None):
    sender_layers = len(memory["keys"])
    keys = []
    values = []
    for receiver_layer in range(receiver_layers):
        sender_layer = int(round(receiver_layer * max(1, sender_layers - 1) / max(1, receiver_layers - 1)))
        layer_keys = memory["keys"][sender_layer]
        layer_values = memory["values"][sender_layer]
        sender_heads = layer_keys.shape[0]
        head_indices = [
            int(round(head * max(1, sender_heads - 1) / max(1, receiver_heads - 1)))
            for head in range(receiver_heads)
        ]
        layer_keys = layer_keys[head_indices]
        layer_values = layer_values[head_indices]
        if layer_keys.shape[-1] < receiver_head_dim:
            padding = receiver_head_dim - layer_keys.shape[-1]
            layer_keys = F.pad(layer_keys, (0, padding))
            layer_values = F.pad(layer_values, (0, padding))
        else:
            layer_keys = layer_keys[..., :receiver_head_dim]
            layer_values = layer_values[..., :receiver_head_dim]
        keys.append(layer_keys.to(dtype=output_dtype or layer_keys.dtype))
        values.append(layer_values.to(dtype=output_dtype or layer_values.dtype))
    output = {"keys": keys, "values": values}
    if "answer_token_mask" in memory:
        output["answer_token_mask"] = memory["answer_token_mask"]
    return output
