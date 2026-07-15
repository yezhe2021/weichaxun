import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2c1_writer import shape_only_memory


def initialize_head_logits(receiver_heads, sender_heads):
    logits = torch.full((receiver_heads, sender_heads), -6.0)
    for receiver_head in range(receiver_heads):
        source = int(round(receiver_head * max(1, sender_heads - 1) / max(1, receiver_heads - 1)))
        logits[receiver_head, source] = 6.0
    return logits


def initialize_projection(receiver_dim, sender_dim):
    weight = torch.zeros(receiver_dim, sender_dim)
    diagonal = min(receiver_dim, sender_dim)
    weight[:diagonal, :diagonal] = torch.eye(diagonal)
    return weight


class EnhancedGlobalNativeKVWriter(nn.Module):
    def __init__(
        self,
        sender_layers,
        sender_heads,
        sender_head_dim,
        receiver_layers,
        receiver_heads,
        receiver_head_dim,
        top_k=6,
        adapter_mode="per_head",
        adapter_rank=32,
        teacher_k_rms=None,
    ):
        super().__init__()
        if adapter_mode not in {"shared_full", "per_head"}:
            raise ValueError(f"Unsupported adapter_mode={adapter_mode}")
        self.sender_layers = int(sender_layers)
        self.sender_heads = int(sender_heads)
        self.sender_head_dim = int(sender_head_dim)
        self.receiver_layers = int(receiver_layers)
        self.receiver_heads = int(receiver_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.top_k = min(int(top_k), self.sender_layers)
        self.adapter_mode = adapter_mode
        self.adapter_rank = int(adapter_rank)
        sender_depth = torch.linspace(0.0, 1.0, self.sender_layers)
        receiver_depth = torch.linspace(0.0, 1.0, self.receiver_layers)
        self.register_buffer(
            "relative_depth_distance",
            (receiver_depth[:, None] - sender_depth[None, :]).abs(),
            persistent=True,
        )
        self.key_layer_logits = nn.Parameter(torch.zeros(self.receiver_layers, self.sender_layers))
        self.value_layer_logits = nn.Parameter(torch.zeros(self.receiver_layers, self.sender_layers))
        self.key_depth_strength = nn.Parameter(torch.full((self.receiver_layers, 1), 3.0))
        self.value_depth_strength = nn.Parameter(torch.full((self.receiver_layers, 1), 3.0))
        head_init = initialize_head_logits(self.receiver_heads, self.sender_heads)
        self.key_head_logits = nn.Parameter(head_init.unsqueeze(0).repeat(self.receiver_layers, 1, 1))
        self.value_head_logits = nn.Parameter(head_init.unsqueeze(0).repeat(self.receiver_layers, 1, 1))

        base = initialize_projection(self.receiver_head_dim, self.sender_head_dim)
        if self.adapter_mode == "shared_full":
            self.key_base_projection = nn.Parameter(base.unsqueeze(0).repeat(self.receiver_layers, 1, 1))
            self.value_base_projection = nn.Parameter(base.unsqueeze(0).repeat(self.receiver_layers, 1, 1))
            self.register_parameter("key_down", None)
            self.register_parameter("key_up", None)
            self.register_parameter("value_down", None)
            self.register_parameter("value_up", None)
        else:
            repeated = base.unsqueeze(0).unsqueeze(0).repeat(
                self.receiver_layers, self.receiver_heads, 1, 1
            )
            self.key_base_projection = nn.Parameter(repeated.clone())
            self.value_base_projection = nn.Parameter(repeated.clone())
            self.key_down = nn.Parameter(
                torch.empty(self.receiver_layers, self.receiver_heads, self.adapter_rank, self.receiver_head_dim)
            )
            self.key_up = nn.Parameter(
                torch.zeros(self.receiver_layers, self.receiver_heads, self.receiver_head_dim, self.adapter_rank)
            )
            self.value_down = nn.Parameter(
                torch.empty(self.receiver_layers, self.receiver_heads, self.adapter_rank, self.receiver_head_dim)
            )
            self.value_up = nn.Parameter(
                torch.zeros(self.receiver_layers, self.receiver_heads, self.receiver_head_dim, self.adapter_rank)
            )
            nn.init.kaiming_uniform_(self.key_down, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.value_down, a=math.sqrt(5))

        self.key_log_scale = nn.Parameter(torch.zeros(self.receiver_layers, self.receiver_heads))
        self.value_log_scale = nn.Parameter(torch.zeros(self.receiver_layers, self.receiver_heads))
        if teacher_k_rms is None:
            teacher_k_rms = torch.ones(self.receiver_layers, self.receiver_heads)
        if tuple(teacher_k_rms.shape) != (self.receiver_layers, self.receiver_heads):
            raise ValueError(f"Unexpected teacher_k_rms shape {tuple(teacher_k_rms.shape)}")
        self.register_buffer("teacher_k_rms", teacher_k_rms.float().clamp_min(1e-6), persistent=True)
        self.key_routing_dense = False
        self.key_routing_noise = False
        self.value_routing_dense = False
        self.value_routing_noise = False

    def set_routing_mode(self, dense=False, noise=False, part="joint"):
        if part not in {"key", "value", "joint"}:
            raise ValueError(part)
        if part in {"key", "joint"}:
            self.key_routing_dense = bool(dense)
            self.key_routing_noise = bool(noise)
        if part in {"value", "joint"}:
            self.value_routing_dense = bool(dense)
            self.value_routing_noise = bool(noise)

    def _routing_weights(self, kind, deterministic=False):
        if kind == "key":
            logits = self.key_layer_logits
            strength = F.softplus(self.key_depth_strength)
            dense = self.key_routing_dense
            noise = self.key_routing_noise
        else:
            logits = self.value_layer_logits
            strength = F.softplus(self.value_depth_strength)
            dense = self.value_routing_dense
            noise = self.value_routing_noise
        scores = logits - strength * self.relative_depth_distance
        if self.training and noise and not deterministic:
            uniform = torch.rand_like(scores).clamp_(1e-6, 1.0 - 1e-6)
            scores = scores + 0.1 * (-torch.log(-torch.log(uniform)))
        if dense and not deterministic:
            return scores.softmax(dim=-1)
        values, indices = scores.topk(self.top_k, dim=-1)
        selected = values.softmax(dim=-1)
        weights = torch.zeros_like(scores)
        return weights.scatter(-1, indices, selected)

    def _map_layer(self, source_stack, receiver_layer, kind):
        routing = self._routing_weights(kind)[receiver_layer]
        mixed_layer = torch.einsum("s,shtd->htd", routing, source_stack)
        head_logits = self.key_head_logits if kind == "key" else self.value_head_logits
        head_weights = head_logits[receiver_layer].softmax(dim=-1)
        mixed_heads = torch.einsum("rs,std->rtd", head_weights, mixed_layer)
        base_projection = self.key_base_projection if kind == "key" else self.value_base_projection
        if self.adapter_mode == "shared_full":
            mapped = torch.einsum("htd,od->hto", mixed_heads, base_projection[receiver_layer])
        else:
            mapped = torch.einsum("htd,hod->hto", mixed_heads, base_projection[receiver_layer])
            down = self.key_down if kind == "key" else self.value_down
            up = self.key_up if kind == "key" else self.value_up
            low_rank = torch.einsum("hto,hro->htr", mapped, down[receiver_layer])
            residual = torch.einsum("htr,hor->hto", F.silu(low_rank), up[receiver_layer])
            mapped = mapped + residual
        if kind == "key":
            current_rms = mapped.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
            mapped = mapped * (self.teacher_k_rms[receiver_layer] / current_rms.detach()).view(-1, 1, 1)
            scale = self.key_log_scale[receiver_layer]
        else:
            scale = self.value_log_scale[receiver_layer]
        return mapped * scale.exp().view(-1, 1, 1)

    def forward(self, memory, output_dtype=None):
        if len(memory["keys"]) != self.sender_layers:
            raise ValueError(f"Expected {self.sender_layers} sender layers, got {len(memory['keys'])}")
        key_stack = torch.stack([tensor.float() for tensor in memory["keys"]], dim=0)
        value_stack = torch.stack([tensor.float() for tensor in memory["values"]], dim=0)
        keys = []
        values = []
        for receiver_layer in range(self.receiver_layers):
            key = self._map_layer(key_stack, receiver_layer, "key")
            value = self._map_layer(value_stack, receiver_layer, "value")
            keys.append(key.to(dtype=output_dtype or key.dtype))
            values.append(value.to(dtype=output_dtype or value.dtype))
        output = {"keys": keys, "values": values}
        if "answer_token_mask" in memory:
            output["answer_token_mask"] = memory["answer_token_mask"]
        return output

    def routing_diagnostics(self):
        rows = []
        for kind in ("key", "value"):
            weights = self._routing_weights(kind, deterministic=True).detach().cpu()
            for receiver_layer in range(self.receiver_layers):
                nonzero = torch.nonzero(weights[receiver_layer] > 0, as_tuple=False).flatten()
                rows.append(
                    {
                        "kind": kind,
                        "receiver_layer": receiver_layer,
                        "sender_layers": nonzero.tolist(),
                        "weights": weights[receiver_layer, nonzero].tolist(),
                        "entropy": float(
                            -(weights[receiver_layer, nonzero] * weights[receiver_layer, nonzero].log()).sum()
                        ),
                        "mean_depth_distance": float(
                            (weights[receiver_layer] * self.relative_depth_distance[receiver_layer].cpu()).sum()
                        ),
                    }
                )
        return rows

    def key_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("key_"):
                yield parameter

    def value_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("value_"):
                yield parameter

    def set_trainable_part(self, part):
        if part not in {"key", "value", "joint"}:
            raise ValueError(part)
        for name, parameter in self.named_parameters():
            if part == "joint":
                parameter.requires_grad_(True)
            elif part == "key":
                parameter.requires_grad_(name.startswith("key_"))
            else:
                parameter.requires_grad_(name.startswith("value_"))


__all__ = ["EnhancedGlobalNativeKVWriter", "shape_only_memory"]
