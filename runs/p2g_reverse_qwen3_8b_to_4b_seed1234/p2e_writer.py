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


class StructurePreservingNativeKVWriter(nn.Module):
    """Map complete sender KV into the frozen receiver's per-layer KV interface."""

    def __init__(
        self,
        sender_layers,
        sender_heads,
        sender_head_dim,
        receiver_layers,
        receiver_heads,
        receiver_head_dim,
        top_k=6,
        adapter_rank=32,
        shared_routing=False,
        route_residual_scale=0.25,
        teacher_k_rms=None,
    ):
        super().__init__()
        self.sender_layers = int(sender_layers)
        self.sender_heads = int(sender_heads)
        self.sender_head_dim = int(sender_head_dim)
        self.receiver_layers = int(receiver_layers)
        self.receiver_heads = int(receiver_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.top_k = min(int(top_k), self.sender_layers)
        self.adapter_rank = int(adapter_rank)
        self.shared_routing = bool(shared_routing)
        self.route_residual_scale = float(route_residual_scale)

        sender_depth = torch.linspace(0.0, 1.0, self.sender_layers)
        receiver_depth = torch.linspace(0.0, 1.0, self.receiver_layers)
        self.register_buffer(
            "relative_depth_distance",
            (receiver_depth[:, None] - sender_depth[None, :]).abs(),
            persistent=True,
        )
        head_init = initialize_head_logits(self.receiver_heads, self.sender_heads)
        if self.shared_routing:
            self.shared_layer_logits = nn.Parameter(torch.zeros(self.receiver_layers, self.sender_layers))
            self.shared_depth_strength = nn.Parameter(torch.full((self.receiver_layers, 1), 3.0))
            self.key_layer_residual = nn.Parameter(torch.zeros(self.receiver_layers, self.sender_layers))
            self.value_layer_residual = nn.Parameter(torch.zeros(self.receiver_layers, self.sender_layers))
            self.shared_head_logits = nn.Parameter(
                head_init.unsqueeze(0).repeat(self.receiver_layers, 1, 1)
            )
            self.key_head_residual = nn.Parameter(
                torch.zeros(self.receiver_layers, self.receiver_heads, self.sender_heads)
            )
            self.value_head_residual = nn.Parameter(
                torch.zeros(self.receiver_layers, self.receiver_heads, self.sender_heads)
            )
        else:
            self.key_layer_logits = nn.Parameter(torch.zeros(self.receiver_layers, self.sender_layers))
            self.value_layer_logits = nn.Parameter(torch.zeros(self.receiver_layers, self.sender_layers))
            self.key_depth_strength = nn.Parameter(torch.full((self.receiver_layers, 1), 3.0))
            self.value_depth_strength = nn.Parameter(torch.full((self.receiver_layers, 1), 3.0))
            self.key_head_logits = nn.Parameter(
                head_init.unsqueeze(0).repeat(self.receiver_layers, 1, 1)
            )
            self.value_head_logits = nn.Parameter(
                head_init.unsqueeze(0).repeat(self.receiver_layers, 1, 1)
            )

        base = initialize_projection(self.receiver_head_dim, self.sender_head_dim)
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
        self.routing_dense = False

    def set_routing_mode(self, dense=False):
        self.routing_dense = bool(dense)

    def _layer_scores(self, kind):
        if self.shared_routing:
            main = self.shared_layer_logits - F.softplus(self.shared_depth_strength) * self.relative_depth_distance
            residual = self.key_layer_residual if kind == "key" else self.value_layer_residual
            return main, main + self.route_residual_scale * residual.tanh()
        logits = self.key_layer_logits if kind == "key" else self.value_layer_logits
        strength = self.key_depth_strength if kind == "key" else self.value_depth_strength
        scores = logits - F.softplus(strength) * self.relative_depth_distance
        return scores, scores

    def layer_routing(self, kind, deterministic=False):
        main_scores, scores = self._layer_scores(kind)
        if self.routing_dense and not deterministic:
            return scores.softmax(dim=-1)
        support_scores = main_scores if self.shared_routing else scores
        indices = support_scores.topk(self.top_k, dim=-1).indices
        selected_scores = scores.gather(-1, indices)
        weights = torch.zeros_like(scores)
        return weights.scatter(-1, indices, selected_scores.softmax(dim=-1))

    def head_routing(self, kind):
        if self.shared_routing:
            residual = self.key_head_residual if kind == "key" else self.value_head_residual
            logits = self.shared_head_logits + self.route_residual_scale * residual.tanh()
        else:
            logits = self.key_head_logits if kind == "key" else self.value_head_logits
        return logits.softmax(dim=-1)

    def _map_layer(self, source_stack, receiver_layer, kind):
        layer_weights = self.layer_routing(kind)[receiver_layer]
        mixed_layer = torch.einsum("s,shtd->htd", layer_weights, source_stack)
        head_weights = self.head_routing(kind)[receiver_layer]
        mixed_heads = torch.einsum("rs,std->rtd", head_weights, mixed_layer)
        base_projection = self.key_base_projection if kind == "key" else self.value_base_projection
        mapped = torch.einsum("htd,hod->hto", mixed_heads, base_projection[receiver_layer])
        down = self.key_down if kind == "key" else self.value_down
        up = self.key_up if kind == "key" else self.value_up
        low_rank = torch.einsum("hto,hro->htr", mapped, down[receiver_layer])
        residual = torch.einsum("htr,hor->hto", F.silu(low_rank), up[receiver_layer])
        mapped = mapped + residual
        if kind == "key":
            current_rms = mapped.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
            calibration = self.teacher_k_rms[receiver_layer] / current_rms.detach()
            mapped = mapped * calibration.view(-1, 1, 1)
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
            keys.append(self._map_layer(key_stack, receiver_layer, "key").to(output_dtype or torch.float32))
            values.append(self._map_layer(value_stack, receiver_layer, "value").to(output_dtype or torch.float32))
        output = {"keys": keys, "values": values}
        if "answer_token_mask" in memory:
            output["answer_token_mask"] = memory["answer_token_mask"]
        return output

    def routing_difference_tensors(self):
        key_layers = self.layer_routing("key", deterministic=True)
        value_layers = self.layer_routing("value", deterministic=True)
        key_heads = self.head_routing("key")
        value_heads = self.head_routing("value")
        return {
            "layer_l1": (key_layers - value_layers).abs().mean(),
            "head_l1": (key_heads - value_heads).abs().mean(),
            "layer_support_disagreement": (
                (key_layers > 0) != (value_layers > 0)
            ).float().mean(),
        }

    def routing_diagnostics(self):
        rows = []
        for kind in ("key", "value"):
            layer_weights = self.layer_routing(kind, deterministic=True).detach().cpu()
            head_weights = self.head_routing(kind).detach().cpu()
            for receiver_layer in range(self.receiver_layers):
                nonzero = torch.nonzero(layer_weights[receiver_layer] > 0, as_tuple=False).flatten()
                rows.append(
                    {
                        "kind": kind,
                        "receiver_layer": receiver_layer,
                        "sender_layers": nonzero.tolist(),
                        "layer_weights": layer_weights[receiver_layer, nonzero].tolist(),
                        "head_mapping": head_weights[receiver_layer].tolist(),
                        "mean_depth_distance": float(
                            (layer_weights[receiver_layer] * self.relative_depth_distance[receiver_layer].cpu()).sum()
                        ),
                    }
                )
        return rows


__all__ = ["StructurePreservingNativeKVWriter", "shape_only_memory"]
