import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2a_common import get_layers


def identity_head_logits(receiver_heads, sender_heads):
    logits = torch.full((receiver_heads, sender_heads), -6.0)
    for receiver_head in range(receiver_heads):
        source = int(round(receiver_head * max(1, sender_heads - 1) / max(1, receiver_heads - 1)))
        logits[receiver_head, source] = 6.0
    return logits


class LlamaSpecificExternalReader(nn.Module):
    """Read raw Llama Native-KV from a frozen Qwen residual stream."""

    def __init__(
        self,
        receiver,
        sender_layers,
        sender_kv_heads,
        sender_head_dim,
        variant="minimal_reader",
        top_k=2,
        query_rank=32,
        output_rank=32,
        max_gate=0.5,
        gate_init=0.02,
    ):
        super().__init__()
        if variant not in {"minimal_reader", "routed_reader"}:
            raise ValueError(f"Unknown Reader variant: {variant}")
        self.variant = variant
        self.receiver_layers = len(get_layers(receiver))
        self.receiver_query_heads = int(receiver.config.num_attention_heads)
        self.receiver_kv_heads = int(receiver.config.num_key_value_heads)
        self.receiver_head_dim = int(receiver.config.head_dim)
        self.hidden_size = int(receiver.config.hidden_size)
        self.sender_layers = int(sender_layers)
        self.sender_kv_heads = int(sender_kv_heads)
        self.sender_head_dim = int(sender_head_dim)
        self.top_k = min(int(top_k), self.sender_layers)
        self.max_gate = float(max_gate)
        if self.sender_head_dim != self.receiver_head_dim:
            raise ValueError(
                "Experiment B v1 requires equal sender/receiver head dimensions; "
                f"got {self.sender_head_dim} and {self.receiver_head_dim}"
            )
        if self.receiver_query_heads % self.receiver_kv_heads != 0:
            raise ValueError("Receiver query heads must be divisible by receiver KV heads")

        receiver_depth = torch.linspace(0.0, 1.0, self.receiver_layers)
        sender_depth = torch.linspace(0.0, 1.0, self.sender_layers)
        distance = (receiver_depth[:, None] - sender_depth[None, :]).abs()
        self.register_buffer("relative_depth_distance", distance, persistent=True)
        fixed = distance.argmin(dim=-1)
        self.register_buffer("fixed_sender_layer", fixed, persistent=True)

        head_init = identity_head_logits(self.receiver_kv_heads, self.sender_kv_heads)
        if variant == "routed_reader":
            self.layer_logits = nn.Parameter(torch.zeros(self.receiver_layers, self.sender_layers))
            self.depth_strength = nn.Parameter(torch.full((self.receiver_layers, 1), 3.0))
            self.head_logits = nn.Parameter(
                head_init.unsqueeze(0).repeat(self.receiver_layers, 1, 1)
            )
        else:
            fixed_head = head_init.softmax(dim=-1).unsqueeze(0).repeat(
                self.receiver_layers, 1, 1
            )
            self.register_buffer("fixed_head_weights", fixed_head, persistent=True)

        self.query_down = nn.ModuleList(
            [nn.Linear(self.hidden_size, query_rank, bias=False) for _ in range(self.receiver_layers)]
        )
        self.query_up = nn.ModuleList(
            [nn.Linear(query_rank, self.hidden_size, bias=False) for _ in range(self.receiver_layers)]
        )
        self.output_down = nn.ModuleList(
            [nn.Linear(self.hidden_size, output_rank, bias=False) for _ in range(self.receiver_layers)]
        )
        self.output_up = nn.ModuleList(
            [nn.Linear(output_rank, self.hidden_size, bias=False) for _ in range(self.receiver_layers)]
        )
        for module in (*self.query_up, *self.output_up):
            nn.init.zeros_(module.weight)

        init = max(-0.999, min(0.999, gate_init / max(self.max_gate, 1e-8)))
        gate_logit = float(torch.atanh(torch.tensor(init)))
        self.gate_logits = nn.Parameter(torch.full((self.receiver_layers,), gate_logit))
        self._memory = None
        self._pending = {}
        self._diagnostics = None

    def gates(self):
        return self.max_gate * torch.tanh(self.gate_logits)

    def layer_routing(self):
        if self.variant == "minimal_reader":
            weights = torch.zeros(
                self.receiver_layers,
                self.sender_layers,
                device=self.fixed_sender_layer.device,
            )
            return weights.scatter(1, self.fixed_sender_layer[:, None], 1.0)
        scores = self.layer_logits - F.softplus(self.depth_strength) * self.relative_depth_distance
        indices = scores.topk(self.top_k, dim=-1).indices
        selected = scores.gather(-1, indices).softmax(dim=-1)
        return torch.zeros_like(scores).scatter(-1, indices, selected)

    def head_routing(self):
        if self.variant == "minimal_reader":
            return self.fixed_head_weights
        return self.head_logits.softmax(dim=-1)

    def _mapped_kv(self, source_layer, receiver_layer, dtype):
        key = self._memory["keys"][source_layer]
        value = self._memory["values"][source_layer]
        if key.shape[0] != self.sender_kv_heads or value.shape[0] != self.sender_kv_heads:
            raise ValueError(f"Unexpected sender head count at layer {source_layer}")
        head_weights = self.head_routing()[receiver_layer].to(key.device)
        mapped_key = torch.einsum("rs,std->rtd", head_weights, key.float()).to(dtype)
        mapped_value = torch.einsum("rs,std->rtd", head_weights, value.float()).to(dtype)
        return mapped_key, mapped_value

    def _external_output(self, attention, hidden_states, receiver_layer):
        batch, query_length, _ = hidden_states.shape
        head_dim = attention.head_dim
        query = attention.q_norm(
            attention.q_proj(hidden_states).view(batch, query_length, -1, head_dim)
        ).transpose(1, 2)
        query_flat = query.transpose(1, 2).reshape(batch, query_length, -1).contiguous()
        query_delta = self.query_up[receiver_layer](
            F.silu(self.query_down[receiver_layer](query_flat.float()))
        )
        query = (query_flat + query_delta.to(query_flat.dtype)).view(
            batch, query_length, self.receiver_query_heads, head_dim
        ).transpose(1, 2)

        layer_weights = self.layer_routing()[receiver_layer]
        selected_layers = torch.nonzero(layer_weights > 0, as_tuple=False).flatten()
        groups = self.receiver_query_heads // self.receiver_kv_heads
        readout = None
        aggregate_probability = None
        for source_layer in selected_layers.tolist():
            key, value = self._mapped_kv(source_layer, receiver_layer, query.dtype)
            key = key.unsqueeze(0).expand(batch, -1, -1, -1).repeat_interleave(groups, dim=1)
            value = value.unsqueeze(0).expand(batch, -1, -1, -1).repeat_interleave(groups, dim=1)
            scores = torch.matmul(query.float(), key.transpose(-1, -2).float()) / math.sqrt(head_dim)
            probability = scores.softmax(dim=-1)
            source_readout = torch.matmul(probability, value.float())
            weight = layer_weights[source_layer]
            readout = source_readout * weight if readout is None else readout + source_readout * weight
            aggregate_probability = (
                probability * weight
                if aggregate_probability is None
                else aggregate_probability + probability * weight
            )

        readout = readout.to(hidden_states.dtype).transpose(1, 2).reshape(
            batch, query_length, self.hidden_size
        ).contiguous()
        projected = attention.o_proj(readout)
        correction = self.output_up[receiver_layer](
            F.silu(self.output_down[receiver_layer](projected.float()))
        )
        projected = projected + correction.to(projected.dtype)
        gate = self.gates()[receiver_layer].to(projected.dtype)
        external = gate * projected

        if self._diagnostics is not None:
            slot = self._diagnostics.setdefault(
                str(receiver_layer), {"calls": 0, "readout_norm": 0.0, "delta_norm": 0.0}
            )
            slot["calls"] += 1
            slot["readout_norm"] += float(projected.detach().float().norm(dim=-1).mean().cpu())
            slot["delta_norm"] += float(external.detach().float().norm(dim=-1).mean().cpu())
            slot["gate"] = float(gate.detach().float().cpu())
            slot["query_delta_norm"] = float(
                query_delta.detach().float().norm(dim=-1).mean().cpu()
            )
            slot["selected_sender_layers"] = selected_layers.detach().cpu().tolist()
            slot["selected_layer_weights"] = layer_weights[selected_layers].detach().cpu().tolist()
            slot["attention_entropy"] = float(
                (
                    -(aggregate_probability * aggregate_probability.clamp_min(1e-8).log())
                    .sum(dim=-1)
                    .mean()
                    / math.log(max(2, aggregate_probability.shape[-1]))
                )
                .detach()
                .cpu()
            )
            answer_mask = self._memory.get("answer_token_mask")
            if answer_mask is not None and answer_mask.numel() == aggregate_probability.shape[-1]:
                slot["target_attention_mass"] = float(
                    aggregate_probability[..., answer_mask].sum(dim=-1).mean().detach().cpu()
                )
            if self._diagnostics.get("_capture_training_tensors", False):
                query_index = int(self._diagnostics.get("_capture_query_index", query_length - 1))
                query_index = max(0, min(query_length - 1, query_index))
                route = aggregate_probability[:, :, query_index, :]
                slot["route_tensor"] = route
                slot["readout_tensor"] = projected[:, query_index, :]
                slot["route_entropy_tensor"] = (
                    -(route * route.clamp_min(1e-8).log()).sum(dim=-1).mean()
                )
                if answer_mask is not None and answer_mask.numel() == route.shape[-1]:
                    slot["target_mass_tensor"] = route[..., answer_mask].sum(dim=-1).mean()
        return external

    @contextmanager
    def inject(self, model, memory, diagnostics=None):
        if len(memory["keys"]) != self.sender_layers:
            raise ValueError(
                f"Expected {self.sender_layers} sender layers, got {len(memory['keys'])}"
            )
        self._memory = memory
        self._diagnostics = diagnostics
        handles = []
        for receiver_layer, layer in enumerate(get_layers(model)):
            attention = layer.self_attn

            def pre_hook(module, args, kwargs, receiver_layer=receiver_layer):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                self._pending[receiver_layer] = self._external_output(
                    module, hidden, receiver_layer
                )

            def post_hook(module, args, kwargs, output, receiver_layer=receiver_layer):
                external = self._pending.pop(receiver_layer)
                if isinstance(output, tuple):
                    return (output[0] + external,) + output[1:]
                return output + external

            handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(attention.register_forward_hook(post_hook, with_kwargs=True))
        try:
            yield diagnostics
        finally:
            for handle in handles:
                handle.remove()
            self._pending.clear()
            self._memory = None
            self._diagnostics = None

    def routing_diagnostics(self):
        layer_weights = self.layer_routing().detach().cpu()
        head_weights = self.head_routing().detach().cpu()
        rows = []
        for receiver_layer in range(self.receiver_layers):
            selected = torch.nonzero(layer_weights[receiver_layer] > 0, as_tuple=False).flatten()
            rows.append(
                {
                    "receiver_layer": receiver_layer,
                    "sender_layers": selected.tolist(),
                    "layer_weights": layer_weights[receiver_layer, selected].tolist(),
                    "head_mapping": head_weights[receiver_layer].tolist(),
                    "gate": float(self.gates()[receiver_layer].detach().cpu()),
                }
            )
        return rows


__all__ = ["LlamaSpecificExternalReader"]
