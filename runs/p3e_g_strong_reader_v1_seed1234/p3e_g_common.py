import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from p3e_c1_common import LearnableCanonicalHeadReader


class RMSNorm(nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = float(eps)

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized * self.weight


class StrongOutputControl(nn.Module):
    def __init__(self, hidden_size, rank, old_gate, eps):
        super().__init__()
        if not 0.0 < float(old_gate) < 1.0:
            raise ValueError(f"Old scalar gate must be in (0,1), got {old_gate}")
        self.adapter_norm = RMSNorm(hidden_size, eps)
        self.adapter_down = nn.Linear(hidden_size, rank, bias=False)
        self.adapter_up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.orthogonal_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)
        self.gate_norm = RMSNorm(hidden_size, eps)
        self.gate_linear = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, math.log(float(old_gate) / (1.0 - float(old_gate))))
        self.old_gate = float(old_gate)

    def forward(self, projected, hidden):
        update = self.adapter_up(F.silu(self.adapter_down(self.adapter_norm(projected)))).to(projected.dtype)
        adapted = projected + update
        token_gate = torch.sigmoid(self.gate_linear(self.gate_norm(hidden))).to(projected.dtype)
        delta = token_gate * adapted
        return delta, {
            "projected": projected,
            "adapter_update": update,
            "adapted": adapted,
            "token_gate": token_gate,
            "delta": delta,
        }


class StrongCanonicalReader(nn.Module):
    def __init__(self, model, old_checkpoint, output_rank=128):
        super().__init__()
        metadata = old_checkpoint["reader_metadata"]
        self.selected_layers = list(metadata["selected_layers"])
        self.base = LearnableCanonicalHeadReader(
            model, self.selected_layers, metadata["rank"], metadata["gate_init"],
            metadata["top_k"], 0.25
        )
        self.base.load_state_dict(old_checkpoint["reader"])
        self.base.requires_grad_(False)
        hidden_size = int(model.config.hidden_size)
        eps = float(getattr(model.config, "rms_norm_eps", 1e-6))
        self.output_rank = int(output_rank)
        self.controls = nn.ModuleList([
            StrongOutputControl(hidden_size, output_rank, float(branch.gate.detach()), eps)
            for branch in self.base.branches
        ])
        self._memory, self._trace = None, None
        self._queries, self._hidden = {}, {}

    def new_parameters(self):
        return [parameter for control in self.controls for parameter in control.parameters()]

    def initial_equivalence_error(self):
        errors = []
        for branch, control in zip(self.base.branches, self.controls):
            reconstructed = torch.sigmoid(control.gate_linear.bias.detach()).item()
            errors.append(abs(reconstructed - float(branch.gate.detach())))
        return max(errors)

    def static_gate_values(self):
        return torch.stack([torch.sigmoid(control.gate_linear.bias.detach()).squeeze() for control in self.controls])

    def routes(self):
        return self.base.routes()

    def set_temperature(self, value):
        self.base.set_temperature(value)

    def forward_branch(self, local, q_states, hidden, keys, values, mask, native_o_proj):
        branch = self.base.branches[local]
        query = branch.query_adapter(branch.native_query(q_states)).float()
        scores = torch.einsum("bsqd,tcd->bsqct", query, keys.float()) / math.sqrt(branch.head_dim)
        scores = scores.masked_fill(~mask[None, None, None, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        canonical_readout = torch.einsum("bsqct,tcd->bsqcd", attention, values.float())
        route = branch.routing_weights()
        readout = torch.einsum("qc,bsqcd->bsqd", route, canonical_readout)
        flattened = readout.reshape(readout.shape[0], readout.shape[1], branch.query_heads * branch.head_dim)
        projected = native_o_proj(flattened.to(q_states.dtype))
        delta, details = self.controls[local](projected, hidden)
        details.update({"attention": attention, "route": route, "headwise_readout": readout})
        return delta, details

    @contextmanager
    def inject(self, model, memory, trace=None):
        expected = (len(self.selected_layers), 16, 128)
        keys = memory["keys"]
        if keys.ndim != 4 or (keys.shape[0], keys.shape[2], keys.shape[3]) != expected:
            raise RuntimeError(f"Strong Reader/memory mismatch: {tuple(keys.shape)}")
        self._memory, self._trace = memory, trace
        handles = []
        for local, layer_index in enumerate(self.selected_layers):
            attention = model.model.layers[layer_index].self_attn

            def q_hook(module, args, output, local=local):
                self._queries[local] = output

            def pre_hook(module, args, kwargs, local=local):
                self._hidden[local] = kwargs.get("hidden_states", args[0] if args else None)

            def output_hook(module, args, kwargs, output, local=local, layer_index=layer_index):
                if local not in self._queries or local not in self._hidden:
                    raise RuntimeError("Strong Reader hook did not capture Query/hidden")
                delta, details = self.forward_branch(
                    local, self._queries.pop(local), self._hidden.pop(local),
                    self._memory["keys"][local], self._memory["values"][local],
                    self._memory["mask"], module.o_proj,
                )
                if self._trace is not None:
                    self._trace.setdefault(layer_index, []).append(details)
                if isinstance(output, tuple):
                    return (output[0] + delta,) + output[1:]
                return output + delta

            handles.append(attention.q_norm.register_forward_hook(q_hook))
            handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(attention.register_forward_hook(output_hook, with_kwargs=True))
        try:
            yield trace
        finally:
            for handle in handles:
                handle.remove()
            self._queries.clear()
            self._hidden.clear()
            self._memory, self._trace = None, None

    def metadata(self):
        return {
            "experiment": "P3-E-G Strong Reader V1",
            "selected_layers": self.selected_layers,
            "canonical_memory": "[16,T,16,128]",
            "base_reader": "frozen_C1_QueryAdapter_and_head_routing",
            "native_o_proj": "frozen_4096_to_2560",
            "post_o_proj_adapter": f"RMSNorm_2560_to_{self.output_rank}_to_2560_residual",
            "token_gate": "RMSNorm_2560_to_1_sigmoid",
            "adapter_up_zero_initialized": True,
            "gate_weight_zero_initialized": True,
            "initial_output_equivalent_to_C1": True,
        }


def load_old_reader(model, checkpoint):
    metadata = checkpoint["reader_metadata"]
    reader = LearnableCanonicalHeadReader(
        model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"],
        metadata["top_k"], 0.25
    ).to(model.device)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    reader.eval()
    return reader


def assert_trainable_boundary(model, reader, optimizer):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Receiver backbone is not frozen")
    expected = {id(parameter) for parameter in reader.new_parameters()}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual:
        raise RuntimeError("Optimizer must contain exactly new output-control parameters")
    if any(parameter.requires_grad for parameter in reader.base.parameters()):
        raise RuntimeError("Base C1 Reader is not frozen")


def assert_frozen_gradients(model, reader):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Gradient reached Receiver backbone")
    if any(parameter.grad is not None for parameter in reader.base.parameters()):
        raise RuntimeError("Gradient reached frozen C1 Reader")
