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
        value = value.float()
        return value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class EvidenceAssimilationAdapter(nn.Module):
    def __init__(self, hidden_size=2560, bottleneck=128, beta_init=0.01, eps=1e-6):
        super().__init__()
        if not 0.0 < beta_init < 0.1:
            raise ValueError("beta_init must be in (0,0.1)")
        self.hidden_norm = RMSNorm(hidden_size, eps)
        self.evidence_norm = RMSNorm(hidden_size, eps)
        self.hidden_down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.evidence_down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        nn.init.orthogonal_(self.hidden_down.weight)
        nn.init.orthogonal_(self.evidence_down.weight)
        nn.init.zeros_(self.up.weight)
        self.beta_logit = nn.Parameter(torch.tensor(math.atanh(beta_init / 0.1)))
        self.beta_init = float(beta_init)

    def beta(self):
        return 0.1 * torch.tanh(self.beta_logit)

    def forward(self, hidden, evidence):
        fused = F.silu(self.hidden_down(self.hidden_norm(hidden)) +
                       self.evidence_down(self.evidence_norm(evidence)))
        assimilation = self.up(fused).to(evidence.dtype)
        correction = self.beta().to(evidence.dtype) * assimilation
        return correction, {"assimilation": assimilation, "correction": correction,
                            "beta": self.beta(), "fused": fused}


class EvidenceAssimilationReader(nn.Module):
    def __init__(self, model, c1_checkpoint, bottleneck=128, beta_init=0.01):
        super().__init__()
        metadata = c1_checkpoint["reader_metadata"]
        self.selected_layers = list(metadata["selected_layers"])
        self.base = LearnableCanonicalHeadReader(
            model, self.selected_layers, metadata["rank"], metadata["gate_init"],
            metadata["top_k"], 0.25
        )
        self.base.load_state_dict(c1_checkpoint["reader"])
        self.base.requires_grad_(False)
        self.assimilation_start = len(self.selected_layers) // 2
        hidden_size = int(model.config.hidden_size)
        eps = float(getattr(model.config, "rms_norm_eps", 1e-6))
        self.adapters = nn.ModuleList([
            EvidenceAssimilationAdapter(hidden_size, bottleneck, beta_init, eps)
            for _ in self.selected_layers[self.assimilation_start:]
        ])
        self.bottleneck, self.beta_init = int(bottleneck), float(beta_init)
        self._memory, self._trace = None, None
        self._queries, self._raw_hidden = {}, {}

    def new_parameters(self):
        return [parameter for adapter in self.adapters for parameter in adapter.parameters()]

    def old_gates(self):
        return torch.stack([branch.gate.detach() for branch in self.base.branches])

    def betas(self):
        return torch.stack([adapter.beta() for adapter in self.adapters])

    def set_temperature(self, value):
        self.base.set_temperature(value)

    def branch_evidence(self, local, q_states, keys, values, mask, native_o_proj):
        branch = self.base.branches[local]
        query = branch.query_adapter(branch.native_query(q_states)).float()
        scores = torch.einsum("bsqd,tcd->bsqct", query, keys.float()) / math.sqrt(branch.head_dim)
        scores = scores.masked_fill(~mask[None, None, None, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        canonical_readout = torch.einsum("bsqct,tcd->bsqcd", attention, values.float())
        route = branch.routing_weights()
        readout = torch.einsum("qc,bsqcd->bsqd", route, canonical_readout)
        flattened = readout.reshape(readout.shape[0], readout.shape[1], branch.query_heads * branch.head_dim)
        evidence = native_o_proj(flattened.to(q_states.dtype))
        return evidence, {"attention": attention, "route": route, "headwise_readout": readout}

    @contextmanager
    def inject(self, model, memory, trace=None):
        keys = memory["keys"]
        if keys.ndim != 4 or keys.shape[0] != 16 or keys.shape[-2:] != (16, 128):
            raise RuntimeError(f"Assimilation Reader/memory mismatch: {tuple(keys.shape)}")
        self._memory, self._trace = memory, trace
        handles = []
        for local, layer_index in enumerate(self.selected_layers):
            layer = model.model.layers[layer_index]
            attention = layer.self_attn

            def layer_pre_hook(module, args, kwargs, local=local):
                self._raw_hidden[local] = kwargs.get("hidden_states", args[0] if args else None)

            def q_hook(module, args, output, local=local):
                self._queries[local] = output

            def attention_hook(module, args, kwargs, output, local=local, layer_index=layer_index):
                if local not in self._queries or local not in self._raw_hidden:
                    raise RuntimeError("Assimilation hooks did not capture Query/raw hidden")
                evidence, details = self.branch_evidence(
                    local, self._queries.pop(local), self._memory["keys"][local],
                    self._memory["values"][local], self._memory["mask"], module.o_proj
                )
                old_gate = self.base.branches[local].gate.to(evidence.dtype)
                old_delta = old_gate * evidence
                raw_hidden = self._raw_hidden.pop(local)
                if local >= self.assimilation_start:
                    correction, assimilation_details = self.adapters[local - self.assimilation_start](
                        raw_hidden, evidence
                    )
                    delta = old_delta + correction
                    details.update(assimilation_details)
                else:
                    delta = old_delta
                    details.update({"assimilation": None, "correction": None, "beta": None})
                details.update({"evidence": evidence, "old_gate": old_gate,
                                "old_delta": old_delta, "delta": delta})
                if self._trace is not None:
                    self._trace.setdefault(layer_index, []).append(details)
                if isinstance(output, tuple):
                    return (output[0] + delta,) + output[1:]
                return output + delta

            handles.append(layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))
            handles.append(attention.q_norm.register_forward_hook(q_hook))
            handles.append(attention.register_forward_hook(attention_hook, with_kwargs=True))
        try:
            yield trace
        finally:
            for handle in handles:
                handle.remove()
            self._queries.clear()
            self._raw_hidden.clear()
            self._memory, self._trace = None, None

    def metadata(self):
        return {
            "experiment": "P3-E-H Evidence Assimilation Reader V2",
            "selected_layers": self.selected_layers,
            "unchanged_front_layers": self.selected_layers[:self.assimilation_start],
            "assimilation_layers": self.selected_layers[self.assimilation_start:],
            "base_reader": "complete_frozen_C1",
            "bottleneck": self.bottleneck,
            "beta": "0.1*tanh(beta_logit)",
            "beta_init": self.beta_init,
            "up_zero_initialized": True,
            "initial_output_equivalent_to_C1": True,
            "query_path_modified": False,
            "native_o_proj_frozen": True,
        }


def load_c1_reader(model, checkpoint):
    metadata = checkpoint["reader_metadata"]
    reader = LearnableCanonicalHeadReader(
        model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"],
        metadata["top_k"], 0.25
    ).to(model.device)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    reader.eval()
    return reader


def assert_boundaries(model, reader, optimizer):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Receiver backbone is not frozen")
    if any(parameter.requires_grad for parameter in reader.base.parameters()):
        raise RuntimeError("C1 Reader is not frozen")
    expected = {id(parameter) for parameter in reader.new_parameters()}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual:
        raise RuntimeError("Optimizer must contain exactly Assimilation parameters")


def assert_frozen_gradients(model, reader):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Gradient reached Receiver backbone")
    if any(parameter.grad is not None for parameter in reader.base.parameters()):
        raise RuntimeError("Gradient reached frozen C1 Reader")
