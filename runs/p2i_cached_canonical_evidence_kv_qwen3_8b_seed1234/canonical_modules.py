import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


def decoder_layers(model):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    language_model = getattr(base, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model.layers
    raise RuntimeError("Could not locate decoder layers")


def model_hidden_size(model):
    config = getattr(model.config, "text_config", model.config)
    return int(config.hidden_size)


def full_attention_layers(model):
    layers = decoder_layers(model)
    config = getattr(model.config, "text_config", model.config)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return list(range(len(layers)))
    selected = [index for index, kind in enumerate(layer_types) if kind == "full_attention"]
    if not selected:
        raise RuntimeError("Receiver exposes no full-attention layers")
    return selected


class CanonicalEvidenceWriter(nn.Module):
    """Pool sender-native KV atoms into a receiver-independent evidence interface."""

    def __init__(
        self,
        sender_layers=36,
        sender_heads=8,
        sender_head_dim=128,
        slots=256,
        canonical_dim=256,
        atom_dim=64,
    ):
        super().__init__()
        self.sender_layers = int(sender_layers)
        self.sender_heads = int(sender_heads)
        self.sender_head_dim = int(sender_head_dim)
        self.slots = int(slots)
        self.canonical_dim = int(canonical_dim)
        self.atom_dim = int(atom_dim)

        self.key_input = nn.Linear(self.sender_head_dim, self.atom_dim, bias=False)
        self.value_input = nn.Linear(self.sender_head_dim, self.atom_dim, bias=False)
        self.layer_embedding = nn.Parameter(torch.empty(self.sender_layers, self.atom_dim))
        self.head_embedding = nn.Parameter(torch.empty(self.sender_heads, self.atom_dim))
        self.slot_queries = nn.Parameter(torch.empty(self.slots, self.atom_dim))
        self.support_norm = nn.LayerNorm(self.atom_dim)
        self.slot_norm = nn.LayerNorm(3 * self.atom_dim)
        self.slot_fusion = nn.Sequential(
            nn.Linear(3 * self.atom_dim, self.canonical_dim),
            nn.SiLU(),
            nn.Linear(self.canonical_dim, self.canonical_dim),
        )
        self.key_output = nn.Linear(self.canonical_dim, self.canonical_dim, bias=False)
        self.value_output = nn.Linear(self.canonical_dim, self.canonical_dim, bias=False)
        self.key_norm = nn.LayerNorm(self.canonical_dim)
        self.value_norm = nn.LayerNorm(self.canonical_dim)
        self.slot_gate_logits = nn.Parameter(torch.full((self.slots,), 2.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.key_input.weight)
        nn.init.orthogonal_(self.value_input.weight)
        nn.init.normal_(self.layer_embedding, std=0.02)
        nn.init.normal_(self.head_embedding, std=0.02)
        nn.init.normal_(self.slot_queries, std=1.0 / math.sqrt(self.atom_dim))
        nn.init.orthogonal_(self.key_output.weight)
        nn.init.orthogonal_(self.value_output.weight)

    def _validate(self, memory):
        if len(memory["keys"]) != self.sender_layers or len(memory["values"]) != self.sender_layers:
            raise ValueError("Unexpected sender layer count")
        expected = (self.sender_heads, self.sender_head_dim)
        for key, value in zip(memory["keys"], memory["values"]):
            if key.ndim != 3 or value.ndim != 3:
                raise ValueError("Native KV tensors must be [heads, tokens, head_dim]")
            if (key.shape[0], key.shape[-1]) != expected or tuple(key.shape) != tuple(value.shape):
                raise ValueError(f"Unexpected native KV geometry: {tuple(key.shape)}")

    def forward(self, memory, output_dtype=None, return_diagnostics=False):
        self._validate(memory)
        key = torch.stack([tensor.float() for tensor in memory["keys"]], dim=0)
        value = torch.stack([tensor.float() for tensor in memory["values"]], dim=0)
        layer_bias = self.layer_embedding[:, None, None, :]
        head_bias = self.head_embedding[None, :, None, :]
        key_atoms = self.key_input(key)
        value_atoms = self.value_input(value)
        support = self.support_norm(key_atoms + value_atoms + layer_bias + head_bias)
        support = support.reshape(-1, self.atom_dim)
        key_atoms = key_atoms.reshape(-1, self.atom_dim)
        value_atoms = value_atoms.reshape(-1, self.atom_dim)

        scores = torch.matmul(self.slot_queries.float(), support.transpose(0, 1))
        assignment = (scores / math.sqrt(self.atom_dim)).softmax(dim=-1)
        pooled_support = torch.matmul(assignment, support)
        pooled_key = torch.matmul(assignment, key_atoms)
        pooled_value = torch.matmul(assignment, value_atoms)
        shared_state = self.slot_fusion(
            self.slot_norm(torch.cat((pooled_support, pooled_key, pooled_value), dim=-1))
        )
        gate = self.slot_gate_logits.sigmoid().unsqueeze(-1)
        key_e = self.key_norm(self.key_output(shared_state)) * gate
        value_e = self.value_norm(self.value_output(shared_state)) * gate
        dtype = output_dtype or key[0].dtype
        output = {"keys": key_e.to(dtype), "values": value_e.to(dtype)}

        answer_mask = memory.get("answer_token_mask")
        if answer_mask is not None:
            atom_mask = answer_mask.to(assignment.device, torch.bool)
            atom_mask = atom_mask.view(1, 1, -1).expand(
                self.sender_layers, self.sender_heads, -1
            ).reshape(-1)
            output["answer_slot_mass"] = assignment[:, atom_mask].sum(dim=-1)
        if return_diagnostics:
            normalized_entropy = -(
                assignment * assignment.clamp_min(1e-9).log()
            ).sum(dim=-1) / math.log(max(2, assignment.shape[-1]))
            output["diagnostics"] = {
                "assignment": assignment,
                "slot_entropy": normalized_entropy.mean(),
                "slot_usage": gate.squeeze(-1),
                "atom_coverage": assignment.sum(dim=0),
                "slot_cosine": F.normalize(shared_state, dim=-1) @ F.normalize(shared_state, dim=-1).T,
                "gate_mean": gate.mean(),
            }
        return output

    def regularization(self, output):
        diag = output["diagnostics"]
        usage = diag["slot_usage"] / diag["slot_usage"].sum().clamp_min(1e-8)
        usage_loss = (usage * usage.clamp_min(1e-9).log()).sum() + math.log(self.slots)
        coverage = diag["atom_coverage"]
        coverage = coverage / coverage.sum().clamp_min(1e-8)
        coverage_loss = (coverage * coverage.clamp_min(1e-9).log()).sum() + math.log(coverage.numel())
        cosine = diag["slot_cosine"]
        off_diagonal = cosine - torch.eye(self.slots, device=cosine.device)
        diversity_loss = off_diagonal.square().mean()
        return {"usage": usage_loss + 0.25 * coverage_loss, "diversity": diversity_loss}


class CanonicalExternalReader(nn.Module):
    """Receiver-specific reader for one unordered canonical slot set."""

    def __init__(
        self,
        model,
        canonical_dim=256,
        adapter_rank=32,
        max_gate=0.5,
        gate_init=0.01,
        active_layers=None,
    ):
        super().__init__()
        self.hidden_size = model_hidden_size(model)
        self.canonical_dim = int(canonical_dim)
        self.adapter_rank = int(adapter_rank)
        self.layer_count = len(decoder_layers(model))
        self.active_layers = list(active_layers if active_layers is not None else full_attention_layers(model))
        if len(set(self.active_layers)) != len(self.active_layers):
            raise ValueError("Reader layer indices must be unique")
        self.layer_to_slot = {layer: slot for slot, layer in enumerate(self.active_layers)}
        self.max_gate = float(max_gate)

        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.shared_query = nn.Linear(self.hidden_size, self.canonical_dim, bias=False)
        self.shared_output = nn.Linear(self.canonical_dim, self.hidden_size, bias=False)
        count = len(self.active_layers)
        self.query_down = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.adapter_rank, bias=False) for _ in range(count)]
        )
        self.query_up = nn.ModuleList(
            [nn.Linear(self.adapter_rank, self.canonical_dim, bias=False) for _ in range(count)]
        )
        self.output_down = nn.ModuleList(
            [nn.Linear(self.canonical_dim, self.adapter_rank, bias=False) for _ in range(count)]
        )
        self.output_up = nn.ModuleList(
            [nn.Linear(self.adapter_rank, self.hidden_size, bias=False) for _ in range(count)]
        )
        ratio = max(-0.999, min(0.999, gate_init / max(self.max_gate, 1e-8)))
        self.gate_logits = nn.Parameter(torch.full((count,), float(torch.atanh(torch.tensor(ratio)))))
        for module in list(self.query_up) + list(self.output_up):
            nn.init.zeros_(module.weight)
        nn.init.orthogonal_(self.shared_query.weight)
        nn.init.orthogonal_(self.shared_output.weight)
        self._memory = None
        self._pending = {}
        self._diagnostics = None

    def gates(self):
        return self.max_gate * torch.tanh(self.gate_logits)

    def _read(self, hidden, layer_index):
        local = self.layer_to_slot[layer_index]
        normalized = self.input_norm(hidden.float())
        query = self.shared_query(normalized)
        query = query + self.query_up[local](F.silu(self.query_down[local](normalized)))
        keys = self._memory["keys"].float()
        values = self._memory["values"].float()
        probability = torch.matmul(query, keys.transpose(0, 1)) / math.sqrt(self.canonical_dim)
        probability = probability.softmax(dim=-1)
        readout = torch.matmul(probability, values)
        projected = self.shared_output(readout)
        projected = projected + self.output_up[local](F.silu(self.output_down[local](readout)))
        gate = self.gates()[local]
        delta = gate * projected
        if self._diagnostics is not None:
            slot = self._diagnostics.setdefault(str(layer_index), {})
            slot["calls"] = slot.get("calls", 0) + 1
            slot["attention_entropy"] = slot.get("attention_entropy", 0.0) + float(
                (-(probability * probability.clamp_min(1e-9).log()).sum(-1).mean()
                 / math.log(max(2, probability.shape[-1]))).detach().cpu()
            )
            slot["readout_norm"] = slot.get("readout_norm", 0.0) + float(
                projected.detach().norm(dim=-1).mean().cpu()
            )
            slot["delta_norm"] = slot.get("delta_norm", 0.0) + float(
                delta.detach().norm(dim=-1).mean().cpu()
            )
            slot["gate"] = float(gate.detach().cpu())
            answer_mass = self._memory.get("answer_slot_mass")
            if answer_mass is not None:
                slot["target_attention_mass"] = slot.get("target_attention_mass", 0.0) + float(
                    (probability * answer_mass.float()).sum(-1).mean().detach().cpu()
                )
            if self._diagnostics.get("_capture_tensors", False):
                slot["attention_tensor"] = probability[:, -1].mean(0)
                slot["readout_tensor"] = projected[:, -1].mean(0)
                slot["delta_tensor"] = delta[:, -1].mean(0)
        return delta.to(hidden.dtype)

    @contextmanager
    def inject(self, model, memory, diagnostics=None):
        if tuple(memory["keys"].shape) != tuple(memory["values"].shape):
            raise ValueError("Canonical K/V shapes differ")
        if memory["keys"].ndim != 2 or memory["keys"].shape[-1] != self.canonical_dim:
            raise ValueError("Canonical memory must be [slots, canonical_dim]")
        self._memory = memory
        self._diagnostics = diagnostics
        handles = []
        layers = decoder_layers(model)
        for layer_index in self.active_layers:
            layer = layers[layer_index]

            def pre_hook(module, args, kwargs, layer_index=layer_index):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                if hidden is None:
                    raise RuntimeError("Decoder layer input hidden state was not found")
                self._pending[layer_index] = self._read(hidden, layer_index)

            def post_hook(module, args, kwargs, output, layer_index=layer_index):
                delta = self._pending.pop(layer_index)
                if isinstance(output, tuple):
                    return (output[0] + delta,) + output[1:]
                return output + delta

            handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(layer.register_forward_hook(post_hook, with_kwargs=True))
        try:
            yield diagnostics
        finally:
            for handle in handles:
                handle.remove()
            self._pending.clear()
            self._memory = None
            self._diagnostics = None

    def metadata(self):
        return {
            "hidden_size": self.hidden_size,
            "canonical_dim": self.canonical_dim,
            "adapter_rank": self.adapter_rank,
            "layer_count": self.layer_count,
            "active_layers": self.active_layers,
            "max_gate": self.max_gate,
        }


def permute_slots(memory, permutation):
    output = {
        "keys": memory["keys"].index_select(0, permutation),
        "values": memory["values"].index_select(0, permutation),
    }
    if "answer_slot_mass" in memory:
        output["answer_slot_mass"] = memory["answer_slot_mass"].index_select(0, permutation)
    return output


def zero_slots(memory):
    output = {"keys": torch.zeros_like(memory["keys"]), "values": torch.zeros_like(memory["values"])}
    if "answer_slot_mass" in memory:
        output["answer_slot_mass"] = torch.zeros_like(memory["answer_slot_mass"])
    return output


def drop_half_slots(memory):
    keep = torch.arange(0, memory["keys"].shape[0], 2, device=memory["keys"].device)
    output = {
        "keys": memory["keys"].index_select(0, keep),
        "values": memory["values"].index_select(0, keep),
    }
    if "answer_slot_mass" in memory:
        output["answer_slot_mass"] = memory["answer_slot_mass"].index_select(0, keep)
    return output


def mismatched_slots(key_memory, value_memory):
    output = {"keys": key_memory["keys"], "values": value_memory["values"]}
    if "answer_slot_mass" in value_memory:
        output["answer_slot_mass"] = value_memory["answer_slot_mass"]
    return output
