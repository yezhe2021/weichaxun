import math
from contextlib import contextmanager

import torch
import torch.nn as nn

from p3e_a_common import PerQueryHeadResidual
from p3e_b_common import SenderNativeHeadwiseCache, native_memory_to


class DuplicateHeadwiseCache:
    """Fixed, parameter-free Native 8-head -> Canonical 16-head duplication."""
    def __init__(self, index_path, capacity=4):
        self.native = SenderNativeHeadwiseCache(index_path, capacity=capacity)
        self.entries, self.index = self.native.entries, dict(self.native.index)
        self.index.update({"canonical_heads": 16, "canonical_head_dim": 128, "writer": "fixed_repeat_interleave_2", "lossless": True})

    def __len__(self): return len(self.native)

    def load(self, index):
        source = self.native.load(index); payload = dict(source)
        payload["keys"] = source["keys"].repeat_interleave(2, dim=2)
        payload["values"] = source["values"].repeat_interleave(2, dim=2)
        if payload["keys"].shape[-2:] != (16, 128): raise RuntimeError("Duplicate Writer shape mismatch")
        if not torch.equal(payload["keys"][:, :, 0::2], source["keys"]) or not torch.equal(payload["keys"][:, :, 1::2], source["keys"]):
            raise RuntimeError("K duplication mapping is not exact")
        if not torch.equal(payload["values"][:, :, 0::2], source["values"]) or not torch.equal(payload["values"][:, :, 1::2], source["values"]):
            raise RuntimeError("V duplication mapping is not exact")
        return payload


def duplicate_memory_to(payload, device, oracle_support=False):
    keys, values = payload["keys"].float().to(device), payload["values"].float().to(device)
    if keys.shape != values.shape or keys.ndim != 4 or keys.shape[-2:] != (16, 128):
        raise RuntimeError(f"Expected duplicate memory [16,T,16,128], got {tuple(keys.shape)}")
    valid = torch.as_tensor(payload["metadata"]["valid_mask"], dtype=torch.bool, device=device)
    support = torch.as_tensor(payload["metadata"]["support_token_mask"], dtype=torch.bool, device=device)
    mask = valid & support if oracle_support else valid
    if mask.numel() != keys.shape[1] or not mask.any(): raise RuntimeError("Duplicate memory mask mismatch")
    return {"keys": keys, "values": values, "mask": mask, "support_mask": support}


class DiagnosticBranch(nn.Module):
    def __init__(self, duplicate, query_heads=32, native_kv_heads=8, head_dim=128, rank=32, gate_init=0.01):
        super().__init__(); self.duplicate = bool(duplicate); self.query_heads = query_heads; self.native_kv_heads = native_kv_heads
        self.head_dim, self.query_per_kv = head_dim, query_heads // native_kv_heads
        self.query_adapter = PerQueryHeadResidual(query_heads, head_dim, rank)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def native_query(self, q_states):
        if q_states.ndim != 4: raise RuntimeError(f"Expected rank-4 q_norm output, got {tuple(q_states.shape)}")
        if q_states.shape[1] == self.query_heads: q_states = q_states.transpose(1, 2)
        elif q_states.shape[2] != self.query_heads: raise RuntimeError("Cannot locate Query-head axis")
        if q_states.shape[-1] != self.head_dim: raise RuntimeError("Query dimension mismatch")
        return q_states

    def forward(self, q_states, keys, values, mask, native_o_proj):
        query = self.query_adapter(self.native_query(q_states))
        batch, sequence = query.shape[:2]
        query = query.reshape(batch, sequence, self.native_kv_heads, self.query_per_kv, self.head_dim)
        if self.duplicate:
            if keys.shape[1:] != (16, 128): raise RuntimeError("Duplicate K/V geometry mismatch")
            paired_keys = keys.reshape(keys.shape[0], self.native_kv_heads, 2, self.head_dim)
            paired_values = values.reshape(values.shape[0], self.native_kv_heads, 2, self.head_dim)
            scores = torch.einsum("bshgd,thpd->bshgpt", query, paired_keys.float()) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask[None, None, None, None, None, :], torch.finfo(scores.dtype).min)
            pair_attention = scores.softmax(dim=-1)
            pair_readout = torch.einsum("bshgpt,thpd->bshgpd", pair_attention, paired_values.float())
            attention = pair_attention.mean(dim=-2)
            readout = pair_readout.mean(dim=-2)
        else:
            if keys.shape[1:] != (8, 128): raise RuntimeError("Native K/V geometry mismatch")
            scores = torch.einsum("bshgd,thd->bshgt", query, keys.float()) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask[None, None, None, None, :], torch.finfo(scores.dtype).min)
            attention = scores.softmax(dim=-1); pair_attention = None
            readout = torch.einsum("bshgt,thd->bshgd", attention, values.float()); pair_readout = None
        flattened = readout.reshape(batch, sequence, self.query_heads * self.head_dim)
        projected = native_o_proj(flattened.to(q_states.dtype)); delta = self.gate.to(projected.dtype) * projected
        return delta, {"attention": attention, "pair_attention": pair_attention, "headwise_readout": readout,
                       "pair_readout": pair_readout, "projected": projected, "delta": delta}


class DiagnosticHeadwiseReader(nn.Module):
    def __init__(self, model, selected_layers, duplicate, rank=32, gate_init=0.01):
        super().__init__(); config = model.config; self.selected_layers = list(selected_layers); self.duplicate = bool(duplicate)
        self.query_heads, self.kv_heads = int(config.num_attention_heads), int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)); self.rank, self.gate_init = int(rank), float(gate_init)
        self.branches = nn.ModuleList([DiagnosticBranch(duplicate, self.query_heads, self.kv_heads, self.head_dim, rank, gate_init) for _ in self.selected_layers])
        self._memory, self._trace, self._queries = None, None, {}

    @contextmanager
    def inject(self, model, memory, trace=None):
        expected_heads = 16 if self.duplicate else 8
        if memory["keys"].shape[0] != len(self.selected_layers) or memory["keys"].shape[-2:] != (expected_heads, self.head_dim): raise RuntimeError("Reader/memory mismatch")
        self._memory, self._trace = memory, trace; handles = []
        for local, layer_index in enumerate(self.selected_layers):
            attention = model.model.layers[layer_index].self_attn
            def q_hook(module, args, output, local=local): self._queries[local] = output
            def attention_hook(module, args, kwargs, output, local=local, layer_index=layer_index):
                delta, details = self.branches[local](self._queries.pop(local), self._memory["keys"][local], self._memory["values"][local], self._memory["mask"], module.o_proj)
                if self._trace is not None: self._trace.setdefault(layer_index, []).append(details)
                if isinstance(output, tuple): return (output[0] + delta,) + output[1:]
                return output + delta
            handles.append(attention.q_norm.register_forward_hook(q_hook)); handles.append(attention.register_forward_hook(attention_hook, with_kwargs=True))
        try: yield trace
        finally:
            for handle in handles: handle.remove()
            self._queries.clear(); self._memory = None; self._trace = None

    def metadata(self):
        return {"selected_layers": self.selected_layers, "duplicate": self.duplicate, "query_heads": self.query_heads,
                "native_kv_heads": self.kv_heads, "memory_heads": 16 if self.duplicate else 8, "head_dim": self.head_dim,
                "rank": self.rank, "gate_init": self.gate_init, "merge": "independent_token_softmax_then_mean_two_complete_readouts",
                "native_o_proj_frozen": True, "writer_trainable_parameters": 0}


__all__ = ["DuplicateHeadwiseCache", "duplicate_memory_to", "DiagnosticHeadwiseReader", "SenderNativeHeadwiseCache", "native_memory_to"]
