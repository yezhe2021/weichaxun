import math
from contextlib import contextmanager

import torch
import torch.nn as nn

from p3e_a_common import PerQueryHeadResidual
from p3e_c0_common import DuplicateHeadwiseCache, duplicate_memory_to


class SparseCanonicalRouterBranch(nn.Module):
    def __init__(self, query_heads=32, canonical_heads=16, head_dim=128, rank=32, gate_init=0.01, top_k=2, temperature=1.0):
        super().__init__(); self.query_heads, self.canonical_heads, self.head_dim = query_heads, canonical_heads, head_dim
        self.top_k, self.temperature = int(top_k), float(temperature)
        self.query_adapter = PerQueryHeadResidual(query_heads, head_dim, rank)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        logits = torch.full((query_heads, canonical_heads), -2.0)
        for query_head in range(query_heads):
            pair = 2 * (query_head // 4); logits[query_head, pair:pair + 2] = 0.0
        self.router_logits = nn.Parameter(logits)

    def routing_weights(self):
        soft = (self.router_logits.float() / self.temperature).softmax(dim=-1)
        indices = soft.topk(self.top_k, dim=-1).indices
        mask = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
        hard = soft * mask; hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.training:
            return hard.detach() - soft.detach() + soft
        return hard

    def native_query(self, q_states):
        if q_states.ndim != 4: raise RuntimeError(f"Expected rank-4 q_norm output, got {tuple(q_states.shape)}")
        if q_states.shape[1] == self.query_heads: q_states = q_states.transpose(1, 2)
        elif q_states.shape[2] != self.query_heads: raise RuntimeError("Cannot locate Query-head axis")
        if q_states.shape[-1] != self.head_dim: raise RuntimeError("Query head dimension mismatch")
        return q_states

    def forward(self, q_states, keys, values, mask, native_o_proj):
        if keys.shape != values.shape or keys.ndim != 3 or keys.shape[1:] != (self.canonical_heads, self.head_dim):
            raise RuntimeError(f"Expected K/V [T,16,128], got {tuple(keys.shape)}")
        query = self.query_adapter(self.native_query(q_states)).float()
        scores = torch.einsum("bsqd,tcd->bsqct", query, keys.float()) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask[None, None, None, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        canonical_readout = torch.einsum("bsqct,tcd->bsqcd", attention, values.float())
        route = self.routing_weights(); readout = torch.einsum("qc,bsqcd->bsqd", route, canonical_readout)
        flattened = readout.reshape(readout.shape[0], readout.shape[1], self.query_heads * self.head_dim)
        projected = native_o_proj(flattened.to(q_states.dtype)); delta = self.gate.to(projected.dtype) * projected
        return delta, {"attention": attention, "canonical_readout": canonical_readout, "route": route,
                       "headwise_readout": readout, "projected": projected, "delta": delta}


class LearnableCanonicalHeadReader(nn.Module):
    def __init__(self, model, selected_layers, rank=32, gate_init=0.01, top_k=2, temperature=1.0):
        super().__init__(); config = model.config; self.selected_layers = list(selected_layers); self.rank, self.gate_init = int(rank), float(gate_init)
        self.query_heads = int(config.num_attention_heads); self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.canonical_heads, self.top_k = 16, int(top_k)
        self.branches = nn.ModuleList([SparseCanonicalRouterBranch(self.query_heads, self.canonical_heads, self.head_dim, rank, gate_init, top_k, temperature) for _ in self.selected_layers])
        self._memory, self._trace, self._queries = None, None, {}

    def set_temperature(self, value):
        for branch in self.branches: branch.temperature = float(value)

    def gates(self): return torch.stack([branch.gate for branch in self.branches])
    def routes(self): return torch.stack([branch.routing_weights() for branch in self.branches])

    def load_native_reader(self, checkpoint):
        state = checkpoint["reader"]; own = self.state_dict(); copied = []
        with torch.no_grad():
            for name, tensor in state.items():
                if name in own and not name.endswith("router_logits"):
                    own[name].copy_(tensor); copied.append(name)
        expected = len(self.selected_layers) * 3
        if len(copied) != expected: raise RuntimeError(f"Expected {expected} warm-start tensors, copied {len(copied)}")
        return copied

    @contextmanager
    def inject(self, model, memory, trace=None):
        if memory["keys"].shape[0] != len(self.selected_layers) or memory["keys"].shape[-2:] != (16, self.head_dim): raise RuntimeError("Reader/memory mismatch")
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
        return {"selected_layers": self.selected_layers, "query_heads": self.query_heads, "canonical_heads": self.canonical_heads,
                "head_dim": self.head_dim, "rank": self.rank, "gate_init": self.gate_init, "top_k": self.top_k,
                "routing": "per_receiver_layer_32x16_straight_through_topk", "temperature_schedule": "1.0_to_0.25",
                "writer": "fixed_duplicate_writer16_frozen", "native_o_proj_frozen": True,
                "layer_router": False, "output_mlp": False, "canonical_dimension_compression": False}


__all__ = ["LearnableCanonicalHeadReader", "DuplicateHeadwiseCache", "duplicate_memory_to"]
