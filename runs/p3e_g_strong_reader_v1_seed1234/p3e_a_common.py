import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerQueryHeadResidual(nn.Module):
    def __init__(self, heads, dimension, rank):
        super().__init__()
        self.heads, self.dimension, self.rank = heads, dimension, rank
        self.down = nn.Parameter(torch.empty(heads, rank, dimension))
        self.up = nn.Parameter(torch.zeros(heads, dimension, rank))
        for head in range(heads):
            nn.init.orthogonal_(self.down[head])

    def forward(self, query):
        hidden = torch.einsum("bshd,hrd->bshr", query.float(), self.down)
        update = torch.einsum("bshr,hdr->bshd", F.silu(hidden), self.up)
        return query.float() + update


class NativeHeadwiseBranch(nn.Module):
    def __init__(self, query_heads=32, kv_heads=8, head_dim=128, rank=32, gate_init=0.01):
        super().__init__()
        if query_heads % kv_heads: raise ValueError("Query heads must be divisible by KV heads")
        self.query_heads, self.kv_heads, self.head_dim = query_heads, kv_heads, head_dim
        self.query_per_kv = query_heads // kv_heads
        self.query_adapter = PerQueryHeadResidual(query_heads, head_dim, rank)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def native_query(self, q_states):
        if q_states.ndim != 4: raise RuntimeError(f"Expected rank-4 q_norm output, got {tuple(q_states.shape)}")
        if q_states.shape[1] == self.query_heads:
            q_states = q_states.transpose(1, 2)
        elif q_states.shape[2] != self.query_heads:
            raise RuntimeError(f"Cannot locate {self.query_heads} Query heads in {tuple(q_states.shape)}")
        if q_states.shape[-1] != self.head_dim: raise RuntimeError("Native Query head dimension mismatch")
        return q_states

    def forward(self, q_states, keys, values, mask, native_o_proj):
        if keys.shape != values.shape or keys.ndim != 3: raise RuntimeError("Expected K/V [T,H,D]")
        if keys.shape[1:] != (self.kv_heads, self.head_dim): raise RuntimeError(f"Native K/V geometry mismatch: {tuple(keys.shape)}")
        query = self.query_adapter(self.native_query(q_states))
        batch, sequence = query.shape[:2]
        grouped_query = query.reshape(batch, sequence, self.kv_heads, self.query_per_kv, self.head_dim)
        scores = torch.einsum("bshgd,thd->bshgt", grouped_query, keys.float()) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask[None, None, None, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        readout = torch.einsum("bshgt,thd->bshgd", attention, values.float())
        native_heads = readout.reshape(batch, sequence, self.query_heads * self.head_dim)
        projected = native_o_proj(native_heads.to(q_states.dtype))
        return self.gate.to(projected.dtype) * projected, attention


class NativeHeadwiseReader(nn.Module):
    def __init__(self, model, selected_layers, rank=32, gate_init=0.01):
        super().__init__()
        config = model.config
        self.selected_layers = list(selected_layers)
        self.query_heads = int(config.num_attention_heads)
        self.kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.rank, self.gate_init = int(rank), float(gate_init)
        self.branches = nn.ModuleList([
            NativeHeadwiseBranch(self.query_heads, self.kv_heads, self.head_dim, rank, gate_init)
            for _ in self.selected_layers
        ])
        self._memory, self._trace, self._queries = None, None, {}

    def gates(self):
        return torch.stack([branch.gate for branch in self.branches])

    @contextmanager
    def inject(self, model, memory, trace=None):
        expected = (len(self.selected_layers), self.kv_heads, self.head_dim)
        if memory["keys"].ndim != 4 or (memory["keys"].shape[0], memory["keys"].shape[2], memory["keys"].shape[3]) != expected:
            raise RuntimeError(f"Reader/memory interface mismatch: {tuple(memory['keys'].shape)}")
        self._memory, self._trace = memory, trace
        handles = []
        for local, layer_index in enumerate(self.selected_layers):
            attention_module = model.model.layers[layer_index].self_attn

            def q_hook(module, args, output, local=local):
                self._queries[local] = output

            def attention_hook(module, args, kwargs, output, local=local, layer_index=layer_index):
                if local not in self._queries: raise RuntimeError("q_norm hook did not capture pre-RoPE Native Query")
                delta, weights = self.branches[local](
                    self._queries.pop(local), self._memory["keys"][local], self._memory["values"][local],
                    self._memory["mask"], module.o_proj,
                )
                if self._trace is not None:
                    self._trace.setdefault(layer_index, []).append({"attention": weights, "delta": delta})
                if isinstance(output, tuple): return (output[0] + delta,) + output[1:]
                return output + delta

            handles.append(attention_module.q_norm.register_forward_hook(q_hook))
            handles.append(attention_module.register_forward_hook(attention_hook, with_kwargs=True))
        try:
            yield trace
        finally:
            for handle in handles: handle.remove()
            self._queries.clear(); self._memory = None; self._trace = None

    def metadata(self):
        return {
            "selected_layers": self.selected_layers, "query_heads": self.query_heads, "kv_heads": self.kv_heads,
            "head_dim": self.head_dim, "query_per_kv": self.query_heads // self.kv_heads,
            "rank": self.rank, "gate_init": self.gate_init,
            "query": "receiver_q_proj_reshape_q_norm_pre_rope_128d",
            "memory": "native_pre_rope_k_and_native_v_Tx8x128",
            "output": "native_32x128_heads_to_frozen_receiver_o_proj",
            "injection": "self_attention_output_before_decoder_residual",
            "canonical_projection_used": False,
        }


def native_memory_to(payload, device, oracle_support=False):
    keys, values = payload["keys"].float().to(device), payload["values"].float().to(device)
    if keys.shape != values.shape or keys.ndim != 4 or keys.shape[-2:] != (8, 128):
        raise RuntimeError(f"Invalid Native Headwise cache shape: {tuple(keys.shape)}")
    valid = torch.as_tensor(payload["metadata"]["valid_mask"], dtype=torch.bool, device=device)
    support = torch.as_tensor(payload["metadata"]["support_token_mask"], dtype=torch.bool, device=device)
    if valid.numel() != keys.shape[1] or support.numel() != keys.shape[1]: raise RuntimeError("Memory mask length mismatch")
    mask = valid & support if oracle_support else valid
    if not mask.any(): raise RuntimeError("Memory mask is empty")
    return {"keys": keys, "values": values, "mask": mask, "support_mask": support}
