import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer


def load_qwen35(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(device).eval()
    model.requires_grad_(False); return model, tokenizer


def text_backbone(model):
    backbone = getattr(getattr(model, "model", None), "language_model", None)
    if backbone is None or not hasattr(backbone, "layers"): raise RuntimeError("Qwen3.5 text backbone was not found")
    return backbone


class Qwen35CanonicalBranch(nn.Module):
    def __init__(self, query_heads=16, native_dim=256, canonical_heads=16, canonical_dim=128, rank=32, gate_init=0.01, top_k=2, seed=1234):
        super().__init__(); self.query_heads, self.native_dim, self.canonical_heads, self.canonical_dim = query_heads, native_dim, canonical_heads, canonical_dim
        self.rank, self.top_k = rank, top_k; generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.query_base = nn.Parameter(torch.empty(query_heads, canonical_dim, native_dim)); self.query_down = nn.Parameter(torch.empty(query_heads, rank, native_dim)); self.query_up = nn.Parameter(torch.empty(query_heads, canonical_dim, rank))
        self.output_base = nn.Parameter(torch.empty(query_heads, native_dim, canonical_dim)); self.output_down = nn.Parameter(torch.empty(query_heads, rank, canonical_dim)); self.output_up = nn.Parameter(torch.empty(query_heads, native_dim, rank))
        for head in range(query_heads):
            nn.init.orthogonal_(self.query_base[head]); nn.init.orthogonal_(self.query_down[head]); nn.init.orthogonal_(self.output_base[head]); nn.init.orthogonal_(self.output_down[head])
        with torch.no_grad(): self.query_up.normal_(0.0, 1e-3, generator=generator); self.output_up.normal_(0.0, 1e-3, generator=generator)
        self.head_router_logits = nn.Parameter(torch.randn(query_heads, canonical_heads, generator=generator) * 0.02)
        self.group_logits = nn.Parameter(torch.randn(query_heads, 2, generator=generator) * 0.02)
        self.gate = nn.Parameter(torch.tensor(float(gate_init))); self.temperature = 1.0

    def head_route(self):
        soft = (self.head_router_logits.float() / self.temperature).softmax(-1); indices = soft.topk(self.top_k, dim=-1).indices
        mask = torch.zeros_like(soft).scatter_(-1, indices, 1.0); hard = soft * mask; hard = hard / hard.sum(-1, keepdim=True).clamp_min(1e-8)
        return hard.detach() - soft.detach() + soft if self.training else hard

    def forward(self, attention, hidden, keys, values, mask):
        batch, sequence = hidden.shape[:2]
        packed = attention.q_proj(hidden).view(batch, sequence, self.query_heads, self.native_dim * 2)
        native_query, native_gate = torch.chunk(packed, 2, dim=-1); native_query = attention.q_norm(native_query)
        query = torch.einsum("bsqd,qed->bsqe", native_query.float(), self.query_base)
        query_hidden = torch.einsum("bsqd,qrd->bsqr", native_query.float(), self.query_down)
        query = query + torch.einsum("bsqr,qer->bsqe", F.silu(query_hidden), self.query_up)
        scores = torch.einsum("bsqd,gtcd->bsqgct", query, keys.float()) / math.sqrt(self.canonical_dim)
        scores = scores.masked_fill(~mask[None, None, None, None, None, :], torch.finfo(scores.dtype).min)
        token_attention = scores.softmax(-1); canonical_readout = torch.einsum("bsqgct,gtcd->bsqgcd", token_attention, values.float())
        head_route = self.head_route(); grouped = torch.einsum("qc,bsqgcd->bsqgd", head_route, canonical_readout)
        group_route = self.group_logits.float().softmax(-1); readout = torch.einsum("qg,bsqgd->bsqd", group_route, grouped)
        projected = torch.einsum("bsqd,qed->bsqe", readout, self.output_base)
        output_hidden = torch.einsum("bsqd,qrd->bsqr", readout, self.output_down)
        projected = projected + torch.einsum("bsqr,qer->bsqe", F.silu(output_hidden), self.output_up)
        projected = projected * torch.sigmoid(native_gate.float())
        native_output = attention.o_proj(projected.reshape(batch, sequence, self.query_heads * self.native_dim).to(hidden.dtype))
        delta = self.gate.to(native_output.dtype) * native_output
        return delta, {"token_attention": token_attention, "head_route": head_route, "group_route": group_route,
                       "canonical_readout": canonical_readout, "headwise_readout": readout, "projected_heads": projected,
                       "native_o_projected": native_output, "delta": delta}


class Qwen35CanonicalReader(nn.Module):
    def __init__(self, model, rank=32, gate_init=0.01, top_k=2, seed=1234):
        super().__init__(); config = model.config.text_config; self.full_layers = [index for index, kind in enumerate(config.layer_types) if kind == "full_attention"]
        if self.full_layers != [3, 7, 11, 15, 19, 23, 27, 31]: raise RuntimeError(f"Unexpected Qwen3.5 full-attention layers: {self.full_layers}")
        self.memory_group_pairs = [[2 * index, 2 * index + 1] for index in range(len(self.full_layers))]
        self.rank, self.gate_init, self.top_k, self.seed = int(rank), float(gate_init), int(top_k), int(seed)
        self.branches = nn.ModuleList([Qwen35CanonicalBranch(16, 256, 16, 128, rank, gate_init, top_k, seed + index * 1009) for index in range(8)])
        self._memory, self._hidden, self._trace = None, {}, None

    def gates(self): return torch.stack([branch.gate for branch in self.branches])

    @contextmanager
    def inject(self, model, memory, trace=None):
        if memory["keys"].shape[0] != 16 or memory["keys"].shape[-2:] != (16, 128): raise RuntimeError("Qwen3.5 Reader expects [16,T,16,128]")
        self._memory, self._trace = memory, trace; handles = []; backbone = text_backbone(model)
        for local, layer_index in enumerate(self.full_layers):
            attention = backbone.layers[layer_index].self_attn
            def pre_hook(module, args, kwargs, local=local): self._hidden[local] = kwargs.get("hidden_states", args[0] if args else None)
            def output_hook(module, args, kwargs, output, local=local, layer_index=layer_index):
                groups = self.memory_group_pairs[local]; delta, details = self.branches[local](module, self._hidden.pop(local), self._memory["keys"][groups], self._memory["values"][groups], self._memory["mask"])
                if self._trace is not None: self._trace.setdefault(layer_index, []).append(details)
                if isinstance(output, tuple): return (output[0] + delta,) + output[1:]
                return output + delta
            handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True)); handles.append(attention.register_forward_hook(output_hook, with_kwargs=True))
        try: yield trace
        finally:
            for handle in handles: handle.remove()
            self._hidden.clear(); self._memory = None; self._trace = None

    def metadata(self):
        return {"receiver": "Qwen3.5-4B", "full_attention_layers": self.full_layers, "memory_group_pairs": self.memory_group_pairs,
                "query_heads": 16, "native_head_dim": 256, "canonical_heads": 16, "canonical_head_dim": 128,
                "rank": self.rank, "gate_init": self.gate_init, "top_k": self.top_k, "seed": self.seed,
                "native_query_gate_preserved": True, "native_o_proj_frozen": True, "linear_attention_layers_modified": False}


__all__ = ["Qwen35CanonicalReader", "load_qwen35", "text_backbone"]
