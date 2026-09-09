from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def relative_depth_matrix(anchors, target_layers):
    anchors = [float(x) for x in anchors]
    rows = []
    for target in range(target_layers):
        row = torch.zeros(len(anchors), dtype=torch.float32)
        if target <= anchors[0]:
            row[0] = 1.0
        elif target >= anchors[-1]:
            row[-1] = 1.0
        else:
            right = next(i for i, value in enumerate(anchors) if value >= target)
            left = right - 1
            fraction = (target - anchors[left]) / (anchors[right] - anchors[left])
            row[left], row[right] = 1.0 - fraction, fraction
        rows.append(row)
    return torch.stack(rows)


class SharedLowRankResidual(nn.Module):
    def __init__(self, dim, rank, layers, gate):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.layer_embedding = nn.Parameter(torch.zeros(layers, rank))
        self.gate = nn.Parameter(torch.full((layers,), float(gate)))
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        dtype = x.dtype
        normalized = F.rms_norm(
            x, (x.shape[-1],), self.norm.weight.to(dtype), self.norm.eps
        )
        hidden = F.linear(normalized, self.down.weight.to(dtype))
        view = (x.shape[0],) + (1,) * (x.ndim - 2) + (hidden.shape[-1],)
        hidden = hidden + self.layer_embedding.to(dtype).view(view)
        residual = F.linear(F.silu(hidden), self.up.weight.to(dtype))
        gate_view = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x + self.gate.to(dtype).view(gate_view) * residual


class Qwen35FullAttentionWriter(nn.Module):
    """KV-only Qwen3.5 full-attention Sender -> Qwen3-4B native KV channel."""

    def __init__(self, cfg, variant):
        super().__init__()
        if variant not in {"s0", "s1", "s2"}:
            raise ValueError(variant)
        self.variant = variant
        self.source_layers = cfg["source_layers"]
        self.target_layers = cfg["target_layers"]
        self.target_heads = cfg["target_kv_heads"]
        self.target_dim = cfg["target_head_dim"]
        base = relative_depth_matrix(
            cfg["target_depth_anchors"], cfg["target_layers"]
        )
        self.register_buffer("depth_base", base)
        if variant != "s0":
            self.feature = SharedLowRankResidual(
                cfg["joint_dim"],
                cfg["feature_rank"],
                cfg["source_layers"],
                cfg["gate_init"],
            )
            self.calibration = SharedLowRankResidual(
                cfg["joint_dim"],
                cfg["calibration_rank"],
                cfg["target_layers"],
                cfg["gate_init"],
            )
        if variant == "s2":
            self.depth_delta = nn.Parameter(torch.zeros_like(base))
            self.depth_gate = nn.Parameter(
                torch.tensor(float(cfg["depth_gate_init"]))
            )

    def depth_weights(self, dtype):
        if self.variant != "s2":
            return self.depth_base.to(dtype)
        logits = torch.log(self.depth_base.clamp_min(1e-8))
        return torch.softmax(
            logits + self.depth_gate * self.depth_delta, dim=-1
        ).to(dtype)

    def forward(self, key, value):
        # Input has already been deterministically aligned to Qwen3-4B token slots
        # and reshaped from [4,256] to [8,128].
        if key.shape != value.shape or key.shape[0] != self.source_layers:
            raise ValueError(f"unexpected source KV shape: {tuple(key.shape)}")
        tokens = key.shape[1]
        x = torch.cat((key.reshape(self.source_layers, tokens, -1),
                       value.reshape(self.source_layers, tokens, -1)), dim=-1)
        if self.variant != "s0":
            x = self.feature(x)
        x = torch.einsum("js,std->jtd", self.depth_weights(x.dtype), x)
        if self.variant != "s0":
            x = self.calibration(x)
        width = self.target_heads * self.target_dim
        key_out = x[..., :width].reshape(
            self.target_layers, tokens, self.target_heads, self.target_dim
        )
        value_out = x[..., width:].reshape(
            self.target_layers, tokens, self.target_heads, self.target_dim
        )
        return key_out, value_out

    @torch.no_grad()
    def zero_check(self, device):
        zero = torch.zeros(
            self.source_layers, 3, self.target_heads, self.target_dim, device=device
        )
        return max(t.abs().max().item() for t in self(zero, zero))
