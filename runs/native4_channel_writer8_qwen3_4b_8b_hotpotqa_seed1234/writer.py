from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    def __init__(self, dim=256, hidden=512, gate=0.01):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate)))

    def forward(self, x):
        weight = self.norm.weight.to(x.dtype)
        z = F.rms_norm(x, (x.shape[-1],), weight, self.norm.eps)
        z = F.linear(F.silu(F.linear(z, self.up.weight.to(x.dtype))), self.down.weight.to(x.dtype))
        return x + self.gate.to(x.dtype) * z


class HeadBlock(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.norm2 = nn.RMSNorm(dim)
        self.ffn = ResidualMLP(dim, 512, gate=1.0)
        self.attention_gate = nn.Parameter(torch.tensor(0.01))
        self.ffn_gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        layers, tokens, heads, dim = x.shape
        z = x.reshape(layers * tokens, heads, dim)
        n1 = F.rms_norm(z, (dim,), self.norm1.weight.to(z.dtype), self.norm1.eps)
        attended = self.attention(
            n1.float(), n1.float(), n1.float(), need_weights=False
        )[0].to(z.dtype)
        z = z + self.attention_gate.to(z.dtype) * attended
        n2 = F.rms_norm(z, (dim,), self.norm2.weight.to(z.dtype), self.norm2.eps)
        # Use the FFN body explicitly so this block has a single outer residual gate.
        hidden = F.linear(
            F.silu(F.linear(n2, self.ffn.up.weight.to(z.dtype))),
            self.ffn.down.weight.to(z.dtype),
        )
        z = z + self.ffn_gate.to(z.dtype) * hidden
        return z.reshape(layers, tokens, heads, dim)


class Native4ChannelWriter(nn.Module):
    def __init__(self, cfg, stats, variant):
        super().__init__()
        if variant not in {"linear", "mlp", "full"}:
            raise ValueError(variant)
        self.variant = variant
        for name in ("scale8_k", "scale8_v", "scale4_k", "scale4_v"):
            self.register_buffer(name, stats[name].float().clone())
        self.linear = nn.Linear(256, 256, bias=False)
        nn.init.eye_(self.linear.weight)
        self.input_block = ResidualMLP()
        self.head_block = HeadBlock()
        self.feature1 = ResidualMLP()
        self.feature2 = ResidualMLP()
        self.out_k = nn.Linear(256, 128, bias=False)
        self.out_v = nn.Linear(256, 128, bias=False)
        with torch.no_grad():
            self.out_k.weight.zero_()
            self.out_v.weight.zero_()
            self.out_k.weight[:, :128].copy_(torch.eye(128))
            self.out_v.weight[:, 128:].copy_(torch.eye(128))

    def forward(self, key, value):
        dtype = key.dtype
        k = key / self.scale8_k[:, None].to(dtype)
        v = value / self.scale8_v[:, None].to(dtype)
        x = torch.cat((k, v), -1)
        if self.variant == "linear":
            x = F.linear(x, self.linear.weight.to(dtype))
        else:
            x = self.input_block(x)
            if self.variant == "full":
                x = self.head_block(x)
            x = self.feature2(self.feature1(x))
        k4 = F.linear(x, self.out_k.weight.to(dtype))
        v4 = F.linear(x, self.out_v.weight.to(dtype))
        return (
            k4 * self.scale4_k[:, None].to(dtype),
            v4 * self.scale4_v[:, None].to(dtype),
        )

    @torch.no_grad()
    def zero_check(self):
        shape = (36, 3, 8, 128)
        zero = torch.zeros(shape, device=next(self.parameters()).device)
        output = self(zero, zero)
        return max(t.abs().max().item() for t in output)
