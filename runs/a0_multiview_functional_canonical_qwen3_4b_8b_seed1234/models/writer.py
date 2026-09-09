import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    def __init__(self, dim, hidden, gamma):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma), dtype=torch.float32))

    def forward(self, x):
        normalized = F.rms_norm(
            x, (x.shape[-1],), self.norm.weight.to(x.dtype), self.norm.eps
        )
        hidden = F.linear(normalized, self.fc1.weight.to(x.dtype), None)
        residual = F.linear(F.silu(hidden), self.fc2.weight.to(x.dtype), None)
        return x + self.gamma.to(x.dtype) * residual


def gaussian_depth_logits(layers, sigma):
    positions = torch.arange(layers, dtype=torch.float32)
    distance = positions[:, None] - positions[None, :]
    return -(distance.square()) / (2 * float(sigma) ** 2)


class OneStreamTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers, heads, dim = cfg["num_layers"], cfg["num_kv_heads"], cfg["head_dim"]
        self.input_norms = nn.ModuleList([nn.RMSNorm(dim) for _ in range(layers)])
        self.depth_logits = nn.Parameter(
            gaussian_depth_logits(layers, cfg["depth_sigma"])
        )
        identity = torch.eye(heads).unsqueeze(0).repeat(layers, 1, 1)
        self.head_mixer = nn.Parameter(identity)
        self.projectors = nn.ModuleList(
            [
                FeatureProjector(dim, cfg["hidden_dim"], cfg["gamma_init"])
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        normalized = torch.stack(
            [
                F.rms_norm(
                    x[index],
                    (x.shape[-1],),
                    norm.weight.to(x.dtype),
                    norm.eps,
                )
                for index, norm in enumerate(self.input_norms)
            ]
        )
        depth = torch.einsum(
            "dl,lthf->dthf",
            self.depth_logits.softmax(-1).to(x.dtype),
            normalized,
        )
        heads = torch.einsum(
            "doh,dthf->dtof", self.head_mixer.to(x.dtype), depth
        )
        return torch.stack(
            [projector(heads[index]) for index, projector in enumerate(self.projectors)]
        )


class ProtocolTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k = OneStreamTransform(cfg)
        self.v = OneStreamTransform(cfg)

    def forward(self, key, value):
        return self.k(key), self.v(value)
