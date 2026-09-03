from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def relative_depth_matrix(anchors, target_layers):
    anchors = [float(x) for x in anchors]
    matrix = torch.zeros(target_layers, len(anchors))
    for target in range(target_layers):
        if target <= anchors[0]:
            matrix[target, 0] = 1
        elif target >= anchors[-1]:
            matrix[target, -1] = 1
        else:
            right = next(i for i, x in enumerate(anchors) if x >= target)
            left = right - 1
            fraction = (target - anchors[left]) / (anchors[right] - anchors[left])
            matrix[target, left] = 1 - fraction
            matrix[target, right] = fraction
    return matrix


class Calibration(nn.Module):
    def __init__(self, dim, rank, layers):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.embedding = nn.Parameter(torch.zeros(layers, rank))
        self.gate = nn.Parameter(torch.zeros(layers))
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.normal_(self.up.weight, std=0.01)

    def forward(self, x):
        hidden = F.linear(x, self.down.weight)
        hidden = hidden + self.embedding[:, None]
        residual = F.linear(F.silu(hidden), self.up.weight)
        return x + self.gate[:, None, None] * residual


class ScaledFullRankWriter(nn.Module):
    def __init__(self, cfg, scales, variant):
        super().__init__()
        if variant not in {"v0", "v1", "v2"}:
            raise ValueError(variant)
        self.variant = variant
        self.dim = cfg["feature_dim"]
        self.source_layers = cfg["source_layers"]
        self.target_layers = cfg["target_layers"]
        for name in ("source_k", "source_v", "target_k", "target_v"):
            self.register_buffer(f"scale_{name}", scales[name].float())
        base = relative_depth_matrix(
            cfg["target_depth_anchors"], cfg["target_layers"]
        )
        self.register_buffer("depth_base", base)
        if variant != "v0":
            eye = torch.eye(self.dim).repeat(self.source_layers, 1, 1)
            self.feature_k = nn.Parameter(eye.clone())
            self.feature_v = nn.Parameter(eye.clone())
        if variant == "v2":
            initial = torch.log(base.clamp_min(1e-8))
            self.depth_logits_k = nn.Parameter(initial.clone())
            self.depth_logits_v = nn.Parameter(initial.clone())
            self.calibration_k = Calibration(
                self.dim, cfg["calibration_rank"], self.target_layers
            )
            self.calibration_v = Calibration(
                self.dim, cfg["calibration_rank"], self.target_layers
            )

    def standardized_source(self, key, value):
        key = key.float().reshape(self.source_layers, key.shape[1], self.dim)
        value = value.float().reshape(self.source_layers, value.shape[1], self.dim)
        return (
            key / self.scale_source_k[:, None],
            value / self.scale_source_v[:, None],
        )

    def features(self, key, value):
        key, value = self.standardized_source(key, value)
        if self.variant == "v0":
            return key, value
        return (
            torch.einsum("lti,loi->lto", key, self.feature_k),
            torch.einsum("lti,loi->lto", value, self.feature_v),
        )

    def standardized_output(self, key, value):
        key, value = self.features(key, value)
        if self.variant == "v2":
            depth_k = self.depth_logits_k.softmax(-1)
            depth_v = self.depth_logits_v.softmax(-1)
        else:
            depth_k = depth_v = self.depth_base
        key = torch.einsum("jl,ltd->jtd", depth_k, key)
        value = torch.einsum("jl,ltd->jtd", depth_v, value)
        if self.variant == "v2":
            key = self.calibration_k(key)
            value = self.calibration_v(value)
        return key, value

    def forward(self, key, value):
        key, value = self.standardized_output(key, value)
        key = key * self.scale_target_k[:, None]
        value = value * self.scale_target_v[:, None]
        shape = (self.target_layers, key.shape[1], 8, 128)
        return key.reshape(shape), value.reshape(shape)

    def feature_parameters(self):
        return [self.feature_k, self.feature_v]

    def depth_parameters(self):
        if self.variant != "v2":
            return []
        return [self.depth_logits_k, self.depth_logits_v]

    def calibration_parameters(self):
        if self.variant != "v2":
            return []
        return list(self.calibration_k.parameters()) + list(
            self.calibration_v.parameters()
        )
