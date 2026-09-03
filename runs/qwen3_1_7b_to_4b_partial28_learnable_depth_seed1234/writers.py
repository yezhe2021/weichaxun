from __future__ import annotations

import torch
import torch.nn as nn


def repeat_last_indices(source_layers=28, target_layers=36):
    return torch.tensor([i if i < source_layers else source_layers - 1 for i in range(target_layers)], dtype=torch.long)


def repeat_last_matrix(source_layers=28, target_layers=36):
    matrix = torch.zeros(target_layers, source_layers, dtype=torch.float32)
    matrix[torch.arange(target_layers), repeat_last_indices(source_layers, target_layers)] = 1.0
    return matrix


class LinearDepthWriter(nn.Module):
    """Full-rank per-target-layer K/V translation with an explicit depth protocol."""

    def __init__(self, scales, cfg, protocol="repeat_last", learnable_depth=False):
        super().__init__()
        self.source_layers = int(cfg["source_layers"])
        self.target_layers = int(cfg["num_layers"])
        self.protocol = protocol
        self.learnable_depth = bool(learnable_depth)
        output_layers = self.source_layers if protocol == "skip" else self.target_layers
        identity = torch.eye(cfg["feature_dim"], dtype=torch.float32).unsqueeze(0).repeat(output_layers, 1, 1)
        self.weight_k = nn.Parameter(identity.clone())
        self.weight_v = nn.Parameter(identity.clone())
        for name in ("source_k", "source_v", "target_k", "target_v"):
            self.register_buffer(f"scale_{name}", scales[name].float().clamp_min(1e-6), persistent=True)
        self.register_buffer("repeat_indices", repeat_last_indices(self.source_layers, self.target_layers), persistent=True)
        self.register_buffer("depth_base", repeat_last_matrix(self.source_layers, self.target_layers), persistent=True)
        if self.learnable_depth:
            self.depth_delta = nn.Parameter(torch.zeros(self.target_layers, self.source_layers))

    def depth_matrix(self):
        if self.learnable_depth:
            return self.depth_base + self.depth_delta
        return self.depth_base

    def _normalized_sources(self, key, value):
        shape = (self.source_layers, 1, 8, 128)
        return (
            key.float() / self.scale_source_k.view(shape),
            value.float() / self.scale_source_v.view(shape),
        )

    def _route(self, key, value):
        if self.protocol == "skip":
            return key, value
        if self.learnable_depth:
            matrix = self.depth_matrix()
            return torch.einsum("li,ithd->lthd", matrix, key), torch.einsum("li,ithd->lthd", matrix, value)
        return key[self.repeat_indices], value[self.repeat_indices]

    def forward(self, key, value):
        output_dtype = key.dtype
        key, value = self._normalized_sources(key, value)
        key, value = self._route(key, value)
        k = torch.einsum("lti,loi->lto", key.flatten(2), self.weight_k).view_as(key)
        v = torch.einsum("lti,loi->lto", value.flatten(2), self.weight_v).view_as(value)
        layers = k.shape[0]
        shape = (layers, 1, 8, 128)
        k = k * self.scale_target_k[:layers].view(shape)
        v = v * self.scale_target_v[:layers].view(shape)
        return k.to(output_dtype), v.to(output_dtype)


def make_writer(kind, scales, cfg):
    if kind == "skip":
        return LinearDepthWriter(scales, cfg, protocol="skip", learnable_depth=False)
    if kind in ("repeat", "fixed_continued"):
        return LinearDepthWriter(scales, cfg, protocol="repeat_last", learnable_depth=False)
    if kind == "learnable_matrix":
        return LinearDepthWriter(scales, cfg, protocol="repeat_last", learnable_depth=True)
    raise ValueError(kind)


def load_writer(writer, checkpoint, strict=True):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state["writer"] if "writer" in state else state
    writer.load_state_dict(state, strict=strict)
    return writer


def copy_repeat_into_learnable(writer, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state["writer"] if "writer" in state else state
    own = writer.state_dict()
    for name in ("weight_k", "weight_v"):
        own[name].copy_(state[name].float())
    writer.load_state_dict(own, strict=True)
    return writer


def optimizer_groups(writer, linear_lr, depth_lr=None):
    groups = [{"params": [writer.weight_k, writer.weight_v], "lr": linear_lr}]
    if writer.learnable_depth:
        groups.append({"params": [writer.depth_delta], "lr": depth_lr})
    return groups


def parameter_report(writer):
    return {
        "trainable_parameters": sum(p.numel() for p in writer.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in writer.parameters()),
        "learnable_depth_parameters": writer.depth_delta.numel() if writer.learnable_depth else 0,
    }
