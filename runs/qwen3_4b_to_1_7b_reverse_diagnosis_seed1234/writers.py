from __future__ import annotations

import torch
import torch.nn as nn


def front28_matrix(target_layers=28, source_layers=36):
    matrix = torch.zeros(target_layers, source_layers, dtype=torch.float32)
    matrix[torch.arange(target_layers), torch.arange(target_layers)] = 1.0
    return matrix


class ReverseWriter(nn.Module):
    def __init__(self, scales, cfg, learnable_depth=False):
        super().__init__(); self.learnable_depth = bool(learnable_depth)
        self.source_layers, self.target_layers = cfg["source_layers"], cfg["target_layers"]
        identity = torch.eye(cfg["feature_dim"], dtype=torch.float32)[None].repeat(self.target_layers, 1, 1)
        self.weight_k = nn.Parameter(identity.clone()); self.weight_v = nn.Parameter(identity.clone())
        for name in ("source_k", "source_v", "target_k", "target_v"):
            self.register_buffer(f"scale_{name}", scales[name].float().clamp_min(1e-6), persistent=True)
        self.register_buffer("depth_base", front28_matrix(self.target_layers, self.source_layers), persistent=True)
        if self.learnable_depth: self.depth_delta = nn.Parameter(torch.zeros(self.target_layers, self.source_layers))

    def depth_matrix(self): return self.depth_base + self.depth_delta if self.learnable_depth else self.depth_base

    def forward(self, key, value):
        dtype = key.dtype; source_shape = (self.source_layers, 1, 8, 128); target_shape = (self.target_layers, 1, 8, 128)
        key = key.float() / self.scale_source_k.view(source_shape); value = value.float() / self.scale_source_v.view(source_shape)
        if self.learnable_depth:
            matrix = self.depth_matrix(); key = torch.einsum("li,ithd->lthd", matrix, key); value = torch.einsum("li,ithd->lthd", matrix, value)
        else: key, value = key[:self.target_layers], value[:self.target_layers]
        key = torch.einsum("lti,loi->lto", key.flatten(2), self.weight_k).view_as(key)
        value = torch.einsum("lti,loi->lto", value.flatten(2), self.weight_v).view_as(value)
        return (key * self.scale_target_k.view(target_shape)).to(dtype), (value * self.scale_target_v.view(target_shape)).to(dtype)


def make_writer(kind, scales, cfg): return ReverseWriter(scales, cfg, learnable_depth=(kind == "learnable"))
def load_writer(writer, path):
    state = torch.load(path, map_location="cpu", weights_only=False); writer.load_state_dict(state["writer"] if "writer" in state else state, strict=True); return writer
def copy_front_into_learnable(writer, path):
    state = torch.load(path, map_location="cpu", weights_only=False); state = state["writer"] if "writer" in state else state
    own = writer.state_dict(); own["weight_k"].copy_(state["weight_k"]); own["weight_v"].copy_(state["weight_v"]); writer.load_state_dict(own); return writer
def optimizer_groups(writer, linear_lr, depth_lr):
    groups = [{"params": [writer.weight_k, writer.weight_v], "lr": linear_lr}]
    if writer.learnable_depth: groups.append({"params": [writer.depth_delta], "lr": depth_lr})
    return groups
