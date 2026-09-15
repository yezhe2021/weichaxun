from __future__ import annotations

import torch
import torch.nn as nn


class AnchorWriter(nn.Module):
    """Exactly eight independent K/V 1024x1024 feature maps."""

    def __init__(self, cfg, scales):
        super().__init__()
        self.anchor_layers = tuple(cfg["anchor_layers"])
        self.dim = int(cfg["feature_dim"])
        for name in ("source_k", "source_v"):
            self.register_buffer(f"scale_{name}", scales[name].float())
        self.register_buffer("scale_target_k", scales["target_k"][list(self.anchor_layers)].float())
        self.register_buffer("scale_target_v", scales["target_v"][list(self.anchor_layers)].float())
        eye = torch.eye(self.dim).repeat(8, 1, 1)
        self.feature_k = nn.Parameter(eye.clone())
        self.feature_v = nn.Parameter(eye.clone())

    def load_a1(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)["writer"]
        with torch.no_grad():
            self.feature_k.copy_(state["feature_k"])
            self.feature_v.copy_(state["feature_v"])

    def standardized(self, key, value):
        tokens = key.shape[1]
        key = key.float().reshape(8, tokens, self.dim) / self.scale_source_k[:, None]
        value = value.float().reshape(8, tokens, self.dim) / self.scale_source_v[:, None]
        return (
            torch.einsum("lti,loi->lto", key, self.feature_k),
            torch.einsum("lti,loi->lto", value, self.feature_v),
        )

    def forward(self, key, value):
        key, value = self.standardized(key, value)
        key = key * self.scale_target_k[:, None]
        value = value * self.scale_target_v[:, None]
        shape = (8, key.shape[1], 8, 128)
        return key.reshape(shape), value.reshape(shape)

    def zero_check(self):
        key = torch.zeros(8, 2, 4, 256, device=self.feature_k.device)
        value = torch.zeros_like(key)
        output = self(key, value)
        return max(output[0].abs().max().item(), output[1].abs().max().item())


def load_scales(cfg, mode):
    return torch.load(
        f'{cfg["v2_dir"]}/artifacts/{mode}/scales.pt',
        map_location="cpu", weights_only=False,
    )


def make_writer(cfg, mode, initialization):
    writer = AnchorWriter(cfg, load_scales(cfg, mode)).cuda()
    if initialization == "a1":
        writer.load_a1(f'{cfg["v2_dir"]}/artifacts/{mode}/a1/best.pt')
    elif initialization != "identity":
        raise ValueError(initialization)
    return writer
