from __future__ import annotations

import torch
import torch.nn as nn


class Q3ToQ35Writer(nn.Module):
    """Eight independent bias-free full-rank K/V feature maps; nothing else."""

    def __init__(self, cfg, scales):
        super().__init__()
        self.dim = int(cfg["feature_dim"])
        for name in ("source_k", "source_v", "target_k", "target_v"):
            self.register_buffer(f"scale_{name}", scales[name].float())
        eye = torch.eye(self.dim).repeat(8, 1, 1)
        self.feature_k = nn.Parameter(eye.clone())
        self.feature_v = nn.Parameter(eye.clone())

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
        shape = (8, key.shape[1], 4, 256)
        return key.reshape(shape), value.reshape(shape)

    def zero_check(self):
        x = torch.zeros(8, 2, 8, 128, device=self.feature_k.device)
        key, value = self(x, x)
        return max(key.abs().max().item(), value.abs().max().item())


def make_writer(cfg, mode, checkpoint=None):
    scales = torch.load(
        f'{cfg["work_dir"]}/artifacts/{mode}/scales.pt',
        map_location="cpu", weights_only=False,
    )
    writer = Q3ToQ35Writer(cfg, scales).cuda()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["writer"]
        writer.load_state_dict(state)
    return writer
