"""Writer 结构（方案 §七）。

第一轮只使用逐层全秩 Linear Writer（已在 8B→4B 中验证）。
不加入：深度插值 / 层路由 / Head MLP / Layer Residual / Writer v2 / LoRA / Reader / KD。

对每个 token、每一层：
    x_l^K = K_l^S / s_l^{S,K}          # 用 Sender RMS scale 归一化（逐 feature）
    y_l^K = W_l^K x_l^K                # 逐层全秩线性映射
    Ŷ_l^K = y_l^K ⊙ s_l^{R,K}          # 恢复 Receiver 尺度

W_l^K, W_l^V ∈ R^{1024×1024}，从 Identity 初始化（方案 §七：保证初始稳定，允许训练偏离）。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class WriterBase(nn.Module):
    def __init__(self, scales):
        super().__init__()
        for name in ("source_k", "source_v", "target_k", "target_v"):
            self.register_buffer(f"scale_{name}", scales[name].float().clamp_min(1e-6), persistent=True)

    def normalize(self, key, value):
        shape = (key.shape[0], 1, 8, 128)
        return (
            key / self.scale_source_k.view(shape).to(key.dtype),
            value / self.scale_source_v.view(shape).to(value.dtype),
        )

    def restore(self, key, value):
        shape = (key.shape[0], 1, 8, 128)
        return (
            key * self.scale_target_k.view(shape).to(key.dtype),
            value * self.scale_target_v.view(shape).to(value.dtype),
        )


class LinearWriter(WriterBase):
    """逐层全秩 Linear Writer：W^K, W^V ∈ R^{num_layers × feature_dim × feature_dim}，Identity 初始化。"""

    def __init__(self, scales, cfg):
        super().__init__(scales)
        identity = torch.eye(cfg["feature_dim"], dtype=torch.float32).unsqueeze(0).repeat(cfg["num_layers"], 1, 1)
        self.weight_k = nn.Parameter(identity.clone())
        self.weight_v = nn.Parameter(identity.clone())

    def forward(self, key, value):
        key, value = self.normalize(key, value)
        k = torch.einsum("lti,loi->lto", key.flatten(2), self.weight_k.to(key.dtype)).view_as(key)
        v = torch.einsum("lti,loi->lto", value.flatten(2), self.weight_v.to(value.dtype)).view_as(value)
        return self.restore(k, v)


def make_writer(kind, scales, cfg):
    if kind == "linear":
        return LinearWriter(scales, cfg)
    raise ValueError(kind)


def rms_only(key, value, scales):
    """仅做 RMS scale 交换的控制（不学习）：K*scale 直接替换为目标尺度。"""
    shape = (key.shape[0], 1, 8, 128)
    key = key / scales["source_k"].view(shape).to(device=key.device, dtype=key.dtype)
    value = value / scales["source_v"].view(shape).to(device=value.device, dtype=value.dtype)
    return (
        key * scales["target_k"].view(shape).to(device=key.device, dtype=key.dtype),
        value * scales["target_v"].view(shape).to(device=value.device, dtype=value.dtype),
    )


def parameter_report(writer):
    trainable = sum(p.numel() for p in writer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in writer.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total}


def save_writer(path, writer, cfg, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "linear",
            "cfg": cfg,
            "state_dict": writer.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_writer(path, map_location="cpu"):
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    scales = {
        "source_k": checkpoint["state_dict"]["scale_source_k"],
        "source_v": checkpoint["state_dict"]["scale_source_v"],
        "target_k": checkpoint["state_dict"]["scale_target_k"],
        "target_v": checkpoint["state_dict"]["scale_target_v"],
    }
    writer = LinearWriter(scales, checkpoint["cfg"])
    writer.load_state_dict(checkpoint["state_dict"])
    return writer, checkpoint.get("metadata", {})
