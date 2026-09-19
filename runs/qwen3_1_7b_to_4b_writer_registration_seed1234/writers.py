from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LinearCore(WriterBase):
    def __init__(self, scales, cfg):
        super().__init__(scales)
        identity = torch.eye(cfg["feature_dim"], dtype=torch.float32).unsqueeze(0).repeat(cfg["num_layers"], 1, 1)
        self.weight_k = nn.Parameter(identity.clone())
        self.weight_v = nn.Parameter(identity.clone())

    def linear(self, key, value):
        key, value = self.normalize(key.float(), value.float())
        k = torch.einsum("lti,loi->lto", key.flatten(2), self.weight_k).view_as(key)
        v = torch.einsum("lti,loi->lto", value.flatten(2), self.weight_v).view_as(value)
        return k, v

    def forward(self, key, value):
        output_dtype = key.dtype
        key, value = self.linear(key, value)
        key, value = self.restore(key, value)
        return key.to(output_dtype), value.to(output_dtype)


class WriterV2(LinearCore):
    """Linear core plus zero-initialized per-head and optional local-layer residuals."""

    OFFSETS = (-2, -1, 1, 2)

    def __init__(self, scales, cfg, use_local):
        super().__init__(scales, cfg)
        layers, heads, dim = cfg["num_layers"], cfg["num_kv_heads"], cfg["head_dim"]
        self.use_local = bool(use_local)
        self.norm_k = nn.Parameter(torch.ones(layers, heads, dim))
        self.norm_v = nn.Parameter(torch.ones(layers, heads, dim))
        std = float(cfg.get("v2_w1_init_std", 0.02))
        self.w1_k = nn.Parameter(torch.empty(layers, heads, dim, dim).normal_(0.0, std))
        self.w1_v = nn.Parameter(torch.empty(layers, heads, dim, dim).normal_(0.0, std))
        self.w2_k = nn.Parameter(torch.zeros(layers, heads, dim, dim))
        self.w2_v = nn.Parameter(torch.zeros(layers, heads, dim, dim))
        if self.use_local:
            self.beta_k = nn.Parameter(torch.zeros(layers, heads, len(self.OFFSETS)))
            self.beta_v = nn.Parameter(torch.zeros(layers, heads, len(self.OFFSETS)))
            valid = torch.zeros(layers, len(self.OFFSETS), dtype=torch.bool)
            for layer in range(layers):
                for index, delta in enumerate(self.OFFSETS):
                    valid[layer, index] = 0 <= layer + delta < layers
            self.register_buffer("valid_local", valid, persistent=True)

    def local_residual(self, values, beta):
        if not self.use_local:
            return values
        output = values.clone()
        for layer in range(values.shape[0]):
            for index, delta in enumerate(self.OFFSETS):
                source = layer + delta
                if 0 <= source < values.shape[0]:
                    output[layer] = output[layer] + beta[layer, :, index][None, :, None] * values[source]
        return output

    @staticmethod
    def head_residual(values, norm_weight, w1, w2):
        normalized = F.rms_norm(values, (values.shape[-1],), eps=1e-6)
        normalized = normalized * norm_weight[:, None]
        hidden = torch.einsum("lthi,lhji->lthj", normalized, w1)
        residual = torch.einsum("lthi,lhji->lthj", F.silu(hidden), w2)
        return values + residual

    def forward(self, key, value):
        output_dtype = key.dtype
        key, value = self.linear(key, value)
        key = self.local_residual(key, self.beta_k) if self.use_local else key
        value = self.local_residual(value, self.beta_v) if self.use_local else value
        key = self.head_residual(key, self.norm_k, self.w1_k, self.w2_k)
        value = self.head_residual(value, self.norm_v, self.w1_v, self.w2_v)
        key, value = self.restore(key, value)
        return key.to(output_dtype), value.to(output_dtype)

    def diagnostics(self):
        result = {
            "w2_k_rms": self.w2_k.detach().float().square().mean().sqrt().item(),
            "w2_v_rms": self.w2_v.detach().float().square().mean().sqrt().item(),
            "uses_local_layer_residual": self.use_local,
        }
        if self.use_local:
            mask = self.valid_local[:, None, :].expand_as(self.beta_k)
            result.update({
                "beta_k_rms_valid": self.beta_k.detach().float()[mask].square().mean().sqrt().item(),
                "beta_v_rms_valid": self.beta_v.detach().float()[mask].square().mean().sqrt().item(),
            })
        return result


def make_writer(kind, scales, cfg):
    if kind == "linear_continued": return LinearCore(scales, cfg)
    if kind == "v2_h": return WriterV2(scales, cfg, use_local=False)
    if kind == "v2_hl": return WriterV2(scales, cfg, use_local=True)
    raise ValueError(kind)


def load_linear_core(writer, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state["writer"] if "writer" in state else state
    required = ("weight_k", "weight_v")
    missing = [name for name in required if name not in state]
    if missing: raise RuntimeError(f"linear checkpoint missing {missing}")
    with torch.no_grad():
        writer.weight_k.copy_(state["weight_k"].float())
        writer.weight_v.copy_(state["weight_v"].float())
    return writer


def linear_parameters(writer):
    return [writer.weight_k, writer.weight_v]


def residual_parameters(writer):
    linear_ids = {id(writer.weight_k), id(writer.weight_v)}
    return [parameter for parameter in writer.parameters() if parameter.requires_grad and id(parameter) not in linear_ids]


def parameter_report(writer):
    trainable = sum(p.numel() for p in writer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in writer.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total}
