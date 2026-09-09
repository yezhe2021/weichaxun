from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualHeadMLP(nn.Module):
    def __init__(self, dim, hidden, gamma):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.down = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(hidden, dim, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma), dtype=torch.float32))

    def forward(self, x):
        dtype = x.dtype
        normalized = F.rms_norm(x, (x.shape[-1],), self.norm.weight.to(dtype), self.norm.eps)
        hidden = F.linear(normalized, self.down.weight.to(dtype))
        residual = F.linear(F.silu(hidden), self.up.weight.to(dtype))
        return x + self.gamma.to(dtype) * residual


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


class FullWriter(WriterBase):
    def __init__(self, scales, cfg):
        super().__init__(scales)
        layers, heads = cfg["num_layers"], cfg["num_kv_heads"]
        identity = torch.eye(heads).unsqueeze(0).repeat(layers, 1, 1)
        self.head_k = nn.Parameter(identity.clone())
        self.head_v = nn.Parameter(identity.clone())
        self.feature_k = nn.ModuleList([ResidualHeadMLP(cfg["head_dim"], cfg["full_hidden_dim"], cfg["residual_gamma_init"]) for _ in range(layers)])
        self.feature_v = nn.ModuleList([ResidualHeadMLP(cfg["head_dim"], cfg["full_hidden_dim"], cfg["residual_gamma_init"]) for _ in range(layers)])
        mask = torch.full((layers, layers), float("-inf"))
        logits_k = torch.full((layers, layers), -8.0)
        logits_v = torch.full((layers, layers), -8.0)
        for target in range(layers):
            for source in range(max(0, target-cfg["layer_window"]), min(layers, target+cfg["layer_window"]+1)):
                mask[target, source] = 0.0
                logits_k[target, source] = cfg["layer_identity_logit"] if source == target else 0.0
                logits_v[target, source] = cfg["layer_identity_logit"] if source == target else 0.0
        self.register_buffer("layer_mask", mask, persistent=True)
        self.layer_k = nn.Parameter(logits_k)
        self.layer_v = nn.Parameter(logits_v)
        self.calibration_k = nn.ModuleList([ResidualHeadMLP(cfg["head_dim"], cfg["full_hidden_dim"], cfg["residual_gamma_init"]) for _ in range(layers)])
        self.calibration_v = nn.ModuleList([ResidualHeadMLP(cfg["head_dim"], cfg["full_hidden_dim"], cfg["residual_gamma_init"]) for _ in range(layers)])

    @staticmethod
    def apply_modules(values, modules):
        return torch.stack([module(values[layer]) for layer, module in enumerate(modules)])

    def forward(self, key, value):
        output_dtype = key.dtype
        # Keep the gated structural path in FP32. With a zero-initialized gate,
        # the second-step gradients reaching MLP weights can underflow in FP16.
        key, value = self.normalize(key.float(), value.float())
        key = torch.einsum("lhj,ltjd->lthd", self.head_k.to(key.dtype), key)
        value = torch.einsum("lhj,ltjd->lthd", self.head_v.to(value.dtype), value)
        key = self.apply_modules(key, self.feature_k)
        value = self.apply_modules(value, self.feature_v)
        alpha_k = torch.softmax(self.layer_k + self.layer_mask, -1).to(key.dtype)
        alpha_v = torch.softmax(self.layer_v + self.layer_mask, -1).to(value.dtype)
        key = torch.einsum("li,ithd->lthd", alpha_k, key)
        value = torch.einsum("li,ithd->lthd", alpha_v, value)
        key = self.apply_modules(key, self.calibration_k)
        value = self.apply_modules(value, self.calibration_v)
        key, value = self.restore(key, value)
        return key.to(output_dtype), value.to(output_dtype)

    def diagnostics(self):
        return {
            "head_mixer_k": self.head_k.detach().float().cpu().tolist(),
            "head_mixer_v": self.head_v.detach().float().cpu().tolist(),
            "layer_mixer_k": torch.softmax(self.layer_k.detach()+self.layer_mask, -1).float().cpu().tolist(),
            "layer_mixer_v": torch.softmax(self.layer_v.detach()+self.layer_mask, -1).float().cpu().tolist(),
            "feature_gamma_k": [x.gamma.item() for x in self.feature_k],
            "feature_gamma_v": [x.gamma.item() for x in self.feature_v],
            "calibration_gamma_k": [x.gamma.item() for x in self.calibration_k],
            "calibration_gamma_v": [x.gamma.item() for x in self.calibration_v],
        }


def make_writer(kind, scales, cfg):
    if kind == "linear": return LinearWriter(scales, cfg)
    if kind == "full": return FullWriter(scales, cfg)
    raise ValueError(kind)


def rms_only(key, value, scales):
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
