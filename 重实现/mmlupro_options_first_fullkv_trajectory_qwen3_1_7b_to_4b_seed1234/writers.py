from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def continuous_depth(target_layer: int, source_layers: int, target_layers: int) -> float:
    return target_layer * (source_layers - 1) / (target_layers - 1)


def nearest_source_layers(source_layers: int, target_layers: int) -> list[int]:
    return [int(math.floor(continuous_depth(j, source_layers, target_layers) + 0.5)) for j in range(target_layers)]


def interpolation_metadata(source_layers: int, target_layers: int):
    indices, weights = [], []
    for target in range(target_layers):
        position = continuous_depth(target, source_layers, target_layers)
        lower = int(math.floor(position))
        upper = min(lower + 1, source_layers - 1)
        upper_weight = position - lower
        indices.append([lower, upper])
        weights.append([1.0 - upper_weight, upper_weight])
    return indices, weights


def local_five_metadata(source_layers: int, target_layers: int) -> list[list[int]]:
    if source_layers < 5:
        raise ValueError("local-5 mapping requires at least five source layers")
    output = []
    for target in range(target_layers):
        nearest = nearest_source_layers(source_layers, target_layers)[target]
        start = min(max(nearest - 2, 0), source_layers - 5)
        indices = list(range(start, start + 5))
        if len(indices) != 5 or len(set(indices)) != 5:
            raise RuntimeError("local-5 construction failed to produce five unique layers")
        output.append(indices)
    return output


class FullKVWriter(nn.Module):
    """Per-target-layer, K/V-separated, token/head-independent linear Writer."""

    def __init__(self, kind: str, scales: dict[str, torch.Tensor], cfg: dict[str, Any]):
        super().__init__()
        if kind not in {"d0", "d1", "d2"}:
            raise ValueError(kind)
        self.kind = kind
        self.source_layers = int(cfg["source_layers"])
        self.target_layers = int(cfg["target_layers"])
        self.heads = int(cfg["num_kv_heads"])
        self.dim = int(cfg["head_dim"])
        for name in ("source_k", "source_v", "target_k", "target_v"):
            self.register_buffer(f"scale_{name}", scales[name].float().clamp_min(cfg["rms_epsilon"]), persistent=True)

        nearest = nearest_source_layers(self.source_layers, self.target_layers)
        d1_indices, d1_weights = interpolation_metadata(self.source_layers, self.target_layers)
        d2_indices = local_five_metadata(self.source_layers, self.target_layers)
        self.register_buffer("nearest_indices", torch.tensor(nearest, dtype=torch.long), persistent=True)
        self.register_buffer("interpolation_indices", torch.tensor(d1_indices, dtype=torch.long), persistent=True)
        self.register_buffer("interpolation_weights", torch.tensor(d1_weights, dtype=torch.float32), persistent=True)
        self.register_buffer("local_source_indices", torch.tensor(d2_indices, dtype=torch.long), persistent=True)

        input_dim = self.dim * (5 if kind == "d2" else 1)
        self.k_linears = nn.ModuleList([
            nn.Linear(input_dim, self.dim, bias=False) for _ in range(self.target_layers)
        ])
        self.v_linears = nn.ModuleList([
            nn.Linear(input_dim, self.dim, bias=False) for _ in range(self.target_layers)
        ])
        self.reset_calibrated_identity()

    def reset_calibrated_identity(self) -> None:
        identity = torch.eye(self.dim)
        with torch.no_grad():
            for target in range(self.target_layers):
                for family in (self.k_linears, self.v_linears):
                    family[target].weight.zero_()
                    if self.kind in {"d0", "d1"}:
                        family[target].weight.copy_(identity)
                    else:
                        neighbors = self.local_source_indices[target].tolist()
                        nearest = int(self.nearest_indices[target])
                        block = neighbors.index(nearest)
                        begin = block * self.dim
                        family[target].weight[:, begin:begin + self.dim].copy_(identity)

    def normalize(self, key: torch.Tensor, value: torch.Tensor):
        source_k = self.scale_source_k[:, None].to(device=key.device, dtype=key.dtype)
        source_v = self.scale_source_v[:, None].to(device=value.device, dtype=value.dtype)
        return key / source_k, value / source_v

    def _features(self, normalized: torch.Tensor, target: int) -> torch.Tensor:
        if self.kind == "d0":
            return normalized[int(self.nearest_indices[target])]
        if self.kind == "d1":
            indices = self.interpolation_indices[target]
            weights = self.interpolation_weights[target].to(device=normalized.device, dtype=normalized.dtype)
            return normalized[int(indices[0])] * weights[0] + normalized[int(indices[1])] * weights[1]
        indices = self.local_source_indices[target]
        selected = normalized.index_select(0, indices)
        # [5,T,H,D] -> [T,H,5*D], with the exact persisted layer order.
        return selected.permute(1, 2, 0, 3).flatten(-2)

    @staticmethod
    def _linear_fp32(features: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        shape = features.shape[:-1]
        output = F.linear(features.float().reshape(-1, features.shape[-1]), layer.weight)
        return output.reshape(*shape, layer.out_features)

    def forward(self, key: torch.Tensor, value: torch.Tensor):
        expected = (self.source_layers, key.shape[1], self.heads, self.dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"Writer input must be {expected}; K={tuple(key.shape)} V={tuple(value.shape)}")
        output_dtype = key.dtype
        key, value = self.normalize(key, value)
        predicted_k, predicted_v = [], []
        for target in range(self.target_layers):
            predicted_k.append(self._linear_fp32(self._features(key, target), self.k_linears[target]))
            predicted_v.append(self._linear_fp32(self._features(value, target), self.v_linears[target]))
        predicted_k = torch.stack(predicted_k)
        predicted_v = torch.stack(predicted_v)
        target_k = self.scale_target_k[:, None].to(device=predicted_k.device, dtype=predicted_k.dtype)
        target_v = self.scale_target_v[:, None].to(device=predicted_v.device, dtype=predicted_v.dtype)
        return (predicted_k * target_k).to(output_dtype), (predicted_v * target_v).to(output_dtype)

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_layers": self.source_layers,
            "target_layers": self.target_layers,
            "nearest_indices": self.nearest_indices.tolist(),
            "interpolation_indices": self.interpolation_indices.tolist(),
            "interpolation_weights": self.interpolation_weights.tolist(),
            "local_source_indices": self.local_source_indices.tolist(),
            "k_v_separated": True,
            "per_target_layer_independent": True,
            "token_mixing": False,
            "head_mixing": False,
            "bias": False,
            "initialization": "calibrated nearest identity" if self.kind in {"d0", "d2"} else "calibrated fixed interpolation identity",
        }


def load_scales(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("train_only", False):
        raise RuntimeError("RMS scales are not marked train-only")
    return {name: payload[name] for name in ("source_k", "source_v", "target_k", "target_v")}


def make_writer(kind: str, cfg: dict[str, Any]) -> FullKVWriter:
    scales_path = str(cfg["work_dir"]) + "/artifacts/calibration/rms_scales.pt"
    return FullKVWriter(kind, load_scales(scales_path), cfg)


def save_writer(path: str, writer: FullKVWriter, extra: dict[str, Any] | None = None) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "writer_state": {name: value.detach().cpu() for name, value in writer.state_dict().items()},
        "writer_metadata": writer.metadata(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_writer(path: str, writer: FullKVWriter) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["writer_metadata"]["kind"] != writer.kind:
        raise RuntimeError("checkpoint Writer kind mismatch")
    if payload["writer_metadata"]["local_source_indices"] != writer.local_source_indices.tolist():
        raise RuntimeError("checkpoint neighbor ordering mismatch")
    writer.load_state_dict(payload["writer_state"], strict=True)
    return payload
