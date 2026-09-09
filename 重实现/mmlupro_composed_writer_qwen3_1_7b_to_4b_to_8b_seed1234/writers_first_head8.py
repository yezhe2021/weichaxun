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
        if kind not in {"d0", "d1", "d2", "full28"}:
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

        if kind == "full28":
            input_dim = self.dim * self.source_layers
        else:
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
                        nearest = int(self.nearest_indices[target])
                        if self.kind == "full28":
                            block = nearest
                        else:
                            neighbors = self.local_source_indices[target].tolist()
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
        if self.kind in {"full28", "full28_head8"}:
            # [28,T,H,D] -> [T,H,28*D]. Layer is the only concatenated axis.
            return normalized.permute(1, 2, 0, 3).flatten(-2)
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
        # Materialize/cast Full-28 features once per family, then reuse them for
        # all 36 independent target projections.
        shared_key_features = self._features(key, 0).float() if self.kind in {"full28", "full28_head8"} else None
        shared_value_features = self._features(value, 0).float() if self.kind in {"full28", "full28_head8"} else None
        predicted_k, predicted_v = [], []
        for target in range(self.target_layers):
            key_features = shared_key_features if shared_key_features is not None else self._features(key, target)
            value_features = shared_value_features if shared_value_features is not None else self._features(value, target)
            predicted_k.append(self._linear_fp32(key_features, self.k_linears[target]))
            predicted_v.append(self._linear_fp32(value_features, self.v_linears[target]))
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
            "full_source_indices": list(range(self.source_layers)) if self.kind == "full28" else None,
            "input_features_per_head": self.k_linears[0].in_features,
            "k_v_separated": True,
            "per_target_layer_independent": True,
            "token_mixing": False,
            "head_mixing": False,
            "bias": False,
            "initialization": (
                "calibrated fixed interpolation identity"
                if self.kind == "d1" else "calibrated nearest identity"
            ),
        }


class Full28Head8Writer(FullKVWriter):
    """Full-depth projection followed by per-target-head full-head mixing.

    The frozen depth stage is exactly the existing Full-28 Writer.  For each
    target layer, its eight normalized latent heads are concatenated and an
    independent bias-free 1024->128 map is applied for every target head.
    K and V never share parameters.
    """

    def __init__(self, scales: dict[str, torch.Tensor], cfg: dict[str, Any]):
        super().__init__("full28", scales, cfg)
        self.kind = "full28_head8"
        head_input = self.heads * self.dim
        self.k_head_linears = nn.ModuleList([
            nn.Linear(head_input, self.dim, bias=False)
            for _target in range(self.target_layers) for _head in range(self.heads)
        ])
        self.v_head_linears = nn.ModuleList([
            nn.Linear(head_input, self.dim, bias=False)
            for _target in range(self.target_layers) for _head in range(self.heads)
        ])
        self.reset_head_identity()
        self.freeze_depth()
        self.base_full28_checkpoint: str | None = None

    def _head_linear(self, family: nn.ModuleList, target: int, head: int) -> nn.Linear:
        return family[target * self.heads + head]

    def reset_head_identity(self) -> None:
        identity = torch.eye(self.dim)
        with torch.no_grad():
            for target in range(self.target_layers):
                for head in range(self.heads):
                    for family in (self.k_head_linears, self.v_head_linears):
                        layer = self._head_linear(family, target, head)
                        layer.weight.zero_()
                        begin = head * self.dim
                        layer.weight[:, begin:begin + self.dim].copy_(identity)

    def freeze_depth(self) -> None:
        for family in (self.k_linears, self.v_linears):
            for layer in family:
                layer.weight.requires_grad_(False)

    def load_base_full28(self, path: str) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload["writer_metadata"]
        if metadata["kind"] != "full28":
            raise RuntimeError("head extension base checkpoint is not Full-28")
        state = payload["writer_state"]
        with torch.no_grad():
            for target in range(self.target_layers):
                self.k_linears[target].weight.copy_(state[f"k_linears.{target}.weight"])
                self.v_linears[target].weight.copy_(state[f"v_linears.{target}.weight"])
        self.base_full28_checkpoint = path
        self.freeze_depth()
        return payload

    def _mix_heads(self, latent: torch.Tensor, family: nn.ModuleList, target: int) -> torch.Tensor:
        # latent [T,8,128] -> features [T,1024]. Every target head owns its W.
        features = latent.flatten(-2).float()
        return torch.stack([
            self._linear_fp32(features, self._head_linear(family, target, head))
            for head in range(self.heads)
        ], dim=1)

    def forward(self, key: torch.Tensor, value: torch.Tensor):
        expected = (self.source_layers, key.shape[1], self.heads, self.dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"Writer input must be {expected}; K={tuple(key.shape)} V={tuple(value.shape)}")
        output_dtype = key.dtype
        key, value = self.normalize(key, value)
        key_features = self._features(key, 0).float()
        value_features = self._features(value, 0).float()
        predicted_k, predicted_v = [], []
        for target in range(self.target_layers):
            depth_k = self._linear_fp32(key_features, self.k_linears[target])
            depth_v = self._linear_fp32(value_features, self.v_linears[target])
            predicted_k.append(self._mix_heads(depth_k, self.k_head_linears, target))
            predicted_v.append(self._mix_heads(depth_v, self.v_head_linears, target))
        predicted_k, predicted_v = torch.stack(predicted_k), torch.stack(predicted_v)
        target_k = self.scale_target_k[:, None].to(device=predicted_k.device, dtype=predicted_k.dtype)
        target_v = self.scale_target_v[:, None].to(device=predicted_v.device, dtype=predicted_v.dtype)
        return (predicted_k * target_k).to(output_dtype), (predicted_v * target_v).to(output_dtype)

    def head_mixing_diagnostics(self) -> dict[str, Any]:
        rows = []
        with torch.no_grad():
            for target in range(self.target_layers):
                for head in range(self.heads):
                    row: dict[str, Any] = {"target_layer": target, "target_head": head}
                    for label, family in (("k", self.k_head_linears), ("v", self.v_head_linears)):
                        weight = self._head_linear(family, target, head).weight.float()
                        norms = [weight[:, source * self.dim:(source + 1) * self.dim].norm().item()
                                 for source in range(self.heads)]
                        total = sum(value * value for value in norms) ** 0.5
                        same, cross = norms[head], sum(value * value for index, value in enumerate(norms) if index != head) ** 0.5
                        probabilities = torch.tensor(norms).square()
                        probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
                        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()
                        row[label] = {
                            "source_head_block_norms": norms, "same_head_norm": same,
                            "cross_head_norm": cross, "cross_over_total": cross / max(total, 1e-12),
                            "largest_source_head": int(torch.tensor(norms).argmax()),
                            "mixing_entropy": entropy,
                        }
                    rows.append(row)
        return {"rows": rows, "row_count": len(rows)}

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update({
            "kind": self.kind, "full_source_indices": list(range(self.source_layers)),
            "head_mixing": True, "head_input_features": self.heads * self.dim,
            "per_target_head_independent": True, "depth_projection_frozen": True,
            "depth_initialization_checkpoint": self.base_full28_checkpoint,
            "head_initialization": "same-head block identity; all cross-head blocks zero",
        })
        return metadata


def load_scales(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("train_only", False):
        raise RuntimeError("RMS scales are not marked train-only")
    return {name: payload[name] for name in ("source_k", "source_v", "target_k", "target_v")}


def make_writer(kind: str, cfg: dict[str, Any]) -> FullKVWriter:
    scales_path = cfg.get(
        "calibration_path",
        str(cfg["work_dir"]) + "/artifacts/calibration/rms_scales.pt",
    )
    scales = load_scales(scales_path)
    if kind == "full28_head8":
        writer = Full28Head8Writer(scales, cfg)
        writer.load_base_full28(cfg["base_stage_a_checkpoint"])
        return writer
    return FullKVWriter(kind, scales, cfg)


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
