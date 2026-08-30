from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from common import sha256_ints


CAPTURE_CONVENTION = "Qwen3 k_norm(k_proj(hidden)) before RoPE; v_proj(hidden) native V; full Options prefix; FP16 CPU"


def cache_path(cfg: dict[str, Any], family: str, split: str, sample_id: str) -> Path:
    return Path(cfg["work_dir"]) / "cache" / family / split / f"{sample_id}.pt"


def save_cache(
    path: Path,
    sample: dict[str, Any],
    model_path: str,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": sample["id"],
        "model_path": model_path,
        "capture_convention": CAPTURE_CONVENTION,
        "prefix_token_ids": sample["prefix_token_ids"],
        "prefix_token_sha256": sha256_ints(sample["prefix_token_ids"]),
        "context_length": sample["context_length"],
        "pre_key": key.contiguous().cpu().half(),
        "value": value.contiguous().cpu().half(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_cache(path: Path, sample: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if sample is not None:
        expected = sha256_ints(sample["prefix_token_ids"])
        if payload["id"] != sample["id"] or payload["prefix_token_sha256"] != expected:
            raise RuntimeError(f"cache identity/token mismatch: {path}")
        if payload["context_length"] != sample["context_length"]:
            raise RuntimeError(f"cache context length mismatch: {path}")
    return payload
