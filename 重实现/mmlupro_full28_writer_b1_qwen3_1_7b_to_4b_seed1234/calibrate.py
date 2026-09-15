from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from cache_store import cache_path, load_cache
from common import load_json, progress, read_jsonl, save_json


def accumulate(rows, cfg: dict[str, Any], family: str, layers: int):
    shape = (layers, cfg["num_kv_heads"], cfg["head_dim"])
    sumsq_k = torch.zeros(shape, dtype=torch.float64)
    sumsq_v = torch.zeros(shape, dtype=torch.float64)
    token_count = 0
    for index, sample in enumerate(rows, 1):
        payload = load_cache(cache_path(cfg, family, "train", sample["id"]), sample)
        sumsq_k += payload["pre_key"].double().square().sum(dim=1)
        sumsq_v += payload["value"].double().square().sum(dim=1)
        token_count += int(payload["context_length"])
        progress(f"calibration {family}: {index}/{len(rows)}")
    epsilon = float(cfg["rms_epsilon"])
    return (
        (sumsq_k / token_count).sqrt().float().clamp_min(epsilon),
        (sumsq_v / token_count).sqrt().float().clamp_min(epsilon),
        token_count,
    )


def calibrate(cfg: dict[str, Any]) -> None:
    rows = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "manifests" / "train.jsonl")
    rows = rows[: min(len(rows), cfg["calibration_samples"])]
    source_k, source_v, source_tokens = accumulate(rows, cfg, "source17", cfg["source_layers"])
    target_k, target_v, target_tokens = accumulate(rows, cfg, "target4", cfg["target_layers"])
    if source_tokens != target_tokens:
        raise RuntimeError("source and target calibration token counts differ")
    destination = Path(cfg["work_dir"]) / "artifacts" / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({
        "source_k": source_k,
        "source_v": source_v,
        "target_k": target_k,
        "target_v": target_v,
        "train_sample_ids": [row["id"] for row in rows],
        "token_count": source_tokens,
        "train_only": True,
    }, destination / "rms_scales.pt")
    save_json(destination / "summary.json", {
        "train_only": True,
        "sample_count": len(rows),
        "token_count": source_tokens,
        "scale_shapes": {
            "source": list(source_k.shape),
            "target": list(target_k.shape),
        },
        "epsilon": cfg["rms_epsilon"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    calibrate(load_json(args.config))


if __name__ == "__main__":
    main()
