from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from v2_common import Stores, load_json, progress, rows_for, save_json


class Stats:
    def __init__(self):
        self.count = self.nan = self.inf = 0
        self.total = self.square = self.absmax = 0.0
        self.sample = []

    def add(self, tensor):
        x = tensor.float().flatten()
        self.nan += torch.isnan(x).sum().item()
        self.inf += torch.isinf(x).sum().item()
        x = x[torch.isfinite(x)]
        self.count += x.numel()
        self.total += x.double().sum().item()
        self.square += x.double().square().sum().item()
        self.absmax = max(self.absmax, x.abs().max().item())
        stride = max(1, x.numel() // 2048)
        self.sample.append(x[::stride][:2048].cpu())

    def report(self, normalized_rms):
        mean = self.total / max(self.count, 1)
        rms = (self.square / max(self.count, 1)) ** 0.5
        std = max(self.square / max(self.count, 1) - mean * mean, 0) ** 0.5
        values = torch.cat(self.sample) if self.sample else torch.zeros(1)
        return {
            "mean": mean, "rms": rms, "std": std, "abs_max": self.absmax,
            "p50": values.abs().quantile(.50).item(),
            "p95": values.abs().quantile(.95).item(),
            "p99": values.abs().quantile(.99).item(),
            "nan_count": self.nan, "inf_count": self.inf,
            "normalized_rms": normalized_rms,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    rows = rows_for(cfg, args.mode)
    store = Stores(cfg, args.mode, rows)
    sums = {
        "source_k": torch.zeros(8, 1024, dtype=torch.float64),
        "source_v": torch.zeros(8, 1024, dtype=torch.float64),
        "target_k": torch.zeros(36, 1024, dtype=torch.float64),
        "target_v": torch.zeros(36, 1024, dtype=torch.float64),
    }
    counts = {"source": 0, "target": 0}
    stats = {key: Stats() for key in sums}
    for index, sample in enumerate(rows["train"], 1):
        for sender, prefix in (("q35", "source"), ("4b", "target")):
            key, value, mask = store.memory("train", sender, sample)
            valid = mask[0].bool()
            layers = 8 if sender == "q35" else 36
            key = key[:, valid].float().reshape(layers, -1, 1024)
            value = value[:, valid].float().reshape(layers, -1, 1024)
            sums[f"{prefix}_k"] += key.double().square().sum(1)
            sums[f"{prefix}_v"] += value.double().square().sum(1)
            counts[prefix] += key.shape[1]
            stats[f"{prefix}_k"].add(key)
            stats[f"{prefix}_v"].add(value)
        if index % 64 == 0 or index == len(rows["train"]):
            progress(f"{args.mode}: scale statistics {index}/{len(rows['train'])}")
    scales = {
        key: (value / counts[key.split("_")[0]] + cfg["scale_epsilon"]).sqrt()
        .clamp_min(cfg["scale_floor"]).float()
        for key, value in sums.items()
    }
    root = Path(cfg["work_dir"]) / "artifacts" / args.mode
    root.mkdir(parents=True, exist_ok=True)
    torch.save(scales, root / "scales.pt")
    report = {
        key: stats[key].report(
            ((sums[key] / counts[key.split("_")[0]]) / scales[key].double().square())
            .mean().sqrt().item()
        )
        for key in stats
    }
    report["train_only"] = True
    report["scale_shapes"] = {k: list(v.shape) for k, v in scales.items()}
    save_json(root / "scale_audit.json", report)
    progress(f"{args.mode}: fixed source/target scales saved")


if __name__ == "__main__":
    main()
