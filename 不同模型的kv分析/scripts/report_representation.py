"""用 stage_a（alignment-only）checkpoint 计算 K/V 表示指标（NMSE + cosine）。

对每个方向，在 validation 集上：
  pred = Writer(stage_a best.pt, source KV 采样 128 位置)
  target = Receiver native KV（target 缓存，同位置）
  计算逐层 k_nmse / v_nmse / k_cosine / v_cosine + 平均。

输出：{out}.json（逐层 + 平均）与打印汇总。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protocol import (
    Store, cuda, load_json, load_model, progress, representation_loss, sampled_positions,
    save_json, seed_all,
)
from writer import load_writer


def main():
    parser = argparse.ArgumentParser(description="计算 K/V 表示指标")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--writer-checkpoint", required=True, help="stage_a best.pt")
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--sampled-tokens", type=int, default=128)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    seed_all(args.seed)
    cfg = {
        "work_dir": args.workdir,
        "num_layers": args.num_layers,
        "feature_dim": args.feature_dim,
        "source_dir": args.source_dir,
        "target_dir": args.target_dir,
        "sampled_tokens": args.sampled_tokens,
    }
    manifest = load_json(Path(args.workdir) / "artifacts" / "manifest.json")
    validation = manifest["validation"]
    store = Store(cfg, args.mode, {"test": [], "train": [], "validation": validation})

    writer, _ = load_writer(args.writer_checkpoint, map_location="cuda")
    writer = writer.to(cuda()).eval()

    layers = []
    for layer in range(args.num_layers):
        layers.append({"layer": layer, "k_nmse": [], "v_nmse": [], "k_cosine": [], "v_cosine": []})

    for index, sample in enumerate(validation, 1):
        source = store.source("validation", sample["id"])
        target = store.target("validation", sample["id"])
        positions = sampled_positions(target["pre_key"].shape[1], args.sampled_tokens)
        sk = source["pre_key"][:, positions].to(cuda())
        sv = source["value"][:, positions].to(cuda())
        tk = target["pre_key"][:, positions].to(cuda())
        tv = target["value"][:, positions].to(cuda())
        pk, pv = writer(sk, sv)
        _, metrics = representation_loss(pk, pv, tk, tv, 0.0)
        for entry in metrics:
            layer = entry["layer"]
            for key in ("k_nmse", "v_nmse", "k_cosine", "v_cosine"):
                layers[layer][key].append(entry[key])
        if index % 16 == 0:
            progress(f"{args.direction} {index}/{len(validation)}")

    summary = []
    for layer in layers:
        row = {"layer": layer["layer"]}
        for key in ("k_nmse", "v_nmse", "k_cosine", "v_cosine"):
            row[key] = float(sum(layer[key]) / len(layer[key]))
        summary.append(row)

    def mean(key):
        return float(sum(row[key] for row in summary) / len(summary))

    result = {
        "direction": args.direction,
        "writer_checkpoint": args.writer_checkpoint,
        "n_validation": len(validation),
        "sampled_tokens": args.sampled_tokens,
        "per_layer": summary,
        "average": {
            "k_nmse": mean("k_nmse"),
            "v_nmse": mean("v_nmse"),
            "k_cosine": mean("k_cosine"),
            "v_cosine": mean("v_cosine"),
        },
    }
    save_json(args.out, result)
    avg = result["average"]
    print(f"{args.direction}: k_nmse={avg['k_nmse']:.5f} v_nmse={avg['v_nmse']:.5f} "
          f"k_cos={avg['k_cosine']:.4f} v_cos={avg['v_cosine']:.4f}")


if __name__ == "__main__":
    main()
