"""阶段2：KV 资产缓存（方案 §六）。

对每个模型的 context-only prefix 提取：
  pre-RoPE K（k_norm 之后）+ Native V，形状 [L, T, 8, 128]，FP16 保存。
每样本单独保存 {"id", "input_ids", "context_length", "pre_key", "value"}。

同时为每个模型统计逐层、逐 feature 的 RMS scale s ∈ R^{L×1024}（方案 §二）。

用法（单 V100，Sender / Receiver 分开加载缓存）：
  python -u stage2_cache_assets.py --workdir <dir> --mode development \\
      --source-dir source_06 --target-dir target_17 --num-layers 28 \\
      --model <sender> action=source
  python -u stage2_cache_assets.py --workdir <dir> --mode development \\
      --source-dir source_06 --target-dir target_17 --num-layers 28 \\
      --model <receiver> action=target
  python -u stage2_cache_assets.py --workdir <dir> --mode development \\
      --source-dir source_06 --target-dir target_17 --num-layers 28 action=scales
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protocol import (
    Store, capture_native, load_json, load_model, progress, save_json, seed_all, scales,
)


def build_role(cfg, role, mode, rows, model_path, layers):
    model = load_model(model_path, cfg)
    directory_root = Path(cfg["work_dir"]) / "cache" / mode / cfg[f"{role}_dir"]
    try:
        for split, samples in rows.items():
            directory = directory_root / split
            directory.mkdir(parents=True, exist_ok=True)
            for index, sample in enumerate(samples, 1):
                destination = directory / f"{sample['id']}.pt"
                if destination.exists():
                    continue
                key, value = capture_native(model, sample["context_input_ids"], layers)
                temporary = destination.with_suffix(".tmp")
                torch.save({
                    "id": sample["id"],
                    "input_ids": sample["context_input_ids"],
                    "context_length": len(sample["context_input_ids"]),
                    "pre_key": key,
                    "value": value,
                }, temporary)
                temporary.replace(destination)
                progress(f"{role} {split} {index}/{len(samples)}")
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="阶段2：KV 资产缓存")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--source-dir", required=True, help="Sender context KV 的缓存目录名，如 source_06")
    parser.add_argument("--target-dir", required=True, help="Receiver context KV 的缓存目录名，如 target_17")
    parser.add_argument("--model", help="要缓存 KV 的模型（action=source/target 时需要）")
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--sampled-tokens", type=int, default=128)
    parser.add_argument("--receiver-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--direction", default="main", help="scale 按方向存放的目录名，如 self_06 / 06_to_17")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("action", choices=("source", "target", "scales"))
    args = parser.parse_args()

    seed_all(args.seed)
    cfg = {
        "work_dir": args.workdir,
        "receiver_dtype": args.receiver_dtype,
        "num_layers": args.num_layers,
        "feature_dim": args.feature_dim,
        "sampled_tokens": args.sampled_tokens,
        "source_dir": args.source_dir,
        "target_dir": args.target_dir,
    }
    manifest = load_json(Path(args.workdir) / "artifacts" / "manifest.json")
    smoke = {"smoke": 4, "development": 10**9}[args.mode]
    rows = {split: values[:smoke] for split, values in manifest.items()}

    if args.action in ("source", "target"):
        if not args.model:
            raise ValueError(f"action={args.action} requires --model")
        build_role(cfg, args.action, args.mode, rows, args.model, args.num_layers)
    else:
        scales(cfg, args.mode, rows, args.direction)
        progress(f"scales saved (direction={args.direction})")


if __name__ == "__main__":
    main()
