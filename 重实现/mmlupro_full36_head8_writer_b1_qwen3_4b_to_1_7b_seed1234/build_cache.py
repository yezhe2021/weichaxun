from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from cache_store import cache_path, save_cache
from common import cuda, load_json, load_model, progress, read_jsonl, seed_all
from kv_protocol import capture_full_native, validate_native_shapes


def build_family(cfg: dict[str, Any], split: str, family: str, limit: int | None = None) -> None:
    if family == "source17":
        model_path, layers = cfg["model_1_7b"], cfg["target_layers"]
    elif family == "target4":
        model_path, layers = cfg["model_4b"], cfg["target_layers"]
    else:
        raise ValueError(family)
    samples = read_jsonl(Path(cfg["manifest_dir"]) / f"{split}.jsonl")
    if limit is not None:
        samples = samples[:limit]
    model = load_model(model_path, cfg, frozen=True)
    try:
        for index, sample in enumerate(samples, 1):
            destination = cache_path(cfg, family, split, sample["id"])
            if destination.exists():
                progress(f"{family}/{split}: resume skip {index}/{len(samples)}")
                continue
            key, value, _, _ = capture_full_native(
                model, sample["prefix_token_ids"], layers, cuda()
            )
            validate_native_shapes(
                key, value, layers, sample["context_length"],
                cfg["num_kv_heads"], cfg["head_dim"],
            )
            save_cache(destination, sample, model_path, key, value)
            progress(f"{family}/{split}: {index}/{len(samples)} T={sample['context_length']}")
    finally:
        del model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--family", choices=("source17", "target4", "both"), default="both")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    families = ("source17", "target4") if args.family == "both" else (args.family,)
    for family in families:
        build_family(cfg, args.split, family, args.limit)


if __name__ == "__main__":
    main()
