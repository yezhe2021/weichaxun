from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import Store, load_json, progress, rows_for, save_json, seed_all


def verify_source(cfg, mode, rows):
    manifest = load_json(Path(cfg["work_dir"]) / "artifacts" / "manifest.json")
    for split, samples in rows.items():
        required = {sample["id"] for sample in samples} | {sample["shuffle_id"] for sample in samples}
        known = {sample["id"] for sample in manifest[split]}
        if not required <= known:
            raise RuntimeError(f"{split}: samples absent from frozen manifest")
        for sample_id in required:
            path = Path(cfg["work_dir"]) / "cache" / "source1_7" / split / f"{sample_id}.pt"
            record = torch.load(path, map_location="cpu", weights_only=False)
            if record["pre_key"].shape[0] != cfg["source_layers"] or record["value"].shape[0] != cfg["source_layers"]:
                raise RuntimeError(f"{path}: expected exactly {cfg['source_layers']} native source layers")
    progress(f"{mode}: reused 1.7B native 28-layer cache verified")


def compute_scales(cfg, mode, rows):
    path = Path(cfg["work_dir"]) / "artifacts" / mode / "scales.pt"
    if path.exists():
        return
    store = Store(cfg, mode, rows)
    sums = {
        "source_k": torch.zeros(cfg["source_layers"], cfg["feature_dim"], dtype=torch.float64),
        "source_v": torch.zeros(cfg["source_layers"], cfg["feature_dim"], dtype=torch.float64),
        "target_k": torch.zeros(cfg["num_layers"], cfg["feature_dim"], dtype=torch.float64),
        "target_v": torch.zeros(cfg["num_layers"], cfg["feature_dim"], dtype=torch.float64),
    }
    count = 0
    for index, sample in enumerate(rows["train"], 1):
        source, target = store.source("train", sample["id"]), store.target("train", sample["id"])
        positions = target["positions"]
        values = {
            "source_k": source["pre_key"][:, positions].float().flatten(2),
            "source_v": source["value"][:, positions].float().flatten(2),
            "target_k": target["pre_key"].float().flatten(2),
            "target_v": target["value"].float().flatten(2),
        }
        for name, value in values.items():
            sums[name] += value.double().square().sum(1)
        count += len(positions)
        if index % 32 == 0 or index == len(rows["train"]):
            progress(f"{mode}: native-depth RMS {index}/{len(rows['train'])}")
    output = {name: (value / count + 1e-6).sqrt().float() for name, value in sums.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    save_json(path.with_suffix(".json"), {
        "count_tokens": count,
        "source_shape": [cfg["source_layers"], cfg["feature_dim"]],
        "target_shape": [cfg["num_layers"], cfg["feature_dim"]],
        "fixed_depth_interpolation_used": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("action", choices=("source", "scales"))
    args = parser.parse_args()
    cfg = load_json(args.config); seed_all(cfg["seed"])
    rows = rows_for(cfg, args.mode)
    if args.action == "source":
        verify_source(cfg, args.mode, rows)
    else:
        compute_scales(cfg, args.mode, rows)


if __name__ == "__main__":
    main()
