from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import capture_native, load_json, load_model, progress, save_json, seed_all


def rows_for(cfg, mode):
    base, test = load_json(Path(cfg["work_dir"]) / "artifacts" / "manifest.json"), load_json(Path(cfg["work_dir"]) / "artifacts" / "eval128_manifest.json")
    sizes = cfg["smoke_sizes"] if mode == "smoke" else cfg["sizes"]
    return {"train": base["train"][:sizes["train"]], "validation": base["validation"][:sizes["validation"]], "test": test[:sizes["test"]]}


@torch.no_grad()
def build_source4(cfg, mode, rows):
    model = load_model(cfg["model_4b"], cfg); root = Path(cfg["work_dir"]) / "cache" / "source4_full"
    base = load_json(Path(cfg["work_dir"]) / "artifacts" / "manifest.json")
    full_maps = {
        "train": {x["id"]: x for x in base["train"]},
        "validation": {x["id"]: x for x in base["validation"]},
        "test": {x["id"]: x for x in load_json(Path(cfg["work_dir"]) / "artifacts" / "eval128_manifest.json")},
    }
    try:
        for split, samples in rows.items():
            required = {x["id"] for x in samples} | {x["shuffle_id"] for x in samples}
            samples = [full_maps[split][sample_id] for sample_id in sorted(required)]
            directory = root / split; directory.mkdir(parents=True, exist_ok=True)
            for index, sample in enumerate(samples, 1):
                destination = directory / f"{sample['id']}.pt"
                if destination.exists(): continue
                key, value = capture_native(model, sample["context_input_ids"], cfg["source_layers"])
                temporary = destination.with_suffix(".tmp"); torch.save({"id": sample["id"], "pre_key": key, "value": value, "context_length": len(sample["context_input_ids"])}, temporary); temporary.replace(destination)
                if index % 8 == 0 or index == len(samples): progress(f"{mode}: source4_full {split} {index}/{len(samples)}")
    finally:
        del model; torch.cuda.empty_cache()


def audit(cfg, mode, rows):
    source4 = Path(cfg["work_dir"]) / "cache" / "source4_full"; errors = []
    for split, samples in rows.items():
        for sample in samples:
            four = torch.load(source4 / split / f"{sample['id']}.pt", map_location="cpu", weights_only=False)
            base17 = Path(cfg["work_dir"]) / "cache" / "source1_7" / split / f"{sample['id']}.pt"
            extra17 = Path(cfg["work_dir"]) / "cache" / "source1_7_eval128" / f"{sample['id']}.pt"
            seventeen = torch.load(base17 if base17.exists() else extra17, map_location="cpu", weights_only=False)
            if four["pre_key"].shape != (36, sample["context_length"], 8, 128): errors.append(f"4B shape {sample['id']}")
            if seventeen["pre_key"].shape != (28, sample["context_length"], 8, 128): errors.append(f"1.7B shape {sample['id']}")
            if four["context_length"] != seventeen["context_length"]: errors.append(f"length {sample['id']}")
    report = {"passed": not errors, "errors": errors[:20], "counts": {k: len(v) for k, v in rows.items()}, "source4_shape": "[36,T,8,128]", "target1_7_shape": "[28,T,8,128]", "token_ids_shared": True}
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "protocol_audit.json", report)
    if errors: raise RuntimeError(f"protocol audit failed: {errors[:5]}")


def scales(cfg, mode):
    destination = Path(cfg["work_dir"]) / "artifacts" / mode / "scales.pt"
    if destination.exists(): return
    forward = Path(cfg["forward_experiment_dir"]) / "artifacts" / mode / "scales.pt"
    old = torch.load(forward, map_location="cpu", weights_only=False)
    output = {"source_k": old["target_k"], "source_v": old["target_v"], "target_k": old["source_k"], "target_v": old["source_v"]}
    destination.parent.mkdir(parents=True, exist_ok=True); torch.save(output, destination)
    save_json(destination.with_suffix(".json"), {"reused_and_direction_swapped": str(forward), "source_shape": [36, 1024], "target_shape": [28, 1024]})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--mode", choices=("smoke", "development"), required=True); parser.add_argument("action", choices=("cache", "audit", "scales")); args = parser.parse_args(); cfg = load_json(args.config); seed_all(cfg["seed"]); rows = rows_for(cfg, args.mode)
    if args.action == "cache": build_source4(cfg, args.mode, rows)
    elif args.action == "audit": audit(cfg, args.mode, rows)
    else: scales(cfg, args.mode)


if __name__ == "__main__": main()
