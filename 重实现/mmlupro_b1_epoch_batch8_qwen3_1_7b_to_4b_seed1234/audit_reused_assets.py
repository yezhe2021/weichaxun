from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cache_store import cache_path
from common import load_json, read_jsonl, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    manifest_root = Path(cfg["manifest_dir"])
    manifests = {
        split: read_jsonl(manifest_root / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    missing_cache = []
    for split, samples in manifests.items():
        for sample in samples:
            for family in ("source17", "target4"):
                path = cache_path(cfg, family, split, sample["id"])
                if not path.exists():
                    missing_cache.append(str(path))
    checkpoint = torch.load(cfg["stage_a_checkpoint"], map_location="cpu", weights_only=False)
    checks = {
        "manifest_counts": {key: len(value) for key, value in manifests.items()} == {"train": 1024, "validation": 128, "test": 128},
        "all_split_ids_disjoint": len({sample["id"] for rows in manifests.values() for sample in rows}) == 1280,
        "all_paired_caches_present": not missing_cache,
        "calibration_present": Path(cfg["calibration_path"]).exists(),
        "stage_a_checkpoint_present": Path(cfg["stage_a_checkpoint"]).exists(),
        "stage_a_writer_is_d2": checkpoint["writer_metadata"]["kind"] == "d2",
        "stage_b_mode_is_final_only": cfg["stage_b"]["modes"] == ["final"],
        "effective_batch_is_8": cfg["stage_b"]["effective_batch_size"] == 8,
    }
    save_json(Path(cfg["work_dir"]) / "artifacts/reused_assets/audit.json", {
        "passed": all(checks.values()), "checks": checks,
        "missing_cache_count": len(missing_cache), "missing_cache_examples": missing_cache[:10],
    })
    if not all(checks.values()):
        raise RuntimeError(f"reused asset audit failed: {checks}")
    print(f"reused asset audit passed: {checks}", flush=True)


if __name__ == "__main__":
    main()
