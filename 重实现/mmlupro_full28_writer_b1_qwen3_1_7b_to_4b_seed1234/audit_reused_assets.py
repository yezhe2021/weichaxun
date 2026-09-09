from __future__ import annotations

import argparse
from pathlib import Path

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
    stage_a_checkpoint = Path(cfg["stage_a_checkpoint"]).resolve()
    expected_stage_a = (
        Path(cfg["work_dir"]) / "checkpoints/quick/full28/stage_a/best.pt"
    ).resolve()
    references = cfg["local5_reference"]
    checks = {
        "manifest_counts": {key: len(value) for key, value in manifests.items()} == {"train": 1024, "validation": 128, "test": 128},
        "all_split_ids_disjoint": len({sample["id"] for rows in manifests.values() for sample in rows}) == 1280,
        "all_paired_caches_present": not missing_cache,
        "calibration_present": Path(cfg["calibration_path"]).exists(),
        "full28_stage_a_checkpoint_is_local_target": stage_a_checkpoint == expected_stage_a,
        "local5_stage_a_reference_present": Path(references["stage_a_summary"]).exists(),
        "local5_stage_b_reference_present": Path(references["stage_b_summary"]).exists(),
        "local5_formal_reference_present": Path(references["formal_summary"]).exists(),
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
