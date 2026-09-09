from __future__ import annotations

import argparse
from pathlib import Path

from cache_store import CAPTURE_CONVENTION, cache_path, load_cache
from common import load_json, read_jsonl, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    expected = {
        cfg["sender_cache_family"]: (cfg["sender_model"], cfg["source_layers"]),
        cfg["receiver_cache_family"]: (cfg["receiver_model"], cfg["target_layers"]),
    }
    report = {"passed": True, "roles": {}, "splits": {}}
    for family, (model_path, layers) in expected.items():
        report["roles"][family] = model_path
    for split in ("train", "validation", "test"):
        samples = read_jsonl(Path(cfg["manifest_dir"]) / f"{split}.jsonl")
        missing = [
            sample["id"] for sample in samples
            for family in expected
            if not cache_path(cfg, family, split, sample["id"]).is_file()
        ]
        if missing:
            raise RuntimeError(f"missing reverse cache entries in {split}: {missing[:10]}")
        probes = []
        for sample in (samples[0], samples[-1]):
            for family, (model_path, layers) in expected.items():
                payload = load_cache(cache_path(cfg, family, split, sample["id"]), sample)
                shape = tuple(payload["pre_key"].shape)
                expected_shape = (
                    layers, sample["context_length"], cfg["num_kv_heads"], cfg["head_dim"]
                )
                if payload["model_path"] != model_path:
                    raise RuntimeError(
                        f"{family} model mismatch: {payload['model_path']} != {model_path}"
                    )
                if payload["capture_convention"] != CAPTURE_CONVENTION:
                    raise RuntimeError(f"{family} capture convention mismatch")
                if shape != expected_shape or tuple(payload["value"].shape) != expected_shape:
                    raise RuntimeError(f"{family} shape mismatch: {shape} != {expected_shape}")
                probes.append({"sample_id": sample["id"], "family": family, "shape": shape})
        report["splits"][split] = {"count": len(samples), "probes": probes}
    save_json(Path(cfg["work_dir"]) / "artifacts/cache_reuse_audit.json", report)


if __name__ == "__main__":
    main()
