from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch

from common import fixed_samples, load_json, load_tokenizer, progress, save_json


def _opt_version(name):
    try:
        return getattr(importlib.import_module(name), "__version__", "unknown")
    except Exception:
        return "not-installed"


def _file_hash(path):
    try:
        block = Path(path).read_bytes()
        return hashlib.sha256(block[:1 << 20]).hexdigest()[:16]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    tok = load_tokenizer(cfg)
    samples = fixed_samples(cfg, tok)

    out_path = Path(cfg["work_dir"]) / "artifacts" / "rendered_samples.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # overwrite, never append: stale/duplicate rows must not survive re-runs
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metadata = {
        "seed": cfg["seed"],
        "model_path": cfg["model_path"],
        "hotpot_dev": cfg["hotpot_dev"],
        "hotpot_dev_sha256_head": _file_hash(cfg["hotpot_dev"]),
        "prompt_version": "SYSTEM+[1..10] context + Question boundary via offset mapping",
        "count": len(samples),
        "versions": {
            "transformers": _opt_version("transformers"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "fla": _opt_version("fla"),
            "causal_conv1d": _opt_version("causal_conv1d"),
        },
    }
    save_json(Path(cfg["work_dir"]) / "artifacts" / "render_metadata.json", metadata)
    progress(f"rendered {len(samples)} fixed samples (overwrite) -> {out_path} | {metadata['versions']}")


if __name__ == "__main__":
    main()
