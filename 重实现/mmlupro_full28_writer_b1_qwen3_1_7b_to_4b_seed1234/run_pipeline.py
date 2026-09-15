from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import load_json


ROOT = Path(__file__).resolve().parent


def invoke(script: str, config: str, *arguments: str):
    subprocess.run([sys.executable, "-u", str(ROOT / script), "--config", config, *arguments], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Explicit MMLU-Pro Full-KV trajectory experiment stage runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("action", choices=("prepare", "audit", "cache", "calibrate", "gradient_audit", "overfit", "train", "evaluate"))
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--family", choices=("source17", "target4", "both"), default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--writer", choices=("d0", "d1", "d2", "full28"), default="full28")
    parser.add_argument("--stage", choices=("a", "b"), default="a")
    parser.add_argument("--functional-mode", choices=("final", "all", "both"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--scope", choices=("overfit", "formal"), default="formal")
    args = parser.parse_args()
    cfg = load_json(args.config)
    if Path(cfg["work_dir"]).resolve() != ROOT.resolve():
        raise RuntimeError(f"config work_dir={cfg['work_dir']} does not match script directory={ROOT}")
    if args.action == "prepare":
        invoke("data_protocol.py", args.config)
    elif args.action == "audit":
        invoke("audit_protocol.py", args.config)
    elif args.action == "cache":
        if not args.split:
            raise ValueError("cache requires --split")
        extra = ["--split", args.split, "--family", args.family]
        if args.limit is not None:
            extra += ["--limit", str(args.limit)]
        invoke("build_cache.py", args.config, *extra)
    elif args.action == "calibrate":
        invoke("calibrate.py", args.config)
    elif args.action == "gradient_audit":
        extra = ["--writer", args.writer, "--stage", "gradient_audit"]
        if args.checkpoint:
            extra += ["--checkpoint", args.checkpoint]
        if args.functional_mode:
            extra += ["--functional-mode", args.functional_mode]
        invoke("training.py", args.config, *extra)
    elif args.action == "overfit":
        extra = ["--writer", args.writer, "--stage", args.stage, "--overfit"]
        if args.functional_mode:
            extra += ["--functional-mode", args.functional_mode]
        invoke("training.py", args.config, *extra)
    elif args.action == "train":
        extra = ["--writer", args.writer, "--stage", args.stage]
        if args.functional_mode:
            extra += ["--functional-mode", args.functional_mode]
        invoke("training.py", args.config, *extra)
    elif args.action == "evaluate":
        invoke("evaluate.py", args.config, "--scope", args.scope)


if __name__ == "__main__":
    main()
