from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import load_json


ROOT = Path(__file__).resolve().parent


def invoke(script: str, config: str, *arguments: str):
    subprocess.run(
        [sys.executable, "-u", str(ROOT / script), "--config", config, *arguments],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Explicit stage runner; no stage starts implicitly")
    parser.add_argument("--config", required=True)
    parser.add_argument("action", choices=("prepare", "audit", "cache", "calibrate", "overfit", "train", "diagnose", "evaluate"))
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--family", choices=("source17", "target4", "both"), default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--writer", choices=("d0", "d1", "d2"), default="d2")
    parser.add_argument("--checkpoint")
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
    elif args.action == "overfit":
        invoke("training.py", args.config, "--writer", args.writer, "--stage", "both", "--overfit")
    elif args.action == "train":
        invoke("training.py", args.config, "--writer", args.writer, "--stage", "both")
    elif args.action == "diagnose":
        if not args.checkpoint:
            raise ValueError("diagnose requires --checkpoint")
        invoke("diagnostics.py", args.config, "--writer", args.writer, "--checkpoint", args.checkpoint)
    elif args.action == "evaluate":
        extra = ["--writer", args.writer]
        if args.checkpoint:
            extra += ["--checkpoint", args.checkpoint]
        invoke("evaluate.py", args.config, *extra)


if __name__ == "__main__":
    main()
