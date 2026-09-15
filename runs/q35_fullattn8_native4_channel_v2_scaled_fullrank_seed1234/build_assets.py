from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from build_alignment import build
from v2_common import load_json, progress, rows_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    # Execute the already validated pre-RoPE hook into this experiment's new cache.
    subprocess.run([
        sys.executable, "-u", str(Path(cfg["previous_q35_dir"]) / "experiment.py"),
        "--config", args.config, "--mode", args.mode, "extract",
    ], check=True)
    build(cfg, args.mode, rows_for(cfg, args.mode))
    progress(f"{args.mode}: regenerated Qwen3.5 assets and detailed alignment audit")


if __name__ == "__main__":
    main()
