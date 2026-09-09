from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    stages = [
        ("selftest", "protocol.py", ["--action", "selftest"]),
        ("audit", "protocol.py", ["--action", "audit"]),
        ("assets", "build_assets.py", []),
        ("scales", "compute_scales.py", []),
        ("v0", "v0_diagnostic.py", []),
        ("a1_overfit16", "train_a1_overfit16.py", []),
        ("a1", "train_a1_anchor8.py", []),
        ("a2", "train_a2_full36.py", []),
        ("b", "train_b_functional.py", []),
        ("evaluate", "evaluate.py", []),
    ]
    markers = ROOT / "artifacts" / args.mode / "stage_markers"
    markers.mkdir(parents=True, exist_ok=True)
    for name, script, extra in stages:
        marker = markers / f"{name}.done"
        if marker.exists():
            log(f"{args.mode}: skip {name}")
            continue
        argv = [
            sys.executable, "-u", str(ROOT / script),
            "--config", str(CONFIG), "--mode", args.mode, *extra,
        ]
        log(f"{args.mode}: start {name}")
        subprocess.run(argv, cwd=ROOT, check=True)
        marker.write_text(datetime.now().isoformat(), encoding="utf-8")
        log(f"{args.mode}: completed {name}")
    log(f"{args.mode}: pipeline completed")


if __name__ == "__main__":
    main()
