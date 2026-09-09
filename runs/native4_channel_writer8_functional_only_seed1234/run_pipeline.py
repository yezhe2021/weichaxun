from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"
ACTIONS = (
    ("audit", None),
    ("teacher_cache", None),
    ("f0", None),
    ("train", "f1"),
    ("train", "f2"),
    ("evaluate", None),
)


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    markers = Path(cfg["work_dir"]) / "artifacts" / args.mode / "stage_markers"
    markers.mkdir(parents=True, exist_ok=True)
    for action, group in ACTIONS:
        name = action if group is None else group
        marker = markers / f"{name}.done"
        if marker.exists():
            log(f"{args.mode}: skip {name}")
            continue
        argv = [
            sys.executable,
            "-u",
            str(ROOT / "experiment.py"),
            "--config",
            str(CONFIG),
            "--mode",
            args.mode,
            action,
        ]
        if group:
            argv.extend(["--group", group])
        log(f"{args.mode}: start {name}")
        subprocess.run(argv, cwd=ROOT, check=True)
        marker.write_text(datetime.now().isoformat(), encoding="utf-8")
        log(f"{args.mode}: completed {name}")
    log(f"{args.mode}: pipeline completed")


if __name__ == "__main__":
    main()
