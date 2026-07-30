from __future__ import annotations

import argparse
import json
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
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    actions = [("audit", None), ("statistics", None)]
    actions += [("stage_a", variant) for variant in cfg["variants"]]
    actions += [("stage_b", variant) for variant in cfg["variants"]]
    actions += [("evaluate", None)]
    markers = Path(cfg["work_dir"]) / "artifacts" / args.mode / "stage_markers"
    markers.mkdir(parents=True, exist_ok=True)
    for action, variant in actions:
        name = action if variant is None else f"{action}_{variant}"
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
        if variant:
            argv.extend(["--variant", variant])
        log(f"{args.mode}: start {name}")
        subprocess.run(argv, cwd=ROOT, check=True)
        marker.write_text(datetime.now().isoformat(), encoding="utf-8")
        log(f"{args.mode}: completed {name}")
    log(f"{args.mode}: pipeline completed")


if __name__ == "__main__":
    main()
