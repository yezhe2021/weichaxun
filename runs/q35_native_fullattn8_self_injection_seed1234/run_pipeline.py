from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    markers = ROOT / "artifacts" / args.mode / "stage_markers"
    markers.mkdir(parents=True, exist_ok=True)
    for action in ("manifest", "extract", "train", "evaluate"):
        marker = markers / f"{action}.done"
        if marker.exists():
            print(f"{args.mode}: skip {action}", flush=True); continue
        print(f"[{datetime.now():%F %T}] {args.mode}: start {action}", flush=True)
        subprocess.run([
            sys.executable, "-u", str(ROOT / "experiment.py"),
            "--config", str(ROOT / "config.json"), "--mode", args.mode, action,
        ], cwd=ROOT, check=True)
        marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    print(f"{args.mode}: pipeline completed", flush=True)


if __name__ == "__main__": main()
