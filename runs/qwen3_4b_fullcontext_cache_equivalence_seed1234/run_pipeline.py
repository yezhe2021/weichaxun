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
    stages = [
        ("render", "render_dataset.py", []),
        ("audit", "audit.py", ["--mode", args.mode]),
    ]
    for name, script, extra in stages:
        marker = markers / f"{name}.done"
        if marker.exists():
            print(f"{args.mode}: skip {name}", flush=True)
            continue
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {args.mode}: start {name}", flush=True)
        subprocess.run([
            sys.executable, "-u", str(ROOT / script),
            "--config", str(ROOT / "config.json"), *extra,
        ], cwd=ROOT, check=True)
        marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    print(f"{args.mode}: pipeline completed", flush=True)


if __name__ == "__main__":
    main()
