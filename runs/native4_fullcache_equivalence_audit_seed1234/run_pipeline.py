from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    completion = ROOT / "artifacts" / args.mode / "completion.json"
    if completion.exists():
        print(f"{args.mode}: audit already complete", flush=True)
        return
    subprocess.run([
        sys.executable, "-u", str(ROOT / "audit.py"),
        "--config", str(ROOT / "config.json"), "--mode", args.mode,
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
