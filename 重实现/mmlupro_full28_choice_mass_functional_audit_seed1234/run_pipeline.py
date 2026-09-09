from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, "-u", str(ROOT / "choice_mass_audit.py"), "--config", args.config]
    if args.smoke:
        command.extend(["--splits", "validation", "--limit", "2", "--output-subdir", "smoke"])
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
