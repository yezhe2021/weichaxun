import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    args = parser.parse_args()
    subprocess.run([sys.executable, "-u", str(ROOT / "evaluate.py"), "--mode", args.mode], cwd=ROOT, check=True)
