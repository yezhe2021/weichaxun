import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    subprocess.run(["nvidia-smi"], check=True)
    for stage in ("teachers", "train", "evaluate"):
        print(f"START {stage}", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "experiment.py"), stage], cwd=ROOT, check=True)
        print(f"DONE {stage}", flush=True)
    print("ALL B0 MIXED-HUB RECEIVER STAGE-B EXPERIMENTS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
