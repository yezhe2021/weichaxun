import argparse
import datetime
import os
import subprocess
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get(
    "EXPERIMENT_PYTHON", "/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
)
CONFIG = ROOT / "config.json"
STAGES = (
    "selftest",
    "structure",
    "baseline",
    "protocol",
    "s0",
    "train",
    "evaluate",
)


def log(message):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def stage(mode, action):
    marker = ROOT / "artifacts" / mode / "stage_markers" / f"{action}.done"
    if marker.exists():
        log(f"SKIP completed stage {action}")
        return
    argv = [
        PYTHON,
        "-u",
        str(ROOT / "experiment.py"),
        "--config",
        str(CONFIG),
        "--mode",
        mode,
        action,
    ]
    log("RUN " + " ".join(argv))
    subprocess.run(argv, cwd=ROOT, check=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.datetime.now().isoformat(), encoding="utf-8")
    log(f"DONE {action}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    try:
        for action in STAGES:
            stage(args.mode, action)
        log(f"Pipeline completed: mode={args.mode}; 8B/Cross intentionally not run")
    except Exception as error:
        log(f"Pipeline failed: {type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
