import argparse
import fcntl
import os
import traceback

from common import configuration, run_root, save_json, seed_all
from evaluate import run_oracle


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    args = parser.parse_args(); cfg = configuration(args.mode); seed_all(cfg["seed"])
    root = run_root(cfg); root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("Synchronized Oracle pipeline already running")
    (root / "pipeline.pid").write_text(str(os.getpid()) + "\n")
    save_json(root / "run_config.json", cfg)
    save_json(root / "status.json", {"status": "running", "stage": "oracle", "pid": os.getpid()})
    try:
        run_oracle(cfg)
        save_json(root / "status.json", {"status": "completed", "stage": "oracle", "pid": os.getpid()})
    except BaseException as error:
        save_json(root / "status.json", {"status": "failed", "stage": "oracle", "error": str(error), "pid": os.getpid()})
        traceback.print_exc(); raise


if __name__ == "__main__": main()
