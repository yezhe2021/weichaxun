import fcntl
import os
import traceback

from common import configuration, run_root, save_json, seed_all
from locality import run


def main():
    cfg = configuration("study"); seed_all(cfg["seed"])
    root = run_root(cfg); root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("16-slot soft-locality study is already running")
    (root / "pipeline.pid").write_text(str(os.getpid()) + "\n")
    save_json(root / "run_config.json", cfg)
    save_json(root / "status.json", {"status": "running", "stage": "soft_locality_training", "pid": os.getpid()})
    try:
        run(cfg)
        save_json(root / "status.json", {"status": "completed", "stage": "soft_locality_training", "pid": os.getpid()})
    except BaseException as error:
        save_json(root / "status.json", {"status": "failed", "stage": "soft_locality_training", "error": str(error), "pid": os.getpid()})
        traceback.print_exc(); raise


if __name__ == "__main__": main()
