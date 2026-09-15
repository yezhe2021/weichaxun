from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import traceback

from common import ROOT, configuration, digest, log, read_json, run_root, save_json, seed_all

STAGES = ("prepare", "cache_qwen", "cache_llama", "train_qwen", "train_llama", "compare")


def worker(cfg, stage):
    if stage == "prepare":
        from prepare import prepare
        prepare(cfg)
    elif stage.startswith("cache_"):
        from protocol import build
        build(cfg, stage[6:])
    elif stage.startswith("train_"):
        from sufficiency import train_family
        train_family(cfg, stage[6:])
    else:
        from sufficiency import compare
        compare(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=STAGES)
    args = parser.parse_args(); cfg = configuration("study"); seed_all(cfg["seed"])
    root = run_root(cfg); root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(cfg, args.worker); return
    code_hash = digest({path.name: path.read_text(encoding="utf-8") for path in sorted(ROOT.glob("*.py"))})
    lock = (root / "pipeline.lock").open("w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("training-sufficiency pipeline already running")
    (root / "pipeline.pid").write_text(str(os.getpid()) + "\n")
    save_json(root / "run_config.json", cfg)
    current = None
    try:
        for stage in STAGES:
            current = stage; marker = root / "completed" / f"{stage}.json"
            if marker.exists():
                saved = read_json(marker)
                if saved != {"signature": cfg["signature"], "code_hash": code_hash, "stage": stage}:
                    raise RuntimeError("existing marker belongs to different code/config")
                log(f"RESUME skip {stage}"); continue
            save_json(root / "status.json", {"status": "running", "stage": stage, "pid": os.getpid()})
            log(f"START {stage}")
            subprocess.run([sys.executable, "-u", str(ROOT / "run_pipeline.py"), "--worker", stage], cwd=ROOT, check=True)
            save_json(marker, {"signature": cfg["signature"], "code_hash": code_hash, "stage": stage})
            log(f"DONE {stage}")
        save_json(root / "status.json", {"status": "completed", "stage": "compare", "pid": os.getpid()})
        log("ALL EXPERIMENTS COMPLETED")
    except BaseException as error:
        save_json(root / "status.json", {"status": "failed", "stage": current, "error": str(error), "pid": os.getpid()})
        traceback.print_exc(); raise


if __name__ == "__main__": main()
