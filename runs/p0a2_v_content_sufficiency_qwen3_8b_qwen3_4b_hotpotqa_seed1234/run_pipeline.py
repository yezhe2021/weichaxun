from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from p0a2 import load_json, progress, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = str(Path(args.config).resolve())
    cfg = load_json(config_path)
    work = Path(cfg["work_dir"])
    status_path = work / "pipeline_status.json"
    status = {"state": "running", "started_at": time.time(), "completed_stages": []}
    save_json(status_path, status)

    def run(stage, *extra):
        progress(f"Pipeline stage started: {stage} {' '.join(map(str, extra))}")
        subprocess.run(
            [sys.executable, str(work / "p0a2.py"), "--config", config_path, stage, *map(str, extra)],
            cwd=work,
            check=True,
        )

    def mark(stage):
        status["completed_stages"].append(stage)
        status["last_completed_at"] = time.time()
        save_json(status_path, status)
        progress(f"Pipeline stage completed: {stage}")

    try:
        progress("P0-A2 pipeline started")
        run("cpu_self_test")
        mark("cpu_self_test")

        smoke_limit = max(cfg["smoke_train_size"], cfg["smoke_validation_size"])
        run("text_embeddings", "--limit", smoke_limit)
        mark("smoke_text_embeddings")
        for mode in cfg["modes"]:
            run("readouts", "--mode", mode, "--limit", smoke_limit)
        mark("smoke_readouts")
        for mode in cfg["modes"]:
            for sender in ("a", "b"):
                run("train_probe", "--mode", mode, "--sender", sender, "--smoke")
                run("evaluate_probe", "--mode", mode, "--sender", sender, "--smoke")
        mark("smoke_passed")

        run("text_embeddings")
        mark("formal_text_embeddings")
        for mode in cfg["modes"]:
            run("readouts", "--mode", mode)
        mark("formal_readouts")
        for mode in cfg["modes"]:
            for sender in ("a", "b"):
                run("train_probe", "--mode", mode, "--sender", sender)
                run("evaluate_probe", "--mode", mode, "--sender", sender)
        mark("formal_probe_evaluation")

        formal = {}
        for mode in cfg["modes"]:
            formal[mode] = {}
            for sender in ("a", "b"):
                path = work / "metrics" / f"{mode}_{sender}_formal_evaluation.json"
                formal[mode][sender] = load_json(path)
        save_json(work / "metrics" / "final_evaluation.json", formal)
        mark("final_evaluation")
        status["state"] = "completed"
        status["finished_at"] = time.time()
        save_json(status_path, status)
        progress("P0-A2 pipeline completed")
    except Exception as exc:
        status["state"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        status["failed_at"] = time.time()
        save_json(status_path, status)
        progress(f"Pipeline failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
