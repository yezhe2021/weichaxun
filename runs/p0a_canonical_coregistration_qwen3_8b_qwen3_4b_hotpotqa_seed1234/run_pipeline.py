from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from p0a import (
    final_evaluation,
    load_json,
    progress,
    save_json,
    train_model,
)


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

    def mark(stage):
        status["completed_stages"].append(stage)
        status["last_completed_at"] = time.time()
        save_json(status_path, status)
        progress(f"Pipeline stage completed: {stage}")

    def command(stage, *extra):
        progress(f"Pipeline stage started: {stage}")
        subprocess.run(
            [sys.executable, str(work / "p0a.py"), "--config", config_path, stage, *map(str, extra)],
            cwd=work,
            check=True,
        )
        mark(stage + ("_smoke" if extra else ""))

    try:
        progress("P0-A pipeline started")
        command("prepare")
        command("cpu_self_test")
        command("sanity")

        smoke_limit = max(cfg["smoke_train_size"], cfg["smoke_validation_size"])
        command("extract_a", "--limit", smoke_limit)
        command("extract_b", "--limit", smoke_limit)
        train_model(cfg, "private", smoke=True)
        mark("private_smoke")
        train_model(cfg, "shared", smoke=True)
        mark("shared_smoke")
        mark("smoke_passed")

        command("extract_a")
        command("extract_b")
        train_model(cfg, "private", smoke=False)
        mark("private_formal")
        train_model(cfg, "shared", smoke=False)
        mark("shared_formal")
        final_evaluation(cfg)
        mark("final_evaluation")
        status["state"] = "completed"
        status["finished_at"] = time.time()
        save_json(status_path, status)
        progress("P0-A pipeline completed")
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
