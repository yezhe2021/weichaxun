from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from p0b import load_json, progress, save_json


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
            [sys.executable, str(work / "p0b.py"), "--config", config_path, stage, *map(str, extra)],
            cwd=work,
            check=True,
        )

    def mark(stage):
        status["completed_stages"].append(stage)
        status["last_completed_at"] = time.time()
        save_json(status_path, status)
        progress(f"Pipeline stage completed: {stage}")

    try:
        progress("P0-B pipeline started")
        run("cpu_self_test")
        mark("cpu_self_test")
        limit = cfg["smoke_validation_size"]
        run("extract_a", "--limit", limit)
        run("extract_b", "--limit", limit)
        mark("smoke_selector_extraction")
        run("evaluate", "--limit", limit)
        mark("smoke_passed")
        run("extract_a")
        run("extract_b")
        mark("formal_selector_extraction")
        run("evaluate")
        mark("final_evaluation")
        status["state"] = "completed"
        status["finished_at"] = time.time()
        save_json(status_path, status)
        progress("P0-B pipeline completed")
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
