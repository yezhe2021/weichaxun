from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment import (
    final_evaluation,
    generation_sanity,
    load_json,
    phase1_run,
    phase2,
    progress,
    save_json,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
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

    def stage(command):
        progress(f"Pipeline stage started: {command}")
        subprocess.run(
            [sys.executable, str(work / "experiment.py"), "--config", config_path, command],
            cwd=work,
            check=True,
        )
        mark(command)

    def complete_kv_index(role):
        index_path = work / "kv" / role / "index.json"
        if not index_path.exists():
            return False
        try:
            rows = json.loads(index_path.read_text(encoding="utf-8"))
            return bool(rows) and all(
                (work / "kv" / role / row["relative_path"]).is_file() for row in rows
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return False

    try:
        progress("Pipeline started")
        stage("split")
        gate0_path = work / "metrics" / "gate0.json"
        gate0_ok = False
        if gate0_path.exists():
            try:
                gate0_ok = bool(json.loads(gate0_path.read_text(encoding="utf-8")).get("passed"))
            except (OSError, json.JSONDecodeError):
                gate0_ok = False
        if gate0_ok:
            mark("gate0_reused")
        else:
            stage("gate0")
        if complete_kv_index("sender"):
            mark("extract_sender_reused")
        else:
            stage("extract_sender")
        if complete_kv_index("receiver"):
            mark("extract_receiver_reused")
        else:
            stage("extract_receiver")

        formal_summary_path = work / "phase1" / "formal" / "summary.json"
        formal = None
        if formal_summary_path.exists():
            try:
                candidate = json.loads(formal_summary_path.read_text(encoding="utf-8"))
                if Path(candidate["best_checkpoint"]).is_file():
                    formal = candidate
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                formal = None
        if formal is not None:
            mark("phase1_formal_reused")
        else:
            smoke = []
            for lr in cfg["phase1_smoke_learning_rates"]:
                tag = "smoke_lr_" + str(lr).replace(".", "p")
                smoke.append(phase1_run(cfg, lr, tag, smoke=True))
            stable = [x for x in smoke if x["validation_relative_improvement"] > 0]
            if not stable:
                raise RuntimeError("Both Phase-I smoke learning rates failed to improve validation reconstruction")
            selected = min(stable, key=lambda x: x["best_validation_loss"])
            save_json(work / "phase1" / "smoke_selection.json", {"candidates": smoke, "selected": selected})
            mark("phase1_smoke")
            formal = phase1_run(cfg, selected["learning_rate"], "formal", smoke=False)
            mark("phase1_formal")
        if formal["validation_relative_improvement"] < cfg["phase1_min_validation_improvement"]:
            status["state"] = "stopped_at_gate1"
            status["gate1"] = formal
            save_json(status_path, status)
            return
        phase1_checkpoint = formal["best_checkpoint"]
        mark("gate1_passed")

        generation_sanity(cfg, phase1_checkpoint)
        mark("phase1_generation_sanity")
        stage("traces")
        phase2(cfg, phase1_checkpoint)
        mark("phase2")
        final_evaluation(cfg, phase1_checkpoint, str(work / "phase2" / "best.pt"))
        mark("final_evaluation")
        status["state"] = "completed"
        status["finished_at"] = time.time()
        save_json(status_path, status)
        progress("Pipeline completed")
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
