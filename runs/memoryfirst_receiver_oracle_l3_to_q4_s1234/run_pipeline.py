import argparse
import fcntl
import json
import os
import subprocess
import sys
import traceback

from common import configuration, run_root, save_json, seed_all

STAGES = ("oracle_audit", "prepare_teachers", "stage_b", "evaluate")


def execute_stage(cfg, stage):
    if stage == "oracle_audit":
        from experiment import oracle_audit
        oracle_audit(cfg)
    elif stage == "prepare_teachers":
        from experiment import prepare_teachers
        prepare_teachers(cfg)
    elif stage == "stage_b":
        from experiment import train_stage_b
        train_stage_b(cfg)
    elif stage == "evaluate":
        from experiment import finalize
        finalize(cfg)
    else: raise ValueError(stage)


def done_path(cfg, stage):
    return run_root(cfg) / "stage_status" / f"{stage}.json"


def run_internal(cfg, stage):
    save_json(run_root(cfg) / "status.json", {"status": "running", "stage": stage, "pid": os.getpid()})
    try:
        execute_stage(cfg, stage)
        save_json(done_path(cfg, stage), {"signature": cfg["signature"], "status": "completed", "stage": stage})
    except BaseException as error:
        save_json(run_root(cfg) / "status.json", {"status": "failed", "stage": stage,
                                                  "error": f"{type(error).__name__}: {error}", "pid": os.getpid()})
        traceback.print_exc(); raise


def completed(cfg, stage):
    path = done_path(cfg, stage)
    if not path.exists(): return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("signature") == cfg["signature"] and payload.get("status") == "completed"


def invoke(cfg, stage):
    if completed(cfg, stage):
        print(f"SKIP completed stage: {stage}", flush=True); return
    print(f"START stage: {stage}", flush=True)
    subprocess.run([sys.executable, "-u", __file__, "--mode", cfg["mode"], stage], check=True)
    print(f"DONE stage: {stage}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    parser.add_argument("action", choices=("all", *STAGES), default="all", nargs="?")
    args = parser.parse_args(); cfg = configuration(args.mode); seed_all(cfg["seed"])
    root = run_root(cfg); root.mkdir(parents=True, exist_ok=True); save_json(root / "run_config.json", cfg)
    if args.action != "all": run_internal(cfg, args.action); return
    lock = (root / "pipeline.lock").open("w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("Memory-first pipeline already running")
    (root / "pipeline.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    invoke(cfg, "oracle_audit")
    audit = json.loads((root / "oracle_audit" / "summary.json").read_text(encoding="utf-8"))
    should_train = args.mode == "smoke" or audit["gate"]["passed"]
    if should_train:
        invoke(cfg, "prepare_teachers"); invoke(cfg, "stage_b")
    else:
        print(f"ORACLE GATE FAILED: gain={audit['gain_correct']} correct; Stage-B intentionally skipped", flush=True)
    invoke(cfg, "evaluate")
    save_json(root / "status.json", {"status": "completed", "stage": "all",
                                      "stage_b_executed": should_train, "pid": os.getpid()})
    print("ALL MEMORY-FIRST RECEIVER STAGES COMPLETED", flush=True)


if __name__ == "__main__": main()
