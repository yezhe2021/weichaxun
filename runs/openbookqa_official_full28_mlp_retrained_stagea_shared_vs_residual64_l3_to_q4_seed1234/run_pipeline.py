import argparse
import fcntl
import os
import subprocess
import sys
import traceback

from common import configuration, load_model, run_root, save_json, seed_all

STAGES = ("asset_audit", "stage_a", "select_stage_a", "train_shared", "train_residual", "evaluate")


def execute(cfg, stage):
    if stage == "asset_audit":
        from data import audit_assets
        audit_assets(cfg)
    elif stage == "stage_a":
        from experiment import train_stage_a
        train_stage_a(cfg)
    else:
        from experiment import evaluate, select_stage_a, train_branch
        model = load_model(cfg, "qwen")
        try:
            if stage == "select_stage_a":
                select_stage_a(cfg, model)
            elif stage == "train_shared":
                train_branch(cfg, model, "shared")
            elif stage == "train_residual":
                train_branch(cfg, model, "residual")
            elif stage == "evaluate":
                evaluate(cfg, model)
            else:
                raise ValueError(stage)
        finally:
            import torch
            del model
            torch.cuda.empty_cache()


def done_path(cfg, stage):
    return run_root(cfg) / "stage_status" / f"{stage}.json"


def run_stage(cfg, stage):
    save_json(run_root(cfg) / "status.json", {"status": "running", "stage": stage, "pid": os.getpid()})
    try:
        execute(cfg, stage)
        save_json(done_path(cfg, stage), {"signature": cfg["signature"], "status": "completed", "stage": stage})
    except BaseException as error:
        save_json(run_root(cfg) / "status.json", {"status": "failed", "stage": stage,
                                                   "error": f"{type(error).__name__}: {error}", "pid": os.getpid()})
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    parser.add_argument("action", choices=("all", *STAGES), default="all", nargs="?")
    parser.add_argument("--force-stage", action="store_true")
    args = parser.parse_args()
    cfg = configuration(args.mode)
    seed_all(cfg["seed"])
    root = run_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "run_config.json", cfg)
    if args.action != "all":
        run_stage(cfg, args.action)
        return
    lock = (root / "pipeline.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Official retrained-Stage-A pipeline already running")
    (root / "pipeline.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    for stage in STAGES:
        done = done_path(cfg, stage)
        if done.exists() and not args.force_stage:
            payload = __import__("json").loads(done.read_text(encoding="utf-8"))
            if payload.get("signature") == cfg["signature"] and payload.get("status") == "completed":
                print(f"SKIP completed stage: {stage}", flush=True)
                continue
        print(f"START stage: {stage}", flush=True)
        subprocess.run([sys.executable, "-u", __file__, "--mode", args.mode, stage], check=True)
        print(f"DONE stage: {stage}", flush=True)
    save_json(root / "status.json", {"status": "completed", "stage": "all", "pid": os.getpid()})
    print("ALL OFFICIAL RETRAINED-STAGE-A EXPERIMENT STAGES COMPLETED", flush=True)


if __name__ == "__main__":
    main()
