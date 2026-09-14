import argparse
import fcntl
import os
import subprocess
import sys
import traceback

from common import configuration, load_model, run_root, save_json, seed_all

STAGES = ("prepare", "cache_llama", "cache_qwen", "phase0_audit", "teachers_baselines",
          "stage_a", "select_stage_a", "stage_b", "evaluate")


def execute_stage(cfg, stage):
    if stage == "prepare":
        from data import prepare_manifests
        prepare_manifests(cfg)
    elif stage == "cache_llama":
        from data import cache_llama
        cache_llama(cfg)
    elif stage == "cache_qwen":
        from data import cache_qwen_and_pairs
        cache_qwen_and_pairs(cfg)
    elif stage == "phase0_audit":
        from data import phase0_audit
        phase0_audit(cfg)
    elif stage == "teachers_baselines":
        from experiment import evaluate_native_baselines, prepare_oracle_teachers
        model = load_model(cfg, "qwen")
        try:
            prepare_oracle_teachers(cfg, model)
            evaluate_native_baselines(cfg, model)
        finally:
            import torch
            del model; torch.cuda.empty_cache()
    elif stage == "stage_a":
        from experiment import train_stage_a
        train_stage_a(cfg)
    elif stage == "select_stage_a":
        from experiment import prepare_oracle_teachers, select_stage_a
        model = load_model(cfg, "qwen")
        try:
            prepare_oracle_teachers(cfg, model)
            select_stage_a(cfg, model)
        finally:
            import torch
            del model; torch.cuda.empty_cache()
    elif stage == "stage_b":
        from experiment import train_stage_b
        model = load_model(cfg, "qwen")
        try: train_stage_b(cfg, model)
        finally:
            import torch
            del model; torch.cuda.empty_cache()
    elif stage == "evaluate":
        from experiment import finalize
        finalize(cfg)
        from compare_linear import compare
        compare(cfg)
    else:
        raise ValueError(stage)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    parser.add_argument("action", choices=("all", "preflight", *STAGES), default="all", nargs="?")
    parser.add_argument("--force-stage", action="store_true")
    args = parser.parse_args()
    cfg = configuration(args.mode); seed_all(cfg["seed"])
    root = run_root(cfg); root.mkdir(parents=True, exist_ok=True)
    save_json(root / "run_config.json", cfg)
    if args.action == "preflight":
        for stage in STAGES[:4]:
            done = done_path(cfg, stage)
            if done.exists() and not args.force_stage:
                payload = __import__("json").loads(done.read_text(encoding="utf-8"))
                if payload.get("signature") == cfg["signature"] and payload.get("status") == "completed":
                    print(f"SKIP completed preflight stage: {stage}", flush=True); continue
            print(f"START preflight stage: {stage}", flush=True)
            subprocess.run([sys.executable, "-u", __file__, "--mode", args.mode, stage], check=True)
            print(f"DONE preflight stage: {stage}", flush=True)
        save_json(root / "status.json", {"status": "awaiting_phase0_review", "stage": "phase0_audit",
                                         "pid": os.getpid()})
        print("PREFLIGHT COMPLETED; REVIEW phase0_samples.jsonl BEFORE TRAINING", flush=True)
        return
    if args.action != "all":
        run_internal(cfg, args.action); return
    lock = (root / "pipeline.lock").open("w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("Standard-prompt pipeline already running")
    (root / "pipeline.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    for stage in STAGES:
        done = done_path(cfg, stage)
        if done.exists() and not args.force_stage:
            payload = __import__("json").loads(done.read_text(encoding="utf-8"))
            if payload.get("signature") == cfg["signature"] and payload.get("status") == "completed":
                print(f"SKIP completed stage: {stage}", flush=True); continue
        print(f"START stage: {stage}", flush=True)
        subprocess.run([sys.executable, "-u", __file__, "--mode", args.mode, stage], check=True)
        print(f"DONE stage: {stage}", flush=True)
    save_json(root / "status.json", {"status": "completed", "stage": "all", "pid": os.getpid()})
    print("ALL NATIVE-QUESTION OPTIONS-KV EXPERIMENT STAGES COMPLETED", flush=True)


if __name__ == "__main__": main()
