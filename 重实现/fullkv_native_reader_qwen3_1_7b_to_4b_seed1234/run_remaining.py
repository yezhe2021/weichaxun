from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
CONFIG = ROOT / "config.json"
MARKERS = ROOT / "artifacts" / "remaining_markers"


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def cuda_ready() -> bool:
    try:
        probe = subprocess.run(
            [PYTHON, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return probe.returncode == 0
    except Exception:
        return False


def wait_for_cuda() -> None:
    announced = False
    while not cuda_ready():
        if not announced:
            log("CUDA unavailable; waiting before the next stage")
            announced = True
        time.sleep(60)
    if announced:
        log("CUDA recovered")


def is_cuda_failure(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        "cuda is unavailable", "can't initialize nvml", "failed to initialize nvml",
        "cuda driver", "cuda error", "device_count=0", "no cuda gpus are available",
    )
    return any(pattern in lowered for pattern in patterns)


def run_stage(name: str, argv: list[str]) -> None:
    MARKERS.mkdir(parents=True, exist_ok=True)
    marker = MARKERS / f"{name}.done"
    if marker.exists():
        log(f"skip completed stage: {name}")
        return
    while True:
        wait_for_cuda()
        log(f"start stage: {name}")
        process = subprocess.Popen(
            [PYTHON, "-u", *argv], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        recent: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            recent.append(line)
            if len(recent) > 1000:
                recent.pop(0)
        returncode = process.wait()
        output_tail = "".join(recent)
        if returncode == 0:
            marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
            log(f"completed stage: {name}")
            return
        if is_cuda_failure(output_tail):
            log(f"CUDA failure in {name}; stage will retry from its resumable boundary")
            time.sleep(60)
            continue
        raise RuntimeError(f"non-CUDA failure in stage={name}, exit={returncode}")


def pipeline() -> None:
    config = str(CONFIG)
    run = str(ROOT / "run_pipeline.py")
    train = str(ROOT / "training.py")
    diagnose = str(ROOT / "diagnostics.py")

    stages: list[tuple[str, list[str]]] = [
        ("cache_train64", [run, "--config", config, "cache", "--split", "train", "--family", "both"]),
        ("cache_validation16", [run, "--config", config, "cache", "--split", "validation", "--family", "both"]),
        ("cache_test32", [run, "--config", config, "cache", "--split", "test", "--family", "both"]),
        ("calibrate_train64", [run, "--config", config, "calibrate"]),
        ("stage_a_d0", [train, "--config", config, "--writer", "d0", "--stage", "a"]),
        ("stage_a_d1", [train, "--config", config, "--writer", "d1", "--stage", "a"]),
        ("stage_a_d2", [train, "--config", config, "--writer", "d2", "--stage", "a"]),
        ("diagnose_d0", [diagnose, "--config", config, "--writer", "d0", "--checkpoint", "checkpoints/quick/d0/stage_a/best.pt"]),
        ("diagnose_d1", [diagnose, "--config", config, "--writer", "d1", "--checkpoint", "checkpoints/quick/d1/stage_a/best.pt"]),
        ("diagnose_d2", [diagnose, "--config", config, "--writer", "d2", "--checkpoint", "checkpoints/quick/d2/stage_a/best.pt"]),
        ("stage_b_d0", [train, "--config", config, "--writer", "d0", "--stage", "b"]),
        ("stage_b_d1", [train, "--config", config, "--writer", "d1", "--stage", "b"]),
        ("stage_b_d2", [train, "--config", config, "--writer", "d2", "--stage", "b"]),
        ("evaluate_d0", [run, "--config", config, "evaluate", "--writer", "d0"]),
        ("evaluate_d1", [run, "--config", config, "evaluate", "--writer", "d1"]),
        ("evaluate_d2", [run, "--config", config, "evaluate", "--writer", "d2"]),
    ]
    for name, argv in stages:
        run_stage(name, argv)
    completion = ROOT / "artifacts" / "remaining_completion.json"
    completion.write_text(json.dumps({
        "completed": True,
        "completed_at": datetime.now().isoformat(),
        "ordered_stages": [name for name, _ in stages],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log("remaining pipeline completed")


if __name__ == "__main__":
    pipeline()
