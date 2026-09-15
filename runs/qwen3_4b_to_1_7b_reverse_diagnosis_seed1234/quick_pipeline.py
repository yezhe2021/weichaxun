from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.quick.json"
PYTHON = sys.executable


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def run(action: str) -> None:
    log(f"quick validation: start {action}")
    subprocess.run(
        [PYTHON, "-u", str(ROOT / "experiment.py"), "--config", str(CONFIG), "--mode", "development", action],
        cwd=ROOT,
        check=True,
    )


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    artifacts = ROOT / "artifacts" / "development"
    r1_checkpoint = artifacts / "r1_continued_ce" / "selected.pt"
    r2_stage_a = artifacts / "r2_stage_a" / "best.pt"
    if not r1_checkpoint.exists() or not r2_stage_a.exists():
        raise FileNotFoundError(f"missing prerequisite: r1={r1_checkpoint.exists()} r2_stage_a={r2_stage_a.exists()}")

    # Reuse the safely saved 150-step R1 checkpoint. Only train the missing R2 branch.
    run("ce_r2")
    run("eval_continued")
    run("eval_r2")

    r1 = load(artifacts / "r1_continued_ce" / "summary.json")
    r2 = load(artifacts / "r2_learnable_ce" / "summary.json")
    baseline = load(artifacts / "receiver17_baseline" / "summary.json")
    conditions = {**r1, **r2}
    comparisons = {}
    for name in ("r1_continued_ce", "r2_learnable_ce"):
        correct = conditions[f"{name}_correct"]
        shuffled = conditions[f"{name}_shuffled"]
        comparisons[name] = {
            "correct_f1": correct["f1"],
            "shuffled_f1": shuffled["f1"],
            "correct_minus_shuffled": correct["f1"] - shuffled["f1"],
            "correct_em": correct["em"],
            "shuffled_em": shuffled["em"],
            "correct_nll": correct["nll"],
            "shuffled_nll": shuffled["nll"],
        }
    output = {
        "purpose": "rapid feasibility validation; not final metric optimization",
        "test_samples": 32,
        "r1_checkpoint": str(r1_checkpoint),
        "r2_ce_updates": 150,
        "conditions": conditions,
        "comparisons": comparisons,
        "receiver17_reference_128": baseline,
    }
    destination = artifacts / "quick_validation"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log("quick validation completed")


if __name__ == "__main__":
    main()
