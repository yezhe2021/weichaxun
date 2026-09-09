from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common import load_json


def run_stage(cfg, name, script, args, markers):
    marker = markers / f"{name}.done"
    required = REQUIRED_ARTIFACTS.get(name)
    if marker.exists() and (required is None or Path(cfg["work_dir"], required).exists()):
        print(f"pipeline: skip {name}", flush=True)
        return
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] pipeline: start {name}", flush=True)
    subprocess.run([sys.executable, "-u", str(Path(cfg["work_dir"]) / script),
                    "--config", str(Path(cfg["work_dir"]) / "config.json"), *args],
                   cwd=cfg["work_dir"], check=True)
    marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")


REQUIRED_ARTIFACTS = {
    "render": "artifacts/rendered_samples.jsonl",
    "audit": "metrics/smoke1_audit.json",
    "equivalence_smoke": "outputs/smoke/equivalence_samples.jsonl",
    "ablations_smoke": "outputs/smoke/ablation_samples.jsonl",
    "evaluate_smoke": "metrics/smoke_metrics.json",
    "equivalence_formal": "outputs/formal/equivalence_samples.jsonl",
    "ablations_formal": "outputs/formal/ablation_samples.jsonl",
    "evaluate_formal": "metrics/formal_metrics.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    work = Path(cfg["work_dir"])
    # markers live with the outputs (work_dir), not the code checkout dir
    markers = work / "artifacts" / "stage_markers"
    markers.mkdir(parents=True, exist_ok=True)

    stages = [
        ("render", "render_dataset.py", []),
        ("audit", "audit_cache_structure.py", ["--sample-index", "0", "--split-diag"]),
        ("equivalence_smoke", "run_equivalence.py", ["--mode", "smoke"]),
        ("ablations_smoke", "run_ablations.py", ["--mode", "smoke"]),
        ("evaluate_smoke", "evaluate.py", ["--mode", "smoke"]),
    ]
    if args.mode == "formal":
        stages += [
            ("equivalence_formal", "run_equivalence.py", ["--mode", "formal"]),
            ("ablations_formal", "run_ablations.py", ["--mode", "formal"]),
            ("evaluate_formal", "evaluate.py", ["--mode", "formal"]),
        ]
    for name, script, extra in stages:
        run_stage(cfg, name, script, extra, markers)
    print(f"pipeline [{args.mode}]: completed", flush=True)


if __name__ == "__main__":
    main()
