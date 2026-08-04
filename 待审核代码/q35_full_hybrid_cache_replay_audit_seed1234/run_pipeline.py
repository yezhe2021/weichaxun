from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_stage(name, script, args, markers):
    marker = markers / f"{name}.done"
    if marker.exists():
        print(f"pipeline: skip {name}", flush=True)
        return
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] pipeline: start {name}", flush=True)
    subprocess.run([sys.executable, "-u", str(ROOT / script), "--config", str(ROOT / "config.json"), *args],
                   cwd=ROOT, check=True)
    marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    markers = ROOT / "artifacts" / "stage_markers"
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
        run_stage(name, script, extra, markers)
    print(f"pipeline [{args.mode}]: completed", flush=True)


if __name__ == "__main__":
    main()
