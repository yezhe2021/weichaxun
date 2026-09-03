from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from data import load_json


ROOT = Path(__file__).resolve().parent


def run(script, cfg, *extra):
    subprocess.run([sys.executable, "-u", str(ROOT / script), "--config", str(cfg), *extra], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("smoke", "development"), required=True); args = parser.parse_args()
    cfg_path = ROOT / "config.json"; cfg = load_json(cfg_path); work = Path(cfg["work_dir"])
    markers = work / "artifacts" / args.mode / "stage_markers"; markers.mkdir(parents=True, exist_ok=True)
    stages = [
        ("prepare", "prepare_assets.py", []),
        ("source", "assets.py", ["--mode", args.mode, "source"]),
        ("scales", "assets.py", ["--mode", args.mode, "scales"]),
        ("receiver_selftest", "experiment.py", ["--mode", args.mode, "receiver_selftest"]),
        ("e1_overfit_skip", "experiment.py", ["--mode", args.mode, "overfit_skip"]),
        ("e1_overfit_repeat", "experiment.py", ["--mode", args.mode, "overfit_repeat"]),
        ("e1_stagea_skip", "experiment.py", ["--mode", args.mode, "stagea_skip"]),
        ("e1_stagea_repeat", "experiment.py", ["--mode", args.mode, "stagea_repeat"]),
        ("e1_f0", "experiment.py", ["--mode", args.mode, "f0_e1"]),
        ("e1_ce_skip", "experiment.py", ["--mode", args.mode, "ce_skip"]),
        ("e1_ce_repeat", "experiment.py", ["--mode", args.mode, "ce_repeat"]),
        ("e2_update0", "experiment.py", ["--mode", args.mode, "update0"]),
        ("e2_stagea_fixed", "experiment.py", ["--mode", args.mode, "stagea_fixed"]),
        ("e2_stagea_learnable", "experiment.py", ["--mode", args.mode, "stagea_learnable"]),
        ("e2_f0", "experiment.py", ["--mode", args.mode, "f0_e2"]),
        ("e2_ce_fixed", "experiment.py", ["--mode", args.mode, "ce_fixed"]),
        ("e2_ce_learnable", "experiment.py", ["--mode", args.mode, "ce_learnable"]),
        ("evaluate", "experiment.py", ["--mode", args.mode, "evaluate"]),
    ]
    for name, script, extra in stages:
        marker = markers / f"{name}.done"
        if marker.exists():
            print(f"{args.mode}: skip {name}", flush=True); continue
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {args.mode}: start {name}", flush=True)
        run(script, cfg_path, *extra)
        marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    print(f"{args.mode}: pipeline completed", flush=True)


if __name__ == "__main__":
    main()
