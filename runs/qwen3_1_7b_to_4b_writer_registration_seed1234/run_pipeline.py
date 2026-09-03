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
        ("prepare", "prepare_assets.py", []), ("source1_7", "assets.py", ["--mode", args.mode, "source"]), ("scales", "assets.py", ["--mode", args.mode, "scales"]),
        ("depth_baseline", "experiment.py", ["--mode", args.mode, "depth_baseline"]),
        ("depth_overfit", "experiment.py", ["--mode", args.mode, "depth_overfit"]), ("depth_stagea", "experiment.py", ["--mode", args.mode, "depth_stagea"]),
        ("depth_f0", "experiment.py", ["--mode", args.mode, "depth_f0"]), ("depth_ce", "experiment.py", ["--mode", args.mode, "depth_ce"]),
        ("depth_kd", "experiment.py", ["--mode", args.mode, "depth_kd"]), ("depth_evaluate", "experiment.py", ["--mode", args.mode, "depth_evaluate"]),
        ("update0", "experiment.py", ["--mode", args.mode, "update0"]),
        ("overfit_linear", "experiment.py", ["--mode", args.mode, "overfit_linear"]), ("overfit_v2_h", "experiment.py", ["--mode", args.mode, "overfit_v2_h"]),
        ("overfit_v2_hl", "experiment.py", ["--mode", args.mode, "overfit_v2_hl"]), ("overfit_gate", "experiment.py", ["--mode", args.mode, "overfit_gate"]),
        ("stagea_linear", "experiment.py", ["--mode", args.mode, "stagea_linear"]), ("stagea_v2_h", "experiment.py", ["--mode", args.mode, "stagea_v2_h"]),
        ("stagea_v2_hl", "experiment.py", ["--mode", args.mode, "stagea_v2_hl"]), ("f0", "experiment.py", ["--mode", args.mode, "f0"]),
        ("ce_linear", "experiment.py", ["--mode", args.mode, "ce_linear"]), ("ce_v2_h", "experiment.py", ["--mode", args.mode, "ce_v2_h"]),
        ("ce_v2_hl", "experiment.py", ["--mode", args.mode, "ce_v2_hl"]), ("select", "experiment.py", ["--mode", args.mode, "select"]),
        ("kd", "experiment.py", ["--mode", args.mode, "kd"]), ("evaluate", "experiment.py", ["--mode", args.mode, "evaluate"]),
    ]
    for name, script, extra in stages:
        marker = markers / f"{name}.done"
        if marker.exists(): print(f"{args.mode}: skip {name}", flush=True); continue
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {args.mode}: start {name}", flush=True)
        run(script, cfg_path, *extra); marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    print(f"{args.mode}: pipeline completed", flush=True)


if __name__ == "__main__": main()
