from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from data import load_json


ROOT = Path(__file__).resolve().parent


def run(argv):
    subprocess.run(argv, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("smoke", "development"), required=True); args = parser.parse_args()
    cfg = load_json(ROOT / "config.json"); work = Path(cfg["work_dir"])
    global_marker = work / "artifacts" / "reusable_asset_audit.done"
    if not global_marker.exists():
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] start reusable asset audit", flush=True)
        run([sys.executable, "-u", str(ROOT / "audit_reusable_assets.py"), "--config", str(ROOT / "config.json"), "--repair-missing"])
        global_marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    markers = work / "artifacts" / args.mode / "stage_markers"; markers.mkdir(parents=True, exist_ok=True)
    stages = [
        ("update0", "update0"),
        ("overfit_linear", "overfit_linear"), ("overfit_v2_h", "overfit_v2_h"), ("overfit_v2_hl", "overfit_v2_hl"), ("overfit_gate", "overfit_gate"),
        ("stagea_linear", "stagea_linear"), ("stagea_v2_h", "stagea_v2_h"), ("stagea_v2_hl", "stagea_v2_hl"),
        ("f0", "f0"),
        ("ce_linear", "ce_linear"), ("ce_v2_h", "ce_v2_h"), ("ce_v2_hl", "ce_v2_hl"),
        ("select", "select"), ("kd", "kd"), ("evaluate", "evaluate"),
    ]
    for name, action in stages:
        marker = markers / f"{name}.done"
        if marker.exists(): print(f"{args.mode}: skip {name}", flush=True); continue
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {args.mode}: start {name}", flush=True)
        run([sys.executable, "-u", str(ROOT / "experiment.py"), "--config", str(ROOT / "config.json"), "--mode", args.mode, action])
        marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    print(f"{args.mode}: pipeline completed", flush=True)


if __name__ == "__main__": main()
