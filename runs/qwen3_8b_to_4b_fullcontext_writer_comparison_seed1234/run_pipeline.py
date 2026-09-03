from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("smoke","development"),required=True); args=parser.parse_args()
    markers=ROOT/"artifacts"/args.mode/"stage_markers"; markers.mkdir(parents=True,exist_ok=True)
    stages=[
        ("manifest","assets.py",["manifest"]),("source_cache","assets.py",["source"]),("target_teacher_cache","assets.py",["target"]),("rms","assets.py",["scales"]),
        ("baseline","experiment.py",["baseline"]),("overfit_linear","experiment.py",["overfit_linear"]),("overfit_full","experiment.py",["overfit_full"]),
        ("stagea_linear","experiment.py",["stagea_linear"]),("stagea_full","experiment.py",["stagea_full"]),("f0","experiment.py",["f0"]),
        ("linear_f1","experiment.py",["linear_f1"]),("full_f1","experiment.py",["full_f1"]),("linear_ce","experiment.py",["linear_ce"]),("full_ce","experiment.py",["full_ce"]),
        ("linear_kd","experiment.py",["linear_kd"]),("full_kd","experiment.py",["full_kd"]),("evaluate","experiment.py",["evaluate"]),
    ]
    for name,script,extra in stages:
        marker=markers/f"{name}.done"
        if marker.exists(): print(f"{args.mode}: skip {name}",flush=True); continue
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {args.mode}: start {name}",flush=True)
        subprocess.run([sys.executable,"-u",str(ROOT/script),"--config",str(ROOT/"config.json"),"--mode",args.mode,*extra],cwd=ROOT,check=True)
        marker.write_text(datetime.now().isoformat()+"\n",encoding="utf-8")
    print(f"{args.mode}: pipeline completed",flush=True)


if __name__=="__main__": main()
