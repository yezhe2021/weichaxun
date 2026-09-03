from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from data import load_json


ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("smoke","development"),required=True); args=parser.parse_args(); cfg=ROOT/"config.json"; work=Path(load_json(cfg)["work_dir"]); markers=work/"artifacts"/args.mode/"stage_markers"; markers.mkdir(parents=True,exist_ok=True)
    stages=[
        ("prepare","prepare_assets.py",[]),("cache4","assets.py",["--mode",args.mode,"cache"]),("asset_audit","assets.py",["--mode",args.mode,"audit"]),("scales","assets.py",["--mode",args.mode,"scales"]),
        ("oracle","experiment.py",["--mode",args.mode,"oracle"]),("baseline17","experiment.py",["--mode",args.mode,"baseline17"]),
        ("r1_overfit","experiment.py",["--mode",args.mode,"overfit"]),("r1_stagea","experiment.py",["--mode",args.mode,"stagea_r1"]),("r1_f0","experiment.py",["--mode",args.mode,"f0_r1"]),("r1_ce","experiment.py",["--mode",args.mode,"ce_r1"]),("r1_eval","experiment.py",["--mode",args.mode,"eval_r1"]),
        ("update0","experiment.py",["--mode",args.mode,"update0"]),("continued_stagea","experiment.py",["--mode",args.mode,"stagea_continued"]),("r2_stagea","experiment.py",["--mode",args.mode,"stagea_r2"]),
        ("continued_f0","experiment.py",["--mode",args.mode,"f0_continued"]),("r2_f0","experiment.py",["--mode",args.mode,"f0_r2"]),("continued_ce","experiment.py",["--mode",args.mode,"ce_continued"]),("r2_ce","experiment.py",["--mode",args.mode,"ce_r2"]),("continued_eval","experiment.py",["--mode",args.mode,"eval_continued"]),("r2_eval","experiment.py",["--mode",args.mode,"eval_r2"]),("finalize","experiment.py",["--mode",args.mode,"finalize"]),
    ]
    for name,script,extra in stages:
        marker=markers/f"{name}.done"
        if marker.exists(): print(f"{args.mode}: skip {name}",flush=True); continue
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {args.mode}: start {name}",flush=True)
        subprocess.run([sys.executable,"-u",str(ROOT/script),"--config",str(cfg),*extra],cwd=ROOT,check=True); marker.write_text(datetime.now().isoformat()+"\n",encoding="utf-8")
    print(f"{args.mode}: pipeline completed",flush=True)
if __name__=="__main__":main()
