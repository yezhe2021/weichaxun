from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def link(path,target):
    path,target=Path(path),Path(target).resolve(); path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():
        if path.resolve()==target: return
        path.unlink()
    elif path.exists(): raise RuntimeError(f"refusing to replace {path}")
    path.symlink_to(target,target_is_directory=target.is_dir())


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args(); cfg=load(args.config)
    work=Path(cfg["work_dir"]); old=Path(cfg["reusable_experiment_dir"]); audit=Path(cfg["cache_audit_1_7b_dir"])
    summary=load(audit/"artifacts"/"development"/"summary.json")
    if summary["samples"]!=64: raise RuntimeError("1.7B cache audit is incomplete")
    if not summary["conditions"]["official_native_cache"]["generation_match_vs_text"]==1.0: raise RuntimeError("1.7B official cache generation mismatch")
    link(work/"artifacts"/"manifest.json",old/"artifacts"/"manifest.json")
    link(work/"artifacts"/"protocol.json",old/"artifacts"/"protocol.json")
    for mode in ("smoke","development"):
        link(work/"cache"/mode/"target4",old/"cache"/"development"/"target4")
        link(work/"cache"/mode/"teacher4",old/"cache"/"development"/"teacher4")
    save(work/"artifacts"/"cache_audit_continuation_decision.json",{
        "authorized_to_continue":True,"original_gate_passed":summary["gate"]["passed"],
        "reason":"User accepted the 0.3125 percentage-point FP16 token-top1 mismatch as immaterial.",
        "top1_match_rate":summary["logit_comparisons"]["full_text_vs_official"]["top1_match_rate"],
        "generation_match":summary["conditions"]["official_native_cache"]["generation_match_vs_text"],
        "mean_kl":summary["logit_comparisons"]["full_text_vs_official"]["mean_kl"],
    })
    print("receiver assets linked; cache-audit continuation recorded",flush=True)


if __name__=="__main__": main()
