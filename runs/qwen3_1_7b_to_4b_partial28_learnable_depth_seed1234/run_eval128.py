from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from data import load_json


ROOT = Path(__file__).resolve().parent


def main():
    cfg = ROOT / "config.json"; work = Path(load_json(cfg)["work_dir"]); markers = work / "artifacts" / "eval128" / "stage_markers"; markers.mkdir(parents=True, exist_ok=True)
    conditions = ["partial28_skip_f0", "partial28_skip_ce", "partial28_repeat_f0", "partial28_repeat_ce", "repeat_continued_f0", "repeat_continued_ce", "learnable_matrix_f0", "learnable_matrix_ce"]
    stages = ["prepare", "cache", "baseline4", "baseline17", *conditions, "finalize"]
    for stage in stages:
        marker = markers / f"{stage}.done"
        if marker.exists(): print(f"eval128: skip {stage}", flush=True); continue
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] eval128: start {stage}", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "eval128.py"), "--config", str(cfg), stage], cwd=ROOT, check=True)
        marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    print("eval128: pipeline completed", flush=True)


if __name__ == "__main__": main()
