import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    keys = ("upstream_stage_a_checkpoint", "upstream_residual_checkpoint",
            "downstream_stage_a_checkpoint", "downstream_residual_checkpoint")
    missing = [cfg[key] for key in keys if not Path(cfg[key]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen checkpoints: {missing}")
    subprocess.run(["nvidia-smi"], check=True)
    print("FROZEN LLAMA-QWEN-GEMMA COMPOSABILITY AUDIT STARTED", flush=True)
    subprocess.run([sys.executable, "-u", str(ROOT / "experiment.py")], cwd=ROOT, check=True)
    print("ALL EXPERIMENTS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
