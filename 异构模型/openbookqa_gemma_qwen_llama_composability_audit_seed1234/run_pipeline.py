import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def stamp(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    paths = [cfg[key] for key in ("forward_stage_a_checkpoint", "forward_residual_checkpoint",
                                  "reverse_stage_a_checkpoint", "reverse_residual_checkpoint")]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen checkpoints: {missing}")
    subprocess.run(["nvidia-smi"], check=True)
    stamp("Frozen two-writer composability audit started")
    subprocess.run([sys.executable, "-u", str(ROOT / "experiment.py"), "--config", args.config],
                   cwd=ROOT, check=True)
    stamp("ALL EXPERIMENTS COMPLETED")


if __name__ == "__main__":
    main()
