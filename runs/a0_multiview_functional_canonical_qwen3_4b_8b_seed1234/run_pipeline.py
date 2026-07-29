import argparse
import datetime
import os
import subprocess
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get(
    "EXPERIMENT_PYTHON", "/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
)
CONFIG = ROOT / "config.json"


def log(message):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def command(marker_name, script, mode, *extra):
    marker = ROOT / "artifacts" / mode / "stage_markers" / f"{marker_name}.done"
    if marker.exists():
        log(f"SKIP completed stage {marker_name}")
        return
    argv = [
        PYTHON,
        "-u",
        str(ROOT / script),
        "--config",
        str(CONFIG),
        "--mode",
        mode,
        *extra,
    ]
    log("RUN " + " ".join(argv))
    subprocess.run(argv, check=True, cwd=ROOT)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.datetime.now().isoformat(), encoding="utf-8")
    log(f"DONE {marker_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    mode = args.mode
    try:
        command("selftest", "prepare_assets.py", mode, "--action", "selftest")
        command("structure", "prepare_assets.py", mode, "--action", "structure")
        command("reference", "prepare_assets.py", mode, "--action", "reference")
        command("query_4b", "prepare_assets.py", mode, "--action", "queries", "--sender", "4b")
        command("query_8b", "prepare_assets.py", mode, "--action", "queries", "--sender", "8b")
        command("protocol_audit", "prepare_assets.py", mode, "--action", "audit")
        command("a1", "train_a1_self_bootstrap.py", mode)
        command("a2", "train_a2_cross_coregistration.py", mode)
        command("a3_4b", "train_a3_functional_4b.py", mode)
        command("evaluation", "evaluate_canonical.py", mode)
        command("sender_audit", "audit_sender_leakage.py", mode)
        log(f"Pipeline completed: mode={mode}; A3-8B intentionally not run")
    except Exception as error:
        log(f"Pipeline failed: {type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
