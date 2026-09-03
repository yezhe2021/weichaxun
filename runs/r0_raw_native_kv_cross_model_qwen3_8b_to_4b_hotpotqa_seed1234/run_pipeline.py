#!/home/yezhe/data/miniconda3/envs/attnkv/bin/python
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
CONFIG = ROOT / "config.json"
STATE = ROOT / "artifacts" / "pipeline_state.json"


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"completed": []}


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run(name, *args):
    state = load_state()
    if name in state["completed"]:
        log(f"resume: skip completed stage {name}")
        return
    log(f"stage started: {name}")
    subprocess.run(
        [PYTHON, "-u", str(ROOT / "r0.py"), "--config", str(CONFIG), *args],
        cwd=ROOT,
        check=True,
    )
    state = load_state()
    state["completed"].append(name)
    save_state(state)
    log(f"stage completed: {name}")


def main():
    run("cpu_selftest", "selftest")
    run("prepare", "prepare")
    run("structure", "structure")
    for mode in ("smoke", "formal"):
        run(f"{mode}_replay", "replay", "--mode", mode)
        run(f"{mode}_cache_4b", "extract", "--mode", mode, "--sender", "4b")
        run(f"{mode}_cache_8b", "extract", "--mode", mode, "--sender", "8b")
        run(f"{mode}_lora_self", "train", "--mode", mode, "--reader", "self")
        run(f"{mode}_lora_pair", "train", "--mode", mode, "--reader", "pair")
        run(f"{mode}_evaluation", "evaluate", "--mode", mode)
        if mode == "smoke":
            log("complete smoke test passed; formal R0 is now authorized")
    log("R0 pipeline completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"Pipeline failed: {type(exc).__name__}: {exc}")
        raise
