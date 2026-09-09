#!/home/yezhe/data/miniconda3/envs/attnkv/bin/python
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
CONFIG = ROOT / "config.json"
STATE = ROOT / "artifacts" / "pipeline_state.json"


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"completed": []}


def save(value):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run(name, script, *arguments):
    current = state()
    if name in current["completed"]:
        log(f"resume: skip completed stage {name}")
        return
    log(f"stage started: {name}")
    subprocess.run(
        [PYTHON, "-u", str(ROOT / script), "--config", str(CONFIG), *arguments],
        cwd=ROOT,
        check=True,
    )
    current = state()
    current["completed"].append(name)
    save(current)
    log(f"stage completed: {name}")


def main():
    run("cpu_selftest", "prepare_r1_assets.py", "--mode", "smoke", "--action", "selftest")
    run("structure", "prepare_r1_assets.py", "--mode", "smoke", "--action", "structure")
    for mode in ("smoke", "formal"):
        run(f"{mode}_manifest", "prepare_r1_assets.py", "--mode", mode, "--action", "manifest")
        run(
            f"{mode}_assets_4b",
            "prepare_r1_assets.py",
            "--mode",
            mode,
            "--action",
            "extract",
            "--sender",
            "4b",
        )
        run(
            f"{mode}_assets_8b",
            "prepare_r1_assets.py",
            "--mode",
            mode,
            "--action",
            "extract",
            "--sender",
            "8b",
        )
        run(
            f"{mode}_r1_0_sparse_gate",
            "train_r1_sparse_native_reader.py",
            "--mode",
            mode,
        )
        run(
            f"{mode}_r1_1_reconstruction",
            "train_r1_translator_warmup.py",
            "--mode",
            mode,
        )
        run(
            f"{mode}_r1_1_functional",
            "train_r1_self_canonical.py",
            "--mode",
            mode,
        )
        run(
            f"{mode}_r1_2_cross_evaluation",
            "eval_r1_cross_canonical.py",
            "--mode",
            mode,
        )
        if mode == "smoke":
            log("R1 complete smoke pipeline passed; formal pipeline authorized")
    log("R1 pipeline completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"Pipeline failed: {type(error).__name__}: {error}")
        raise
