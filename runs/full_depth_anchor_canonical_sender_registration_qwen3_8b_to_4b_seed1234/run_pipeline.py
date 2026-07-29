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


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"completed": []}


def save_state(value):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run(name, script, *arguments):
    current = load_state()
    if name in current["completed"]:
        log(f"resume: skip completed stage {name}")
        return
    log(f"stage started: {name}")
    subprocess.run(
        [PYTHON, "-u", str(ROOT / script), "--config", str(CONFIG), *arguments],
        cwd=ROOT,
        check=True,
    )
    current = load_state()
    current["completed"].append(name)
    save_state(current)
    log(f"stage completed: {name}")


def main():
    run("cpu_selftest", "prepare_assets.py", "--mode", "smoke", "--action", "selftest")
    run("structure", "prepare_assets.py", "--mode", "smoke", "--action", "structure")
    for mode in ("smoke", "formal"):
        run(
            f"{mode}_reference",
            "prepare_assets.py",
            "--mode",
            mode,
            "--action",
            "reference",
        )
        run(
            f"{mode}_question_queries",
            "prepare_assets.py",
            "--mode",
            mode,
            "--action",
            "queries",
        )
        run(
            f"{mode}_reader_protocol_gate",
            "verify_reader_protocol.py",
            "--mode",
            mode,
        )
        run(
            f"{mode}_writer_warmup",
            "train_writer_warmup.py",
            "--mode",
            mode,
        )
        run(
            f"{mode}_writer_functional",
            "train_writer_functional.py",
            "--mode",
            mode,
        )
        run(f"{mode}_evaluation", "evaluate.py", "--mode", mode)
        if mode == "smoke":
            log("Full-depth anchor smoke passed; formal pipeline authorized")
    log("Full-depth anchor pipeline completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"Pipeline failed: {type(error).__name__}: {error}")
        raise
