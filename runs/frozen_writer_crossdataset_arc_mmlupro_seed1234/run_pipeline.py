import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def call(script, *arguments):
    command = [sys.executable, "-u", str(ROOT / script), *arguments]
    print("START", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print("DONE ", " ".join(command), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    parser.add_argument("--dataset", choices=("arc_challenge", "mmlu_pro", "all"), default="all")
    parser.add_argument("stage", choices=("prepare", "evaluate", "all"), default="all", nargs="?")
    args = parser.parse_args()
    datasets = ("arc_challenge", "mmlu_pro") if args.dataset == "all" else (args.dataset,)
    if args.stage in ("prepare", "all"):
        call("prepare.py", "--mode", args.mode, "--dataset", args.dataset, "all")
    if args.stage in ("evaluate", "all"):
        for dataset in datasets:
            for receiver in ("qwen", "gemma", "llama"):
                call("evaluate.py", "--mode", args.mode, "--dataset", dataset, "--receiver", receiver)
            call("summarize.py", "--mode", args.mode, "--dataset", dataset)
    print("ALL EXPERIMENTS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
