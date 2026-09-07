import argparse
import os
import traceback
from pathlib import Path

from experiment import ROOT, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int)
    parser.add_argument("--output", default="formal")
    args = parser.parse_args()
    output_root = ROOT / "runs" / args.output
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(output_root / "status.json", {"status": "running", "pid": os.getpid()})
    argv = ["--output", args.output]
    if args.samples is not None:
        argv += ["--samples", str(args.samples)]
    try:
        from experiment import main as experiment_main
        import sys
        sys.argv = ["experiment.py", *argv]
        experiment_main()
    except BaseException as error:
        save_json(output_root / "status.json", {"status": "failed", "error": str(error), "pid": os.getpid()})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
