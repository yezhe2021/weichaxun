import argparse
import os
import sys
import traceback

from experiment import ROOT, main as experiment_main, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int)
    parser.add_argument("--output", default="formal")
    args = parser.parse_args()
    root = ROOT / "runs" / args.output
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "status.json", {"status": "running", "pid": os.getpid()})
    sys.argv = ["experiment.py", "--output", args.output]
    if args.samples is not None:
        sys.argv += ["--samples", str(args.samples)]
    try:
        experiment_main()
    except BaseException as error:
        save_json(root / "status.json", {"status": "failed", "error": str(error), "pid": os.getpid()})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
