import argparse
import py_compile
from pathlib import Path

import torch

from p3e_d_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    files = ["p3e_d_common.py", "prepare_manifest.py", "cache_sender_artifacts.py", "eval_current_system.py", "summarize.py"]
    for name in files:
        py_compile.compile(str(root / name), doraise=True)
    manifest = read_json(args.manifest)
    if manifest["sample_count"] != 64:
        raise RuntimeError("P3-E-D must use exactly 64 fixed validation samples")
    for asset in ("writer", "canonical_reader", "native_reader"):
        if not Path(manifest["assets"][asset]["path"]).is_file():
            raise RuntimeError(f"Missing frozen asset: {asset}")
    write_json(args.out, {
        "status": "passed",
        "samples": 64,
        "conditions": manifest["conditions"],
        "cuda_available": torch.cuda.is_available(),
        "checkpoints_frozen_by_protocol": True,
        "receiver_input_truncation": False,
    })


if __name__ == "__main__":
    main()
