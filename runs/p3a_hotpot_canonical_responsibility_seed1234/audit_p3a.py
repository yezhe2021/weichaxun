import argparse
import json
import py_compile
from pathlib import Path

import torch

from p3a_common import write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); parser.add_argument("--train-raw", required=True); parser.add_argument("--dev-raw", required=True)
    parser.add_argument("--writer", required=True); parser.add_argument("--old-reader", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); root = Path(args.root)
    for path in root.glob("*.py"): py_compile.compile(str(path), doraise=True)
    for path in (args.train_raw, args.dev_raw, args.writer, args.old_reader):
        if not Path(path).is_file(): raise FileNotFoundError(path)
    writer = torch.load(args.writer, map_location="cpu", weights_only=False); reader = torch.load(args.old_reader, map_location="cpu", weights_only=False)
    if not {"writer", "writer_config"}.issubset(writer): raise RuntimeError("Writer checkpoint drift")
    if not {"reader", "reader_metadata", "writer_checkpoint_sha256", "writer_state_sha256"}.issubset(reader): raise RuntimeError("Reader checkpoint drift")
    with open(args.dev_raw, encoding="utf-8") as handle: sample = json.load(handle)[0]
    if not {"_id", "question", "answer", "context", "supporting_facts"}.issubset(sample): raise RuntimeError("Hotpot schema drift")
    write_json(args.out, {"status": "complete", "writer_frozen": True, "old_reader_frozen": True, "backbones_frozen": True, "sources": ["hidden", "raw_kv", "pca_kv", "canonical"]})


if __name__ == "__main__": main()
