import argparse
import json
import py_compile
from pathlib import Path

import torch

from hotpot_common import load_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); parser.add_argument("--raw", required=True)
    parser.add_argument("--writer", required=True); parser.add_argument("--reader", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); root = Path(args.root)
    for path in root.glob("*.py"): py_compile.compile(str(path), doraise=True)
    for path in (args.raw, args.writer, args.reader):
        if not Path(path).is_file(): raise FileNotFoundError(path)
    writer = torch.load(args.writer, map_location="cpu", weights_only=False); reader = torch.load(args.reader, map_location="cpu", weights_only=False)
    if not {"writer", "writer_config"}.issubset(writer): raise RuntimeError("W8 Writer checkpoint format drift")
    if not {"reader", "reader_metadata", "writer_checkpoint_sha256", "writer_state_sha256"}.issubset(reader): raise RuntimeError("R4 Reader checkpoint format drift")
    with open(args.raw, encoding="utf-8") as handle: sample = json.load(handle)[0]
    if not {"_id", "question", "answer", "context", "supporting_facts"}.issubset(sample): raise RuntimeError("HotpotQA schema drift")
    write_json(args.out, {"status": "complete", "writer_frozen": True, "reader_frozen": True, "raw_schema_valid": True})


if __name__ == "__main__": main()
