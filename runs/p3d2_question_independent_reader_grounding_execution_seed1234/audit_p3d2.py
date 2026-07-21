import argparse
from pathlib import Path

import torch

from p3d_common import file_sha256, read_json, write_json
from p3d2_common import CONFIGURATIONS, canonical_cache, load_span_probe


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True); parser.add_argument("--p3d-reader", required=True); parser.add_argument("--model", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); protocol = read_json(args.protocol); errors = []; item = protocol["canonical16"]
    if item["groups"] != 16 or item["memory_dim"] != 256: errors.append("uniform16 interface is not [16,T,256]")
    if file_sha256(item["writer_checkpoint"]) != item["writer_sha256"]: errors.append("P3-C Writer hash mismatch")
    for split in ("train", "validation", "test"):
        cache = canonical_cache(protocol, split); payload = cache.load(0)
        if tuple(payload["keys"].shape[:1] + payload["keys"].shape[2:]) != (16, 256): errors.append(f"{split} Canonical shape mismatch")
        if not cache.index.get("question_independent"): errors.append(f"{split} cache is not question-independent")
    receiver = torch.load(args.p3d_reader, map_location="cpu", weights_only=False)
    if receiver["reader_metadata"]["groups"] != 16 or receiver["reader_metadata"]["memory_dim"] != 256: errors.append("P3-D Reader interface mismatch")
    config = read_json(Path(args.model) / "config.json")
    if config.get("num_hidden_layers") != 36 or config.get("hidden_size") != 2560: errors.append("Unexpected Qwen3-4B geometry")
    probe, probe_path = load_span_probe(protocol, torch.device("cpu")); del probe
    if not probe_path.exists(): errors.append("P3-C fresh span probe missing")
    write_json(args.out, {"status": "complete" if not errors else "failed", "errors": errors, "writer_sha256": item["writer_sha256"], "writer_checkpoint": item["writer_checkpoint"], "p3d_reader": args.p3d_reader, "span_probe": str(probe_path), "configurations": CONFIGURATIONS, "step4_receiver_adaptation_enabled": False})
    if errors: raise SystemExit("; ".join(errors))


if __name__ == "__main__": main()
