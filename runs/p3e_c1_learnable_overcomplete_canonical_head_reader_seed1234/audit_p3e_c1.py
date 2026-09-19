import argparse
from pathlib import Path

import torch

from p3d3_common import file_sha256, read_json, write_json
from p3e_c1_common import DuplicateHeadwiseCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True); parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--native-reader", required=True); parser.add_argument("--c0-success", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); errors = []
    train, validation = DuplicateHeadwiseCache(args.train_cache), DuplicateHeadwiseCache(args.validation_cache)
    if len(train) != 512: errors.append(f"Expected 512 training samples, got {len(train)}")
    if len(validation) != 64: errors.append(f"Expected 64 validation samples, got {len(validation)}")
    if {entry["id"] for entry in train.entries} & {entry["id"] for entry in validation.entries}: errors.append("Train/validation overlap")
    for label, cache in (("train", train), ("validation", validation)):
        payload = cache.load(0)
        if payload["keys"].shape[0] != 16 or payload["keys"].shape[-2:] != (16, 128): errors.append(f"{label} duplicate geometry mismatch")
    checkpoint = torch.load(args.native_reader, map_location="cpu", weights_only=False); metadata = checkpoint.get("reader_metadata", {})
    if metadata.get("selected_layers") is None or metadata.get("rank") != 32: errors.append("Unexpected P3-E-B native Reader metadata")
    c0 = read_json(args.c0_success)
    if c0.get("status") != "complete" or not c0.get("equivalence", {}).get("passed"): errors.append("P3-E-C0 strict duplicate equivalence did not pass")
    result = {"status": "complete" if not errors else "failed", "errors": errors,
        "invariants": {"duplicate_writer16_frozen": True, "writer_trainable_parameters": 0, "canonical_memory": "[16,T,16,128]",
                       "receiver_backbone_frozen": True, "native_o_proj_frozen": True, "train_samples": len(train), "validation_samples": len(validation)},
        "native_reader": args.native_reader, "native_reader_sha256": file_sha256(args.native_reader), "c0_success": args.c0_success,
        "c0_success_sha256": file_sha256(args.c0_success), "selected_layers": metadata.get("selected_layers")}
    write_json(args.out, result)
    if errors: raise RuntimeError("; ".join(errors))


if __name__ == "__main__": main()
