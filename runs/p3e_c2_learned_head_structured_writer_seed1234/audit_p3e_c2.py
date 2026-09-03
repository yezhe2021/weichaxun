import argparse

import torch

from p3d3_common import file_sha256, read_json, write_json
from p3e_c2_common import SenderNativeHeadwiseCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True); parser.add_argument("--validation-cache", required=True); parser.add_argument("--native-reader", required=True)
    parser.add_argument("--c1-reader", required=True); parser.add_argument("--c1-success", required=True); parser.add_argument("--c0-success", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); errors = []; train, validation = SenderNativeHeadwiseCache(args.train_cache), SenderNativeHeadwiseCache(args.validation_cache)
    if len(train) != 512 or len(validation) != 64: errors.append("Expected train512/validation64")
    if {entry["id"] for entry in train.entries} & {entry["id"] for entry in validation.entries}: errors.append("Train/validation overlap")
    for label, cache in (("train", train), ("validation", validation)):
        payload = cache.load(0)
        if payload["keys"].shape[0] != 16 or payload["keys"].shape[-2:] != (8, 128): errors.append(f"{label} Native geometry mismatch")
    native = torch.load(args.native_reader, map_location="cpu", weights_only=False); c1_reader = torch.load(args.c1_reader, map_location="cpu", weights_only=False)
    if native.get("reader_metadata", {}).get("rank") != 32 or c1_reader.get("reader_metadata", {}).get("top_k") != 2: errors.append("Reader metadata mismatch")
    if read_json(args.c0_success).get("status") != "complete" or read_json(args.c1_success).get("status") != "complete": errors.append("C0/C1 completion marker missing")
    write_json(args.out, {"status": "complete" if not errors else "failed", "errors": errors,
        "invariants": {"sender_cache": "Qwen3-8B evidence-only [16,T,8,128]", "writer_output": "[16,T,16,128]", "writer_only_training": True,
                       "K_V_head_route_shared": True, "cross_token_mixing": False, "cross_layer_mixing": False,
                       "receiver_backbone_frozen": True, "writer_training_reader_frozen": True, "fresh_reader_does_not_load_C1": True},
        "native_reader_sha256": file_sha256(args.native_reader), "c1_reader_sha256": file_sha256(args.c1_reader),
        "c0_success_sha256": file_sha256(args.c0_success), "c1_success_sha256": file_sha256(args.c1_success)})
    if errors: raise RuntimeError("; ".join(errors))


if __name__ == "__main__": main()
