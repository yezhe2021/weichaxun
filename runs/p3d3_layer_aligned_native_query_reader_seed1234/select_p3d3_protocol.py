import argparse
from pathlib import Path

from p3d3_common import SELECTED_LAYERS, file_sha256, read_json, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--p3d-protocol", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); source = read_json(args.p3d_protocol); item = source["canonical16"]
    if item["original_layer_indices"] != SELECTED_LAYERS or item["groups"] != 16 or item["memory_dim"] != 256:
        raise RuntimeError("P3-D canonical16 is not the expected uniform16 protocol")
    writer = Path(item["writer_checkpoint"])
    if file_sha256(writer) != item["writer_sha256"]: raise RuntimeError("P3-C Writer hash changed")
    result = {"status": "complete", "protocol": "P3-C uniform16", "writer_checkpoint": str(writer.resolve()), "writer_sha256": item["writer_sha256"],
              "selected_sender_layers": SELECTED_LAYERS, "selected_receiver_layers": SELECTED_LAYERS,
              "fixed_one_to_one_depth_alignment": True, "question_independent": True, "canonical_dim": 256,
              "native_dim": 1024, "source_protocol": str(Path(args.p3d_protocol).resolve())}
    write_json(args.out, result)


if __name__ == "__main__": main()
