import argparse

from p3d3_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--native", required=True); parser.add_argument("--c0", required=True); parser.add_argument("--c1", required=True)
    parser.add_argument("--writer-reader", required=True); parser.add_argument("--fresh-reader", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_json(args.out, {"status": "complete", "experiment": "P3-E-C2 learned overcomplete Head-Structured Writer and fresh Reader",
        "references": {"native_headwise": read_json(args.native), "fixed_duplicate_c0": read_json(args.c0), "learnable_reader_c1": read_json(args.c1)},
        "learned_writer_with_training_reader": read_json(args.writer_reader), "frozen_writer_with_fresh_reader": read_json(args.fresh_reader),
        "protocol_claim_limit": "Fresh Qwen3-4B Reader tests reader decoupling; a second Receiver is still required for receiver-independent public-bus evidence."})


if __name__ == "__main__": main()
