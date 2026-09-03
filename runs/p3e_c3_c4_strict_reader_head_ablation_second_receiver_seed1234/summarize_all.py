import argparse

from p3d3_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--c3a", required=True); parser.add_argument("--c3b", required=True); parser.add_argument("--c4", required=True); parser.add_argument("--writer", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); write_json(args.out, {"status": "complete", "experiment": "C3 strict onboarding, head differentiation, and C4 second Receiver",
        "frozen_writer": args.writer, "c3a": read_json(args.c3a), "c3b": read_json(args.c3b), "c4": read_json(args.c4),
        "execution_order": ["fully_random seeds", "weak_pair seeds", "paired head ablation", "Qwen3.5 Receiver seeds"], "writer_updated": False})


if __name__ == "__main__": main()
