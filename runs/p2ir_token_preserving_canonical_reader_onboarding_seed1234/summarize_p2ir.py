import argparse
from pathlib import Path

from p2ir_common import write_json
import json


def load(path):
    with open(path, encoding="utf-8") as handle: return json.load(handle)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); args = parser.parse_args()
    root = Path(args.root)
    paths = {
        "qwen3_4b_small": root / "qwen3_4b/small/eval/SUCCESS.json",
        "qwen3_4b_full": root / "qwen3_4b/full/eval/SUCCESS.json",
        "qwen3_5_4b_small": root / "qwen3_5_4b/small/eval/SUCCESS.json",
        "qwen3_5_4b_full": root / "qwen3_5_4b/full/eval/SUCCESS.json",
    }
    results = {name: load(path) for name, path in paths.items()}
    hashes = {value["writer_checkpoint_sha256"] for value in results.values()}
    if len(hashes) != 1:
        raise RuntimeError("Receivers were evaluated against different Writer checkpoints")
    q4 = results["qwen3_4b_full"]["threshold_passed"]
    q35 = results["qwen3_5_4b_full"]["threshold_passed"]
    if q4 and q35: verdict = "both_receivers_succeeded_receiver_independent_interface_supported"
    elif q4: verdict = "qwen3_4b_succeeded_qwen3_5_failed_mixed_architecture_reader_problem"
    else: verdict = "qwen3_4b_failed_probe_decodable_not_yet_receiver_executable"
    write_json(root / "SUCCESS.json", {
        "status": "complete", "verdict": verdict, "qwen3_4b_passed": q4,
        "qwen3_5_4b_passed": q35, "writer_checkpoint_sha256": next(iter(hashes)), "results": results,
    })


if __name__ == "__main__":
    main()
