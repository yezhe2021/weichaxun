import argparse
import json
from pathlib import Path

from p2is_common import write_json


def load(path):
    with open(path, encoding="utf-8") as handle: return json.load(handle)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); args = parser.parse_args(); root = Path(args.root)
    configs = ("imitation_only", "q4_only", "dual_only", "full")
    results = {}
    for config in configs:
        results[config] = {
            "qwen3_4b": load(root / f"evaluation/{config}/qwen3_4b/SUCCESS.json"),
            "qwen3_5_4b": load(root / f"evaluation/{config}/qwen3_5_4b/SUCCESS.json"),
        }
    full = results["full"]
    q4_pass = full["qwen3_4b"]["within_five_points_of_old"] and full["qwen3_4b"]["control_gate"]
    q35_pass = full["qwen3_5_4b"]["within_five_points_of_old"] and full["qwen3_5_4b"]["control_gate"]
    if q4_pass and q35_pass:
        verdict = "new_sender_single_writer_onboarded_to_both_frozen_readers"
    elif q4_pass and not q35_pass:
        verdict = "single_reader_bias_qwen3_5_zero_shot_failed"
    else:
        imitation = results["imitation_only"]
        verdict = "canonical_geometry_not_sufficient_for_receiver_execution" if not (imitation["qwen3_4b"]["within_five_points_of_old"] and imitation["qwen3_5_4b"]["within_five_points_of_old"]) else "functional_calibration_failed_despite_imitation"
    hashes = {result[receiver]["old_writer_checkpoint_sha256"] for result in results.values() for receiver in result}
    if len(hashes) != 1: raise RuntimeError("Ablations did not use one old public Writer/Reader interface")
    write_json(root / "SUCCESS.json", {
        "status": "complete", "verdict": verdict, "full_qwen3_4b_passed": q4_pass,
        "full_qwen3_5_4b_passed": q35_pass, "single_main_writer_checkpoint": str(root / "FINAL_W4B_CHECKPOINT.pt"),
        "old_public_writer_checkpoint_sha256": next(iter(hashes)), "results": results,
    })


if __name__ == "__main__": main()
