import argparse

from p3d3_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--result", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--min-correct-f1", type=float, default=0.85); parser.add_argument("--min-gap", type=float, default=0.40); parser.add_argument("--min-off-consistency", type=float, default=0.999)
    args = parser.parse_args(); result = read_json(args.result); correct = result["conditions"]["correct_learned_writer16"]["f1"]
    gap, off = result["correct_shuffled_f1_gap"], result["reader_off_exact_output_consistency"]
    passed = correct >= args.min_correct_f1 and gap >= args.min_gap and off >= args.min_off_consistency
    write_json(args.out, {"status": "complete", "passed": passed, "thresholds": {"correct_f1": args.min_correct_f1, "gap": args.min_gap, "off_consistency": args.min_off_consistency},
                          "observed": {"correct_f1": correct, "gap": gap, "off_consistency": off}, "formal_run_policy": "continue_for_diagnosis_regardless_of_gate"})
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__": main()
