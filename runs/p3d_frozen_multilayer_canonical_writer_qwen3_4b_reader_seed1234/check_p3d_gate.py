import argparse

from p3d_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--eval", required=True); parser.add_argument("--train", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); evaluation, training = read_json(args.eval), read_json(args.train)
    conditions = evaluation["conditions"]
    correct = conditions["correct"]; zero = conditions["zero"]; shuffled = conditions["shuffled"]
    mismatch = 0.5 * (conditions["k_correct_v_wrong"]["f1"] + conditions["k_wrong_v_correct"]["f1"])
    checks = {
        "correct_em_at_least_0_90": correct["exact_match"] >= 0.90,
        "correct_f1_exceeds_zero_by_0_50": correct["f1"] - zero["f1"] >= 0.50,
        "correct_f1_exceeds_shuffled_by_0_50": correct["f1"] - shuffled["f1"] >= 0.50,
        "correct_f1_exceeds_kv_mismatch_by_0_30": correct["f1"] - mismatch >= 0.30,
        "reader_off_matches_question_only": evaluation["reader_off_question_only_consistency"] >= 0.99,
        "gate_opened": training["final_gate_mean"] > training["initial_gate"],
    }
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "metrics": {"correct": correct, "zero": zero, "shuffled": shuffled, "mismatch_f1": mismatch}, "training": training}
    write_json(args.out, result)
    if result["status"] != "passed": raise SystemExit("P3-D small overfit gate failed")


if __name__ == "__main__": main()
