import argparse
from pathlib import Path

from p3b_common import read_json, write_json


def condition(result, name):
    return next(row for row in result["conditions"] if row["condition"] == name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-correct-em", type=float, default=0.90)
    parser.add_argument("--min-zero-gap", type=float, default=0.50)
    args = parser.parse_args()
    result = read_json(args.result)
    correct, zero, shuffled, mismatch = (condition(result, name) for name in ("correct", "zero", "shuffled", "kv_mismatch"))
    passed = correct["current_answer_em"] >= args.min_correct_em and correct["current_answer_em"] - zero["current_answer_em"] >= args.min_zero_gap
    report = {
        "status": "complete" if passed else "failed", "passed": passed,
        "correct_em": correct["current_answer_em"], "zero_em": zero["current_answer_em"],
        "shuffled_current_em": shuffled["current_answer_em"], "shuffled_source_em": shuffled["source_memory_em"],
        "mismatch_em": mismatch["current_answer_em"],
        "criteria": {"min_correct_em": args.min_correct_em, "min_zero_gap": args.min_zero_gap},
        "note": "Cross-sample source following is diagnostic, not a hard gate, because the current question does not identify the source sample's answer span.",
    }
    write_json(args.out, report)
    if not passed:
        raise SystemExit("P3-C Writer/probe overfit gate failed")


if __name__ == "__main__":
    main()
