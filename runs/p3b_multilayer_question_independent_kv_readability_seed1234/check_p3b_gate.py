import argparse
from pathlib import Path

from p3b_common import read_json, write_json


def condition(result, name):
    return next(row for row in result["conditions"] if row["condition"] == name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-correct-em", type=float, default=0.90)
    parser.add_argument("--min-zero-gap", type=float, default=0.50)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for source in ("hidden", "native_kv", "trainable"):
        path = root / "overfit" / "evidence_only" / source / "all36" / "SUCCESS.json"
        result = read_json(path)
        correct, zero, shuffled = condition(result, "correct"), condition(result, "zero"), condition(result, "shuffled")
        passed = correct["current_answer_em"] >= args.min_correct_em and correct["current_answer_em"] - zero["current_answer_em"] >= args.min_zero_gap
        rows.append(
            {
                "source": source,
                "correct_em": correct["current_answer_em"],
                "zero_em": zero["current_answer_em"],
                "shuffled_current_em": shuffled["current_answer_em"],
                "shuffled_source_em": shuffled["source_memory_em"],
                "passed": passed,
            }
        )
    passed = any(row["passed"] for row in rows)
    report = {
        "status": "complete" if passed else "failed",
        "passed": passed,
        "criteria": {"min_correct_em": args.min_correct_em, "min_zero_gap": args.min_zero_gap},
        "branches": rows,
        "note": "Cross-sample source-answer following is reported but is not a gate: the current question does not identify the source sample's answer.",
    }
    write_json(args.out, report)
    if not passed:
        raise SystemExit("P3-B overfit validity gate failed")


if __name__ == "__main__":
    main()
