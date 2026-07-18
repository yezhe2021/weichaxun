import argparse
import csv
import json
from pathlib import Path

from p2id_common import write_json


def read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="P2-I-D validity gate and responsibility table")
    parser.add_argument("--mode", choices=("joint-gate", "summarize"), required=True)
    parser.add_argument("--writer-probe", required=True)
    parser.add_argument("--reader4", required=True)
    parser.add_argument("--reader35", required=True)
    parser.add_argument("--joint")
    parser.add_argument("--joint-rescue")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    writer = read(args.writer_probe)
    reader4 = read(args.reader4)
    reader35 = read(args.reader35)
    writer_pass = bool(writer["writer_probe_passed"])
    reader4_pass = bool(reader4["reader_oracle_passed"])
    reader35_pass = bool(reader35["reader_oracle_passed"])

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "joint-gate":
        allowed = writer_pass and reader4_pass
        write_json(
            output / "JOINT_VALIDITY_GATE.json",
            {
                "status": "complete",
                "joint_allowed": allowed,
                "writer_probe_passed": writer_pass,
                "qwen3_4b_reader_oracle_passed": reader4_pass,
                "qwen3_5_4b_reader_oracle_passed": reader35_pass,
                "reason": "positive controls passed" if allowed else "joint test skipped because a prerequisite failed",
            },
        )
        raise SystemExit(0 if allowed else 3)

    joint = read(args.joint) if args.joint and Path(args.joint).is_file() else None
    rescue = read(args.joint_rescue) if args.joint_rescue and Path(args.joint_rescue).is_file() else None
    joint_pass = None if joint is None else bool(joint["joint_overfit_passed"])
    rescue_pass = None if rescue is None else bool(rescue["joint_overfit_passed"])

    def receiver_verdict(reader_pass, joint_result=None):
        if not writer_pass and reader_pass:
            return "writer_problem"
        if writer_pass and not reader_pass:
            return "reader_or_injection_problem"
        if not writer_pass and not reader_pass:
            return "writer_and_reader_or_implementation_problem"
        if joint_result is False:
            return "writer_reader_protocol_cold_start"
        if joint_result is True:
            return "basic_path_works_full_scale_objective_or_interference"
        return "writer_and_reader_basic_controls_passed"

    receiver_verdicts = {
        "qwen3_4b": receiver_verdict(reader4_pass, joint_pass),
        "qwen3_5_4b": receiver_verdict(reader35_pass, None),
    }
    unique = set(receiver_verdicts.values())
    verdict = next(iter(unique)) if len(unique) == 1 else "receiver_specific_mixed_result"
    explanation = (
        f"Qwen3-4B: {receiver_verdicts['qwen3_4b']}; "
        f"Qwen3.5-4B: {receiver_verdicts['qwen3_5_4b']}."
    )
    if receiver_verdicts["qwen3_4b"] == "writer_reader_protocol_cold_start" and rescue_pass:
        explanation += " Reader warmup rescues 4B joint training, strengthening the cold-start diagnosis."

    rows = [
        {
            "component": "writer_probe",
            "passed": writer_pass,
            "primary_metric": writer.get("test_correct_accuracy"),
            "paired_consistency": writer.get("correct_paired_consistency"),
            "interpretation": "sample-specific canonical content readability",
        },
        {
            "component": "reader_oracle_qwen3_4b",
            "passed": reader4_pass,
            "primary_metric": 0.5 * (reader4["base_em"] + reader4["counterfactual_em"]),
            "paired_consistency": reader4["paired_consistency"],
            "interpretation": "4B Reader and injection capacity",
        },
        {
            "component": "reader_oracle_qwen3_5_4b",
            "passed": reader35_pass,
            "primary_metric": 0.5 * (reader35["base_em"] + reader35["counterfactual_em"]),
            "paired_consistency": reader35["paired_consistency"],
            "interpretation": "Qwen3.5 Reader and injection capacity",
        },
        {
            "component": "real_writer_reader_joint_qwen3_4b",
            "passed": joint_pass,
            "primary_metric": None if joint is None else 0.5 * (joint["base_em"] + joint["counterfactual_em"]),
            "paired_consistency": None if joint is None else joint["paired_consistency"],
            "interpretation": "protocol establishment on the fixed small subset",
        },
    ]
    with open(output / "responsibility_table.csv", "w", encoding="utf-8", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "verdict": verdict,
            "explanation": explanation,
            "writer_probe_passed": writer_pass,
            "reader_oracles": {"qwen3_4b": reader4_pass, "qwen3_5_4b": reader35_pass},
            "receiver_verdicts": receiver_verdicts,
            "joint_overfit_passed": joint_pass,
            "joint_rescue_passed": rescue_pass,
            "rows": rows,
            "attribution_is_valid": bool(writer["synthetic_positive_control"]["passed"]),
        },
    )


if __name__ == "__main__":
    main()
