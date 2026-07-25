import argparse
from pathlib import Path

from p3d3_common import load_jsonl, read_json, write_json


BASELINE_MAP = {
    "question_only": "question_only",
    "reader_c_native_kv": "qwen3_4b_qconditioned_native_reader_c",
    "full_evidence_text": "full_evidence_text",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    oracle = read_json(args.oracle)
    baseline = read_json(args.baseline)
    baseline_records = load_jsonl(args.baseline_records)
    table = {
        label: baseline["conditions"][source]
        for label, source in BASELINE_MAP.items()
    }
    table.update(oracle["conditions"])

    baseline_predictions = {
        condition: {
            row["id"]: row["prediction"]
            for row in baseline_records
            if row["condition"] == source
        }
        for condition, source in BASELINE_MAP.items()
    }
    oracle_records = load_jsonl(
        Path(args.oracle).parent / "per_sample_generation.jsonl"
    )
    switches = {}
    question_predictions = baseline_predictions["question_only"]
    for condition in oracle["conditions"]:
        rows = [row for row in oracle_records if row["condition"] == condition]
        switches[condition] = sum(
            row["prediction"] != question_predictions.get(row["id"], "")
            for row in rows
        ) / max(1, len(rows))
        table[condition]["prediction_switch_vs_question_only"] = switches[
            condition
        ]

    write_json(
        args.out,
        {
            "status": "complete",
            "experiment": "P3-E-O Native Trajectory Oracle Patching",
            "diagnostic_table": table,
            "selected_layers": oracle["selected_layers"],
            "free_running": True,
            "gold_answer_prefix_used": False,
            "current_token_only_patch": True,
            "position_alignment": {
                "o1_o2_o3": "aligned to full-text answer suffix positions",
                "sanity": "O3 with ordinary question-only positions",
            },
            "oracle_detail": args.oracle,
            "baseline_detail": args.baseline,
        },
    )


if __name__ == "__main__":
    main()

