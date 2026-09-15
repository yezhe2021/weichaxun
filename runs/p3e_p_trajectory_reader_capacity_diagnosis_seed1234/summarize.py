import argparse

from p3d3_common import read_json, write_json


BASELINE_MAP = {
    "question_only": "question_only",
    "reader_c_native_kv": "qwen3_4b_qconditioned_native_reader_c",
    "full_evidence_text": "full_evidence_text",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    evaluation = read_json(args.evaluation)
    baseline = read_json(args.baseline)
    audit = read_json(args.audit)
    table = {
        label: baseline["conditions"][source]
        for label, source in BASELINE_MAP.items()
    }
    table.update(evaluation["conditions"])
    reader_f1 = table["reader_c_native_kv"]["f1"]
    text_f1 = table["full_evidence_text"]["f1"]
    for method in ("a0", "a1", "c1"):
        correct = table[f"{method}_correct_kv"]
        shuffled = table[f"{method}_hard_shuffled"]
        denominator = text_f1 - reader_f1
        correct["correct_shuffled_f1_gap"] = correct["f1"] - shuffled["f1"]
        correct["automatic_f1_gap_recovery"] = (
            (correct["f1"] - reader_f1) / denominator
            if abs(denominator) > 1e-12
            else None
        )

    write_json(
        args.out,
        {
            "status": "complete",
            "experiment": "P3-E-P Trajectory-Supervised Memory Interface Diagnosis",
            "diagnostic_table": table,
            "state_diagnostics": evaluation["state_diagnostics"],
            "leakage_audit": audit,
            "strict_accuracy_gap_recovery_formula": "(method_strict - reader_c_strict) / (text_strict - reader_c_strict)",
            "automatic_gap_recovery_uses_f1": True,
            "teacher_used_in_validation_generation": False,
            "selected_layers": [0, 2, 5, 7, 9, 12, 14, 16, 19, 21, 23, 26, 28, 30, 33, 35],
        },
    )


if __name__ == "__main__":
    main()

