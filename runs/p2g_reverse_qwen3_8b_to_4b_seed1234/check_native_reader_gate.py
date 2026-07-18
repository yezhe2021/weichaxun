import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Apply the P2-G1 Native Reader success gate")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--paired-threshold", type=float, default=0.90)
    args = parser.parse_args()

    with open(args.eval, encoding="utf-8") as handle:
        summary = json.load(handle)
    conditions = {row["condition"]: row for row in summary["conditions"]}
    required = (
        "full_text_prefilled_final_base",
        "full_text_prefilled_final_counterfactual",
        "correct_kv",
        "counterfactual_kv",
        "shuffled_kv",
        "mismatched_kv",
        "zero_kv",
        "reader_off",
    )
    missing = [name for name in required if name not in conditions]
    if missing:
        raise RuntimeError(f"Missing required evaluation conditions: {missing}")

    paired = float(summary["kv_paired_consistency"])
    passed = paired >= args.paired_threshold
    report = {
        "status": "passed" if passed else "failed",
        "enter_writer_stage": passed,
        "paired_threshold": args.paired_threshold,
        "native_kv": {
            "base_em": conditions["correct_kv"]["target_em"],
            "counterfactual_em": conditions["counterfactual_kv"]["target_em"],
            "paired_consistency": paired,
            "prediction_switch_rate": summary["kv_prediction_switch_rate"],
        },
        "full_text_upper_bound": {
            "base_em": conditions["full_text_prefilled_final_base"]["target_em"],
            "counterfactual_em": conditions["full_text_prefilled_final_counterfactual"]["target_em"],
            "paired_consistency": summary["full_text_prefilled_final_paired_consistency"],
        },
        "controls": {
            name: conditions[name]["target_em"]
            for name in ("shuffled_kv", "mismatched_kv", "zero_kv", "reader_off")
        },
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    filename = "NATIVE_READER_GATE_PASSED.json" if passed else "NATIVE_READER_GATE_FAILED.json"
    with open(output / filename, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
