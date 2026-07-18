import argparse
import json
from pathlib import Path


def read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def metric(summary, condition):
    row = next(item for item in summary["conditions"] if item["condition"] == condition)
    return row["target_em"]


def main():
    parser = argparse.ArgumentParser(description="Combine P2-I mother/frozen/oracle Receiver evaluations")
    parser.add_argument("--mother4", required=True)
    parser.add_argument("--mother35", required=True)
    parser.add_argument("--frozen4", required=True)
    parser.add_argument("--frozen35", required=True)
    parser.add_argument("--oracle4")
    parser.add_argument("--oracle35")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = {
        "mother_qwen3_4b": args.mother4,
        "mother_qwen3_5_4b": args.mother35,
        "frozen_writer_retrained_qwen3_4b": args.frozen4,
        "frozen_writer_retrained_qwen3_5_4b": args.frozen35,
    }
    if args.oracle4:
        paths["oracle_joint_qwen3_4b"] = args.oracle4
    if args.oracle35:
        paths["oracle_joint_qwen3_5_4b"] = args.oracle35
    rows = []
    summaries = {}
    for name, path in paths.items():
        summary = read(path)
        summaries[name] = summary
        rows.append(
            {
                "branch": name,
                "receiver": summary["receiver"],
                "base_em": metric(summary, "correct_slots"),
                "counterfactual_em": metric(summary, "counterfactual_slots"),
                "paired_consistency": summary["canonical_paired_consistency"],
                "prediction_switch": summary["canonical_prediction_switch"],
                "zero_em": metric(summary, "zero_slots"),
                "reader_off_em": metric(summary, "reader_off"),
                "permutation_em": metric(summary, "slot_permutation"),
                "drop_half_em": metric(summary, "drop_half_slots"),
                "permutation_logit_max_diff": summary["slot_permutation_max_logit_difference"],
                "writer_sha256": summary["writer_sha256"],
            }
        )
    mother_hashes = {
        summaries["mother_qwen3_4b"]["writer_sha256"],
        summaries["mother_qwen3_5_4b"]["writer_sha256"],
        summaries["frozen_writer_retrained_qwen3_4b"]["writer_sha256"],
        summaries["frozen_writer_retrained_qwen3_5_4b"]["writer_sha256"],
    }
    if len(mother_hashes) != 1:
        raise RuntimeError("Main evaluations did not use one identical frozen Writer")
    frozen4 = summaries["frozen_writer_retrained_qwen3_4b"]["canonical_paired_consistency"]
    frozen35 = summaries["frozen_writer_retrained_qwen3_5_4b"]["canonical_paired_consistency"]
    oracle_gains = {}
    for receiver in ("qwen3_4b", "qwen3_5_4b"):
        oracle = summaries.get(f"oracle_joint_{receiver}")
        if oracle is not None:
            frozen = summaries[f"frozen_writer_retrained_{receiver}"]
            oracle_gains[receiver] = oracle["canonical_paired_consistency"] - frozen["canonical_paired_consistency"]
    report = {
        "status": "complete",
        "writer_sha256": next(iter(mother_hashes)),
        "rows": rows,
        "oracle_paired_gains": oracle_gains,
        "success_criteria": {
            "qwen3_4b_paired_ge_0_85": frozen4 >= 0.85,
            "qwen3_5_4b_paired_ge_0_70": frozen35 >= 0.70,
            "same_writer_for_both_receivers": len(mother_hashes) == 1,
            "overall_pass": frozen4 >= 0.85 and frozen35 >= 0.70,
        },
        "interpretation_scope": (
            "This tests one Qwen3-8B Writer with two receiver-specific Readers. "
            "It does not establish multi-sender composition or question-independent memory."
        ),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
