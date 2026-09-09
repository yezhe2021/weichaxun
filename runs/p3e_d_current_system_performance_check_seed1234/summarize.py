import argparse
from pathlib import Path

from p3e_d_common import CONDITIONS, aggregate_mean, read_json, safe_ratio, write_json


def condition_metrics(rows, condition):
    selected = [row["conditions"][condition] for row in rows]
    result = {
        "n": len(selected),
        "em": aggregate_mean([item["em"] for item in selected]),
        "f1": aggregate_mean([item["f1"] for item in selected]),
        "eos_rate": aggregate_mean([item["eos_reached"] for item in selected]),
        "average_output_tokens": aggregate_mean([item["output_tokens"] for item in selected]),
        "average_input_tokens": aggregate_mean([item["input_tokens"] for item in selected]),
        "prediction_switch_from_question_only": aggregate_mean([item["switch_from_question_only"] for item in selected]),
        "input_truncation_count": sum(bool(item["truncated"]) for item in selected),
        "timing_ms": {},
        "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in selected),
        "average_incremental_peak_allocated_bytes": aggregate_mean([item["incremental_peak_allocated_bytes"] for item in selected]),
        "average_memory_runtime_bytes": aggregate_mean([item["memory_runtime_bytes"] for item in selected]),
        "by_type": {},
    }
    for stage in ("receiver_prefill", "receiver_decode_forward", "reader", "generation_wall"):
        result["timing_ms"][stage] = aggregate_mean([item["timing_ms"][stage] for item in selected])
    for kind in ("bridge", "comparison"):
        group = [row["conditions"][condition] for row in rows if row["type"] == kind]
        if group:
            result["by_type"][kind] = {
                "n": len(group),
                "em": aggregate_mean([item["em"] for item in group]),
                "f1": aggregate_mean([item["f1"] for item in group]),
            }
    if condition == "hard_shuffled_canonical_kv":
        result["source_answer_em"] = aggregate_mean([item["source_answer_em"] for item in selected])
        result["source_answer_f1"] = aggregate_mean([item["source_answer_f1"] for item in selected])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sender", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    evaluation = Path(args.evaluation)
    rows = []
    import json
    with (evaluation / "per_example.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    sender_rows = []
    with Path(args.sender).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sender_rows.append(json.loads(line))
    if len(rows) != manifest["sample_count"] or len(sender_rows) != manifest["sample_count"]:
        raise RuntimeError("Incomplete P3-E-D result set")

    metrics = {condition: condition_metrics(rows, condition) for condition in CONDITIONS}
    question = metrics["question_only"]["f1"]
    full = metrics["full_evidence_text"]["f1"]
    summary = metrics["sender_summary_text"]["f1"]
    native = metrics["native_headwise_kv"]["f1"]
    canonical = metrics["learned_canonical_kv"]["f1"]
    shuffled = metrics["hard_shuffled_canonical_kv"]["f1"]
    full_comparison = metrics["full_evidence_text"]["by_type"].get("comparison", {}).get("f1")
    question_comparison = metrics["question_only"]["by_type"].get("comparison", {}).get("f1")
    comparison_regression = None
    if full_comparison is not None and question_comparison is not None:
        comparison_regression = full_comparison - question_comparison
    text_conditions = ("full_evidence_text", "supporting_text", "sender_summary_text")
    truncations = sum(metrics[condition]["input_truncation_count"] for condition in text_conditions)
    low_eos = {condition: metrics[condition]["eos_rate"] for condition in text_conditions if metrics[condition]["eos_rate"] < 0.90}

    payload = {
        "status": "complete",
        "experiment": "P3-E-D Current-System Performance Check",
        "scope": "diagnostic 64-sample check; not a final paper conclusion",
        "samples": len(rows),
        "conditions": metrics,
        "comparisons": {
            "canonical_minus_full_text_f1": canonical - full,
            "canonical_minus_summary_text_f1": canonical - summary,
            "canonical_minus_native_f1": canonical - native,
            "correct_shuffled_f1_gap": canonical - shuffled,
            "communication_gain_retention": safe_ratio(canonical - question, full - question),
            "canonical_shuffled_prediction_switch": aggregate_mean([row["canonical_shuffled_prediction_switch"] for row in rows]),
            "reader_off_exact_question_only_consistency": aggregate_mean([row["question_only_reader_off_exact"] for row in rows]),
        },
        "communication": {
            "average_full_evidence_utf8_bytes": aggregate_mean([row["text_payloads"]["full_evidence_utf8_bytes"] for row in rows]),
            "average_supporting_text_utf8_bytes": aggregate_mean([row["text_payloads"]["supporting_text_utf8_bytes"] for row in rows]),
            "average_summary_utf8_bytes": aggregate_mean([row["text_payloads"]["summary_utf8_bytes"] for row in rows]),
            "average_summary_tokens": aggregate_mean([row["text_payloads"]["summary_output_tokens_sender"] for row in rows]),
            "summary_contains_gold_answer_rate": aggregate_mean([row["text_payloads"]["summary_contains_gold_answer"] for row in rows]),
            "average_native_cache_storage_bytes": aggregate_mean([row["native_cache_storage_bytes"] for row in sender_rows]),
            "average_native_runtime_bytes": aggregate_mean([row["native_runtime_bytes"] for row in sender_rows]),
            "average_canonical_runtime_bytes": aggregate_mean([row["canonical_runtime_bytes"] for row in sender_rows]),
            "canonical_dtype": sorted({row["canonical_dtype"] for row in sender_rows}),
        },
        "sender_timing_ms": {
            "evidence_prefill": aggregate_mean([row["sender_evidence_prefill_ms"] for row in sender_rows]),
            "writer": aggregate_mean([row["writer_ms"] for row in sender_rows]),
            "summary_generation": aggregate_mean([row["summary_generation_ms"] for row in sender_rows]),
        },
        "validity": {
            "text_input_truncation_count": truncations,
            "text_low_eos_conditions": low_eos,
            "full_text_comparison_f1_minus_question_only": comparison_regression,
            "comparison_text_regression_flag": comparison_regression is not None and comparison_regression < -0.05,
            "text_baseline_valid": truncations == 0 and not low_eos and not (comparison_regression is not None and comparison_regression < -0.05),
        },
        "manifest": manifest,
    }
    write_json(Path(args.out) / "summary.json", payload)
    write_json(Path(args.out) / "SUCCESS.json", payload)


if __name__ == "__main__":
    main()
