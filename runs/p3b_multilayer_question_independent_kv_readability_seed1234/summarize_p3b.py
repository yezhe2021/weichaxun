import argparse
import csv
from pathlib import Path

from p3b_common import LAYER_SETS, SOURCES, SENDER_MODES, read_json, write_json


def get_condition(result, name):
    return next(row for row in result["conditions"] if row["condition"] == name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for mode in SENDER_MODES:
        for source in SOURCES:
            for layers in LAYER_SETS:
                path = root / "full" / mode / source / layers / "SUCCESS.json"
                result = read_json(path)
                correct = get_condition(result, "correct")
                shuffled = get_condition(result, "shuffled")
                zero = get_condition(result, "zero")
                mismatch = get_condition(result, "kv_mismatch")
                rows.append(
                    {
                        "sender_mode": mode,
                        "source": source,
                        "layer_set": layers,
                        "correct_em": correct["current_answer_em"],
                        "correct_f1": correct["current_answer_f1"],
                        "shuffled_current_em": shuffled["current_answer_em"],
                        "shuffled_source_em": shuffled["source_memory_em"],
                        "zero_em": zero["current_answer_em"],
                        "kv_mismatch_em": mismatch["current_answer_em"],
                        "start_accuracy": correct["start_accuracy"],
                        "end_accuracy": correct["end_accuracy"],
                        "supporting_fact_recall": correct["supporting_fact_recall"],
                        "correct_shuffled_gap": correct["current_answer_em"] - shuffled["current_answer_em"],
                    }
                )
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def row(mode, source, layers):
        return next(item for item in rows if item["sender_mode"] == mode and item["source"] == source and item["layer_set"] == layers)

    evidence_native_last = row("evidence_only", "native_kv", "last1")
    evidence_native_all = row("evidence_only", "native_kv", "all36")
    evidence_hidden_all = row("evidence_only", "hidden", "all36")
    conditioned_native_all = row("question_evidence", "native_kv", "all36")
    evidence_pca_all = row("evidence_only", "pca", "all36")
    evidence_trainable_all = row("evidence_only", "trainable", "all36")
    question_conditioning_gain = conditioned_native_all["correct_em"] - evidence_native_all["correct_em"]
    multilayer_gain = evidence_native_all["correct_em"] - evidence_native_last["correct_em"]
    verdicts = []
    if question_conditioning_gain >= 0.15:
        verdicts.append("question_conditioning_dominant")
    if multilayer_gain >= 0.10:
        verdicts.append("last_layer_is_a_major_bottleneck")
    if evidence_native_last["correct_em"] >= 0.70 and evidence_pca_all["correct_em"] + 0.15 < evidence_native_last["correct_em"]:
        verdicts.append("projection_information_loss")
    if min(evidence_native_all["correct_em"], evidence_hidden_all["correct_em"], evidence_pca_all["correct_em"]) >= 0.70:
        verdicts.append("use_uncompressed_multilayer_canonical_baseline")
    if evidence_trainable_all["correct_em"] >= 0.70 and max(evidence_native_all["correct_em"], evidence_pca_all["correct_em"]) < 0.70:
        verdicts.append("task_independent_trainable_layer_maps_required")
    if not verdicts:
        verdicts.append("no_positive_causal_readability_path")
    summary = {
        "status": "complete",
        "rows": rows,
        "curves": {
            mode: {
                source: {layers: row(mode, source, layers)["correct_em"] for layers in LAYER_SETS}
                for source in SOURCES
            }
            for mode in SENDER_MODES
        },
        "diagnostic_deltas": {
            "question_conditioning_gain_native_all36": question_conditioning_gain,
            "multilayer_gain_evidence_only_native": multilayer_gain,
        },
        "verdicts": verdicts,
        "gate": read_json(root / "gate" / "SUCCESS.json"),
    }
    write_json(root / "SUCCESS.json", summary)


if __name__ == "__main__":
    main()
