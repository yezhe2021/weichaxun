from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cache_store import cache_path, load_cache
from common import cuda, load_json, load_model, progress, read_jsonl, save_json, seed_all, write_jsonl
from receiver import choice_distribution, prediction, trajectory
from training import diagnose_writer, training_sample
from writers import load_writer, make_writer


@torch.no_grad()
def score(model, sample, condition, pre_key=None, value=None):
    output = trajectory(
        model, sample, condition=condition, pre_key=pre_key, value=value,
        output_hidden_states=False,
    )
    index, label = prediction(output.choice_logits)
    return {
        "prediction_index": index,
        "prediction_label": label,
        "choice_logits": output.choice_logits.float().cpu().tolist(),
        "choice_probabilities": choice_distribution(output.choice_logits).cpu().tolist(),
        "accuracy": float(index == sample["gold_index"]),
    }


def exact_length_donors(samples):
    output, by_length = {}, {}
    for sample in samples:
        by_length.setdefault(sample["context_length"], []).append(sample)
    for values in by_length.values():
        if len(values) > 1:
            for index, sample in enumerate(values):
                output[sample["id"]] = values[(index + 1) % len(values)]
    return output


def base_row(sample, condition, scored):
    return {
        "sample_id": sample["id"], "category": sample["category"],
        "question": sample["question"], "options": sample["options"],
        "num_options": sample["num_options"], "gold_index": sample["gold_index"],
        "gold_label": sample["gold_label"], "prefix_tokens": sample["context_length"],
        "suffix_tokens": sample["suffix_length"], "condition": condition, **scored,
    }


def add_native_comparison(rows, native_scored):
    native_logits = torch.tensor(native_scored["choice_logits"], dtype=torch.float32)
    native_log_prob = F.log_softmax(native_logits, dim=-1)
    for row in rows:
        logits = torch.tensor(row["choice_logits"], dtype=torch.float32)
        row["native_agreement"] = float(row["prediction_index"] == native_scored["prediction_index"])
        row["native_choice_logit_cosine"] = F.cosine_similarity(logits, native_logits, dim=0).item()
        row["native_to_condition_choice_kl"] = F.kl_div(
            F.log_softmax(logits, dim=-1), native_log_prob, reduction="sum", log_target=True
        ).item()


def load_checkpoint_writer(cfg, checkpoint):
    writer = make_writer("d2", cfg).to(cuda()).eval()
    payload = load_writer(str(checkpoint), writer)
    metadata = {key: value for key, value in payload.items() if key != "writer_state"}
    return writer, metadata


def evaluate_4b(cfg, samples, checkpoint_scope, cache_split):
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    root = Path(cfg["work_dir"]) / "checkpoints" / checkpoint_scope / "d2"
    checkpoint_paths = {
        "d2_stage_a": root / "stage_a/best.pt",
        "d2_b1_final_kl": root / "stage_b_final/best.pt",
        "d2_b2_all_token_kl": root / "stage_b_all/best.pt",
    }
    writers, metadata = {}, {}
    for name, path in checkpoint_paths.items():
        writers[name], metadata[name] = load_checkpoint_writer(cfg, path)
    donors, rows = exact_length_donors(samples), []
    try:
        for index, sample in enumerate(samples, 1):
            source = load_cache(cache_path(cfg, "source17", cache_split, sample["id"]), sample)
            target = load_cache(cache_path(cfg, "target4", cache_split, sample["id"]), sample)
            source_k, source_v = source["pre_key"].to(cuda()), source["value"].to(cuda())
            target_k, target_v = target["pre_key"].to(cuda()), target["value"].to(cuda())
            native = score(model, sample, "split_cache", target_k, target_v)
            sample_rows = [
                base_row(sample, "4b_standard_question_first", score(model, sample, "standard_full_text")),
                base_row(sample, "4b_options_first_full_text", score(model, sample, "options_first_full_text")),
                base_row(sample, "4b_native_full_kv", native),
                base_row(sample, "question_only", score(model, sample, "question_only")),
            ]
            predictions = {}
            for condition, writer in writers.items():
                predicted_k, predicted_v = writer(source_k, source_v)
                predictions[condition] = (predicted_k, predicted_v)
                sample_rows.append(base_row(
                    sample, condition + "_correct",
                    score(model, sample, "split_cache", predicted_k, predicted_v),
                ))
            for condition in ("d2_b1_final_kl", "d2_b2_all_token_kl"):
                predicted_k, predicted_v = predictions[condition]
                sample_rows.append(base_row(
                    sample, condition + "_zero",
                    score(model, sample, "split_cache", torch.zeros_like(predicted_k), torch.zeros_like(predicted_v)),
                ))
                if sample["id"] in donors:
                    donor = donors[sample["id"]]
                    donor_cache = load_cache(cache_path(cfg, "source17", cache_split, donor["id"]), donor)
                    donor_k, donor_v = writers[condition](
                        donor_cache["pre_key"].to(cuda()), donor_cache["value"].to(cuda())
                    )
                    shuffled = base_row(
                        sample, condition + "_shuffled_same_length",
                        score(model, sample, "split_cache", donor_k, donor_v),
                    )
                    shuffled["donor_sample_id"] = donor["id"]
                    shuffled["donor_prefix_tokens"] = donor["context_length"]
                    sample_rows.append(shuffled)
            add_native_comparison(sample_rows, native)
            rows.extend(sample_rows)
            progress(f"4B pure-functional evaluation: {index}/{len(samples)}")
        diagnostic_samples = [training_sample(sample) for sample in samples[:cfg["evaluation_diagnostic_samples"]]]
        diagnostics = {
            name: diagnose_writer(cfg, model, writer, diagnostic_samples, cache_split)
            for name, writer in writers.items()
        }
    finally:
        for writer in writers.values():
            del writer
        del model
        torch.cuda.empty_cache()
    return rows, diagnostics, metadata


def evaluate_17(cfg, samples):
    model = load_model(cfg["model_1_7b"], cfg, frozen=True)
    rows = []
    try:
        for index, sample in enumerate(samples, 1):
            for condition, receiver_condition in (
                ("1_7b_standard_question_first", "standard_full_text"),
                ("1_7b_options_first_full_text", "options_first_full_text"),
            ):
                rows.append(base_row(sample, condition, score(model, sample, receiver_condition)))
            progress(f"1.7B evaluation: {index}/{len(samples)}")
    finally:
        del model
        torch.cuda.empty_cache()
    return rows


def aggregate(rows):
    conditions = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        conditions[condition] = {
            "count": len(selected),
            "accuracy": sum(row["accuracy"] for row in selected) / len(selected),
            "native_agreement": (
                sum(row["native_agreement"] for row in selected) / len(selected)
                if "native_agreement" in selected[0] else None
            ),
            "native_choice_logit_cosine": (
                sum(row["native_choice_logit_cosine"] for row in selected) / len(selected)
                if "native_choice_logit_cosine" in selected[0] else None
            ),
            "native_to_condition_choice_kl": (
                sum(row["native_to_condition_choice_kl"] for row in selected) / len(selected)
                if "native_to_condition_choice_kl" in selected[0] else None
            ),
            "category_accuracy": {
                category: sum(row["accuracy"] for row in selected if row["category"] == category)
                / sum(row["category"] == category for row in selected)
                for category in sorted({row["category"] for row in selected})
            },
        }
    return conditions


def matched_control_comparison(rows, correct_condition, shuffled_condition):
    shuffled = [row for row in rows if row["condition"] == shuffled_condition]
    eligible_ids = {row["sample_id"] for row in shuffled}
    correct = [
        row for row in rows
        if row["condition"] == correct_condition and row["sample_id"] in eligible_ids
    ]
    if not shuffled:
        return {
            "matched_count": 0,
            "correct_accuracy": None, "shuffled_accuracy": None,
            "correct_minus_shuffled_accuracy": None,
            "correct_native_agreement": None, "shuffled_native_agreement": None,
            "correct_minus_shuffled_native_agreement": None,
            "correct_choice_kl": None, "shuffled_choice_kl": None,
        }
    if len(correct) != len(shuffled):
        raise RuntimeError(f"matched control size mismatch: {len(correct)} vs {len(shuffled)}")
    mean = lambda selected, key: sum(row[key] for row in selected) / len(selected)
    correct_accuracy, shuffled_accuracy = mean(correct, "accuracy"), mean(shuffled, "accuracy")
    correct_agreement = mean(correct, "native_agreement")
    shuffled_agreement = mean(shuffled, "native_agreement")
    return {
        "matched_count": len(shuffled),
        "correct_accuracy": correct_accuracy, "shuffled_accuracy": shuffled_accuracy,
        "correct_minus_shuffled_accuracy": correct_accuracy - shuffled_accuracy,
        "correct_native_agreement": correct_agreement,
        "shuffled_native_agreement": shuffled_agreement,
        "correct_minus_shuffled_native_agreement": correct_agreement - shuffled_agreement,
        "correct_choice_kl": mean(correct, "native_to_condition_choice_kl"),
        "shuffled_choice_kl": mean(shuffled, "native_to_condition_choice_kl"),
    }


def run(cfg: dict[str, Any], scope: str):
    checkpoint_scope = "overfit" if scope == "overfit" else "quick"
    cache_split = "train" if scope == "overfit" else "test"
    manifest_split = "train" if scope == "overfit" else "test"
    samples = read_jsonl(Path(cfg["work_dir"]) / f"artifacts/manifests/{manifest_split}.jsonl")
    if scope == "overfit":
        samples = samples[:cfg["overfit_samples"]]
    rows, diagnostics, metadata = evaluate_4b(cfg, samples, checkpoint_scope, cache_split)
    rows.extend(evaluate_17(cfg, samples))
    native_by_id = {row["sample_id"]: row for row in rows if row["condition"] == "4b_native_full_kv"}
    for row in rows:
        if "native_agreement" not in row:
            add_native_comparison([row], native_by_id[row["sample_id"]])
    conditions = aggregate(rows)
    comparisons = {}
    for prefix in ("d2_b1_final_kl", "d2_b2_all_token_kl"):
        correct, zero, shuffled = prefix + "_correct", prefix + "_zero", prefix + "_shuffled_same_length"
        matched = matched_control_comparison(rows, correct, shuffled)
        comparisons[prefix] = {
            "correct_minus_zero_accuracy": conditions[correct]["accuracy"] - conditions[zero]["accuracy"],
            "correct_minus_zero_native_agreement": conditions[correct]["native_agreement"] - conditions[zero]["native_agreement"],
            "matched_correct_vs_shuffled": matched,
            "minus_stage_a_accuracy": conditions[correct]["accuracy"] - conditions["d2_stage_a_correct"]["accuracy"],
            "minus_stage_a_native_agreement": conditions[correct]["native_agreement"] - conditions["d2_stage_a_correct"]["native_agreement"],
        }
    root = Path(cfg["work_dir"]) / "artifacts/evaluation" / scope
    write_jsonl(root / "per_sample_generations.jsonl", rows)
    save_json(root / "summary.json", {
        "conditions": conditions, "comparisons": comparisons,
        "scope": scope, "cache_split": cache_split,
        "checkpoint_diagnostics": diagnostics, "checkpoint_metadata": metadata,
        "diagnostic_sample_count": cfg["evaluation_diagnostic_samples"],
        "gold_label_used_only_for_evaluation": True,
        "shuffled_control_requires_exact_prefix_token_length": True,
        "mean_random_accuracy": sum(1.0 / sample["num_options"] for sample in samples) / len(samples),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scope", choices=("overfit", "formal"), default="formal")
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    run(cfg, args.scope)


if __name__ == "__main__":
    main()
