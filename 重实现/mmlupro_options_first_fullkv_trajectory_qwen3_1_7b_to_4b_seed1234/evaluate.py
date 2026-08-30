from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cache_store import cache_path, load_cache
from common import cuda, load_json, load_model, progress, read_jsonl, save_json, seed_all, write_jsonl
from receiver import choice_distribution, prediction, trajectory
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
    output = {}
    by_length = {}
    for sample in samples:
        by_length.setdefault(sample["context_length"], []).append(sample)
    for values in by_length.values():
        if len(values) > 1:
            for index, sample in enumerate(values):
                output[sample["id"]] = values[(index + 1) % len(values)]
    return output


def base_row(sample, condition, scored):
    return {
        "sample_id": sample["id"],
        "category": sample["category"],
        "question": sample["question"],
        "options": sample["options"],
        "num_options": sample["num_options"],
        "gold_index": sample["gold_index"],
        "gold_label": sample["gold_label"],
        "prefix_tokens": sample["context_length"],
        "suffix_tokens": sample["suffix_length"],
        "condition": condition,
        **scored,
    }


def add_native_comparison(rows, native_scored):
    native_logits = torch.tensor(native_scored["choice_logits"], dtype=torch.float32)
    native_log_prob = F.log_softmax(native_logits, dim=-1)
    native_prob = native_log_prob.exp()
    for row in rows:
        logits = torch.tensor(row["choice_logits"], dtype=torch.float32)
        row["native_agreement"] = float(row["prediction_index"] == native_scored["prediction_index"])
        row["native_choice_logit_cosine"] = F.cosine_similarity(logits, native_logits, dim=0).item()
        row["native_to_condition_choice_kl"] = F.kl_div(
            F.log_softmax(logits, dim=-1), native_log_prob, reduction="sum", log_target=True
        ).item()


def load_stage_writer(cfg, writer_kind, checkpoint):
    writer = make_writer(writer_kind, cfg).to(cuda()).eval()
    load_writer(str(checkpoint), writer)
    return writer


def evaluate_4b(cfg, writer_kind, samples, stage):
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    root = Path(cfg["work_dir"]) / "checkpoints/quick" / writer_kind
    writers = {}
    if stage in {"a", "both"}:
        writers["stage_a"] = load_stage_writer(cfg, writer_kind, root / "stage_a/best.pt")
    if stage in {"b", "both"}:
        writers["stage_b"] = load_stage_writer(cfg, writer_kind, root / "stage_b/best.pt")
    donors, rows = exact_length_donors(samples), []
    try:
        for index, sample in enumerate(samples, 1):
            source = load_cache(cache_path(cfg, "source17", "test", sample["id"]), sample)
            target = load_cache(cache_path(cfg, "target4", "test", sample["id"]), sample)
            source_k, source_v = source["pre_key"].to(cuda()), source["value"].to(cuda())
            target_k, target_v = target["pre_key"].to(cuda()), target["value"].to(cuda())
            sample_rows = []
            native = score(model, sample, "split_cache", target_k, target_v)
            for condition, scored in (
                ("4b_standard_question_first", score(model, sample, "standard_full_text")),
                ("4b_options_first_full_text", score(model, sample, "options_first_full_text")),
                ("4b_native_full_kv", native),
                ("question_only", score(model, sample, "question_only")),
            ):
                sample_rows.append(base_row(sample, condition, scored))
            predictions = {}
            for stage_name, writer in writers.items():
                predicted_k, predicted_v = writer(source_k, source_v)
                predictions[stage_name] = (predicted_k, predicted_v)
                sample_rows.append(base_row(
                    sample, f"{writer_kind}_{stage_name}_correct",
                    score(model, sample, "split_cache", predicted_k, predicted_v),
                ))
            if "stage_b" in predictions:
                predicted_k, predicted_v = predictions["stage_b"]
                sample_rows.append(base_row(
                    sample, f"{writer_kind}_stage_b_zero",
                    score(model, sample, "split_cache", torch.zeros_like(predicted_k), torch.zeros_like(predicted_v)),
                ))
                if sample["id"] in donors:
                    donor = donors[sample["id"]]
                    donor_cache = load_cache(cache_path(cfg, "source17", "test", donor["id"]), donor)
                    donor_k, donor_v = writers["stage_b"](
                        donor_cache["pre_key"].to(cuda()), donor_cache["value"].to(cuda())
                    )
                    shuffled = base_row(
                        sample, f"{writer_kind}_stage_b_shuffled_same_length",
                        score(model, sample, "split_cache", donor_k, donor_v),
                    )
                    shuffled["donor_sample_id"] = donor["id"]
                    shuffled["donor_prefix_tokens"] = donor["context_length"]
                    sample_rows.append(shuffled)
            add_native_comparison(sample_rows, native)
            rows.extend(sample_rows)
            progress(f"4B evaluation {writer_kind}: {index}/{len(samples)}")
    finally:
        for writer in writers.values():
            del writer
        del model
        torch.cuda.empty_cache()
    return rows


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


def run(cfg: dict[str, Any], writer_kind: str, stage: str):
    samples = read_jsonl(Path(cfg["work_dir"]) / "artifacts/manifests/test.jsonl")
    rows = evaluate_4b(cfg, writer_kind, samples, stage)
    rows.extend(evaluate_17(cfg, samples))
    native_by_id = {
        row["sample_id"]: row for row in rows if row["condition"] == "4b_native_full_kv"
    }
    for row in rows:
        if "native_agreement" not in row:
            add_native_comparison([row], native_by_id[row["sample_id"]])
    conditions = aggregate(rows)
    comparisons = {}
    correct = f"{writer_kind}_stage_b_correct"
    zero = f"{writer_kind}_stage_b_zero"
    shuffled = f"{writer_kind}_stage_b_shuffled_same_length"
    if correct in conditions and zero in conditions:
        comparisons["stage_b_correct_minus_zero_accuracy"] = conditions[correct]["accuracy"] - conditions[zero]["accuracy"]
        comparisons["stage_b_correct_minus_zero_native_agreement"] = conditions[correct]["native_agreement"] - conditions[zero]["native_agreement"]
    if correct in conditions and shuffled in conditions:
        comparisons["stage_b_correct_minus_shuffled_accuracy"] = conditions[correct]["accuracy"] - conditions[shuffled]["accuracy"]
        comparisons["stage_b_correct_minus_shuffled_native_agreement"] = conditions[correct]["native_agreement"] - conditions[shuffled]["native_agreement"]
    stage_a = f"{writer_kind}_stage_a_correct"
    if correct in conditions and stage_a in conditions:
        comparisons["stage_b_minus_stage_a_accuracy"] = conditions[correct]["accuracy"] - conditions[stage_a]["accuracy"]
        comparisons["stage_b_minus_stage_a_native_agreement"] = conditions[correct]["native_agreement"] - conditions[stage_a]["native_agreement"]
    root = Path(cfg["work_dir"]) / "artifacts/evaluation" / writer_kind
    write_jsonl(root / "per_sample_generations.jsonl", rows)
    save_json(root / "summary.json", {
        "writer": writer_kind, "evaluated_stage": stage,
        "conditions": conditions, "comparisons": comparisons,
        "gold_label_used_only_for_evaluation": True,
        "shuffled_control_requires_exact_prefix_token_length": True,
        "mean_random_accuracy": sum(1.0 / sample["num_options"] for sample in samples) / len(samples),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--writer", choices=("d0", "d1", "d2"), required=True)
    parser.add_argument("--stage", choices=("a", "b", "both"), default="both")
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    run(cfg, args.writer, args.stage)


if __name__ == "__main__":
    main()
