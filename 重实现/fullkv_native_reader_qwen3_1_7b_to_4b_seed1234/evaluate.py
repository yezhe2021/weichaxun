from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cache_store import cache_path, load_cache
from common import answer_f1, cuda, load_json, load_model, load_tokenizer, normalize_answer, progress, read_jsonl, save_json, seed_all, write_jsonl
from receiver import answer_logits, generate
from writers import load_writer, make_writer


def row(sample, condition, prediction, generated_ids, stop_reason, nll):
    return {
        "sample_id": sample["id"],
        "type": sample["type"],
        "context_length": sample["context_length"],
        "condition": condition,
        "question": sample["question"],
        "gold_answer": sample["answer"],
        "prediction": prediction,
        "generated_token_ids": generated_ids,
        "generation_stop_reason": stop_reason,
        "em": float(normalize_answer(prediction) == normalize_answer(sample["answer"])),
        "f1": answer_f1(prediction, sample["answer"]),
        "answer_nll": nll,
    }


def evaluate_condition(model, tokenizer, sample, cfg, condition, **kwargs):
    logits, gold = answer_logits(model, sample, **kwargs)
    prediction, ids, stop = generate(model, tokenizer, sample, cfg, **kwargs)
    return row(sample, condition, prediction, ids, stop, F.cross_entropy(logits, gold).item())


def exact_length_donors(samples):
    output = {}
    for sample in samples:
        choices = [other for other in samples if other["id"] != sample["id"] and other["context_length"] == sample["context_length"]]
        if choices:
            output[sample["id"]] = choices[0]
    return output


def aggregate(rows):
    conditions = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        conditions[condition] = {
            "count": len(selected),
            "em": sum(row["em"] for row in selected) / len(selected),
            "f1": sum(row["f1"] for row in selected) / len(selected),
            "answer_nll": sum(row["answer_nll"] for row in selected) / len(selected),
            "bridge_f1": sum(row["f1"] for row in selected if row["type"] == "bridge") / max(sum(row["type"] == "bridge" for row in selected), 1),
            "comparison_f1": sum(row["f1"] for row in selected if row["type"] == "comparison") / max(sum(row["type"] == "comparison" for row in selected), 1),
        }
    return conditions


def evaluate_4b(cfg, writer_kind, samples, checkpoint):
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    tokenizer = load_tokenizer(cfg["model_4b"])
    writer = make_writer(writer_kind, cfg).to(cuda()).eval()
    load_writer(str(checkpoint), writer)
    donors = exact_length_donors(samples)
    rows = []
    audit_limit = int(cfg["save_predicted_kv_audit_samples"])
    audit_root = Path(cfg["work_dir"]) / "artifacts" / "evaluation" / writer_kind / "predicted_kv_audit"
    try:
        for index, sample in enumerate(samples, 1):
            source = load_cache(cache_path(cfg, "source17", "test", sample["id"]), sample)
            target = load_cache(cache_path(cfg, "target4", "test", sample["id"]), sample)
            source_k = source["pre_key"].to(cuda())
            source_v = source["value"].to(cuda())
            target_k = target["pre_key"].to(cuda())
            target_v = target["value"].to(cuda())
            with torch.no_grad():
                predicted_k, predicted_v = writer(source_k, source_v)
            rows.append(evaluate_condition(model, tokenizer, sample, cfg, "4b_full_text"))
            rows.append(evaluate_condition(model, tokenizer, sample, cfg, "4b_native_full_kv", pre_key=target_k, value=target_v))
            rows.append(evaluate_condition(model, tokenizer, sample, cfg, "question_only", question_only=True))
            rows.append(evaluate_condition(model, tokenizer, sample, cfg, f"{writer_kind}_writer_correct", pre_key=predicted_k, value=predicted_v))

            zero_k, zero_v = torch.zeros_like(predicted_k), torch.zeros_like(predicted_v)
            rows.append(evaluate_condition(model, tokenizer, sample, cfg, f"{writer_kind}_writer_zero", pre_key=zero_k, value=zero_v))

            generator = torch.Generator(device="cpu").manual_seed(cfg["seed"] + index)
            permutation = torch.randperm(source_k.shape[1], generator=generator).to(cuda())
            with torch.no_grad():
                permuted_k, permuted_v = writer(source_k[:, permutation], source_v[:, permutation])
            rows.append(evaluate_condition(model, tokenizer, sample, cfg, f"{writer_kind}_writer_token_permuted", pre_key=permuted_k, value=permuted_v))

            if sample["id"] in donors:
                donor = donors[sample["id"]]
                donor_source = load_cache(cache_path(cfg, "source17", "test", donor["id"]), donor)
                with torch.no_grad():
                    donor_k, donor_v = writer(donor_source["pre_key"].to(cuda()), donor_source["value"].to(cuda()))
                rows.append(evaluate_condition(model, tokenizer, sample, cfg, f"{writer_kind}_writer_cross_sample_same_length", pre_key=donor_k, value=donor_v))

            if index <= audit_limit:
                audit_root.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "sample_id": sample["id"],
                    "native4_pre_key": target["pre_key"],
                    "native4_value": target["value"],
                    "writer_pre_key": predicted_k.detach().cpu(),
                    "writer_value": predicted_v.detach().cpu(),
                    "zero_pre_key": zero_k.detach().cpu(),
                    "zero_value": zero_v.detach().cpu(),
                    "token_permutation": permutation.cpu(),
                }, audit_root / f"{sample['id']}.pt")
            progress(f"4B evaluation {writer_kind}: {index}/{len(samples)}")
    finally:
        del writer, model
        torch.cuda.empty_cache()
    return rows


def evaluate_17_full_text(cfg, samples):
    model = load_model(cfg["model_1_7b"], cfg, frozen=True)
    tokenizer = load_tokenizer(cfg["model_1_7b"])
    rows = []
    try:
        for index, sample in enumerate(samples, 1):
            rows.append(evaluate_condition(model, tokenizer, sample, cfg, "1_7b_full_text"))
            progress(f"1.7B full-text evaluation: {index}/{len(samples)}")
    finally:
        del model
        torch.cuda.empty_cache()
    return rows


def run(cfg: dict[str, Any], writer_kind: str, checkpoint: str | None):
    samples = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "manifests" / "test.jsonl")
    checkpoint = Path(checkpoint) if checkpoint else Path(cfg["work_dir"]) / "checkpoints" / "quick" / writer_kind / "stage_b" / "best.pt"
    rows = evaluate_4b(cfg, writer_kind, samples, checkpoint)
    rows += evaluate_17_full_text(cfg, samples)
    conditions = aggregate(rows)
    question = conditions["question_only"]["f1"]
    native = conditions["4b_native_full_kv"]["f1"]
    writer = conditions[f"{writer_kind}_writer_correct"]["f1"]
    denominator = native - question
    summary = {
        "writer": writer_kind,
        "checkpoint": str(checkpoint),
        "conditions": conditions,
        "comparisons": {
            "writer_correct_minus_zero_f1": writer - conditions[f"{writer_kind}_writer_zero"]["f1"],
            "writer_correct_minus_token_permuted_f1": writer - conditions[f"{writer_kind}_writer_token_permuted"]["f1"],
            "native4_minus_writer_f1": native - writer,
            "retention_f1": (writer - question) / denominator if denominator > 0 else None,
            "retention_denominator": denominator,
        },
        "retention_defined_only_when_native_exceeds_question_only": True,
    }
    root = Path(cfg["work_dir"]) / "artifacts" / "evaluation" / writer_kind
    write_jsonl(root / "per_sample_generations.jsonl", rows)
    save_json(root / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--writer", choices=("d0", "d1", "d2"), required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    run(cfg, args.writer, args.checkpoint)


if __name__ == "__main__":
    main()

