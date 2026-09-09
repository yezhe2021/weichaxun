from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common import cuda, load_json, load_model, progress, read_jsonl, save_json, seed_all, write_jsonl
from kv_protocol import apply_receiver_rope, capture_full_native, official_cache_tensors, validate_native_shapes
from receiver import prediction, trajectory


def tensor_comparison(reference: torch.Tensor, other: torch.Tensor):
    reference, other = reference.float(), other.float()
    return {
        "max_abs": (reference - other).abs().max().item(),
        "mean_abs": (reference - other).abs().mean().item(),
        "cosine": F.cosine_similarity(reference.flatten(), other.flatten(), dim=0).item(),
    }


def hidden_comparison(reference, other):
    layers = []
    for layer, (left, right) in enumerate(zip(reference, other)):
        row = tensor_comparison(left, right)
        row["layer"] = layer
        layers.append(row)
    return {
        "layers": layers,
        "max_abs": max(row["max_abs"] for row in layers),
        "min_cosine": min(row["cosine"] for row in layers),
    }


def run(cfg):
    samples = read_jsonl(Path(cfg["manifest_dir"]) / "test.jsonl")[:cfg["protocol_audit_samples"]]
    model = load_model(cfg[cfg["receiver_model_key"]], cfg, frozen=True)
    records = []
    try:
        for index, sample in enumerate(samples, 1):
            pre_key, native_v, official_cache, _ = capture_full_native(
                model, sample["prefix_token_ids"], cfg["target_layers"], cuda()
            )
            validate_native_shapes(pre_key, native_v, cfg["target_layers"], sample["context_length"], cfg["num_kv_heads"], cfg["head_dim"])
            pre_key, native_v = pre_key.to(cuda()), native_v.to(cuda())
            with torch.no_grad():
                full = trajectory(model, sample, condition="options_first_full_text")
                official = trajectory(model, sample, official_cache=official_cache)
                manual = trajectory(model, sample, pre_key=pre_key, value=native_v)
                standard = trajectory(model, sample, condition="standard_full_text", output_hidden_states=False)
            official_k, official_v = official_cache_tensors(official_cache)
            manual_post_k = apply_receiver_rope(model, pre_key).detach().cpu()
            full_index, full_label = prediction(full.choice_logits)
            official_index, official_label = prediction(official.choice_logits)
            manual_index, manual_label = prediction(manual.choice_logits)
            standard_index, standard_label = prediction(standard.choice_logits)
            records.append({
                "sample_id": sample["id"],
                "category": sample["category"],
                "prefix_tokens": sample["context_length"],
                "suffix_tokens": sample["suffix_length"],
                "gold_label": sample["gold_label"],
                "predictions": {
                    "standard_question_first": standard_label,
                    "options_first_full_text": full_label,
                    "official_native_full_kv": official_label,
                    "manual_native_full_kv": manual_label,
                },
                "correct": {
                    "standard_question_first": standard_index == sample["gold_index"],
                    "options_first_full_text": full_index == sample["gold_index"],
                },
                "full_vs_official_logits": tensor_comparison(full.logits, official.logits),
                "full_vs_manual_logits": tensor_comparison(full.logits, manual.logits),
                "official_vs_manual_logits": tensor_comparison(official.logits, manual.logits),
                "full_vs_official_choice_logits": tensor_comparison(full.choice_logits, official.choice_logits),
                "full_vs_manual_choice_logits": tensor_comparison(full.choice_logits, manual.choice_logits),
                "full_vs_official_hidden": hidden_comparison(full.hidden, official.hidden),
                "full_vs_manual_hidden": hidden_comparison(full.hidden, manual.hidden),
                "cache": {
                    "post_rope_k_max_abs": (official_k.float() - manual_post_k.float()).abs().max().item(),
                    "native_v_max_abs": (official_v.float() - native_v.detach().cpu().float()).abs().max().item(),
                },
                "choice_prediction_match": {
                    "official": official_index == full_index,
                    "manual": manual_index == full_index,
                },
            })
            progress(f"protocol audit {index}/{len(samples)}")
    finally:
        del model
        torch.cuda.empty_cache()

    tolerance = cfg["protocol_tolerances"]
    standard_accuracy = sum(row["correct"]["standard_question_first"] for row in records) / len(records)
    options_accuracy = sum(row["correct"]["options_first_full_text"] for row in records) / len(records)
    relative = options_accuracy / standard_accuracy if standard_accuracy > 0 else None
    options_gate = options_accuracy >= tolerance["options_first_relative_accuracy_min"] * standard_accuracy
    checks = {
        "full_official_logits_max_abs": max(row["full_vs_official_logits"]["max_abs"] for row in records) <= tolerance["split_forward_logits_max_abs"],
        "full_manual_logits_max_abs": max(row["full_vs_manual_logits"]["max_abs"] for row in records) <= tolerance["split_forward_logits_max_abs"],
        "full_official_logits_cosine": min(row["full_vs_official_logits"]["cosine"] for row in records) >= tolerance["split_forward_logits_cosine_min"],
        "full_manual_logits_cosine": min(row["full_vs_manual_logits"]["cosine"] for row in records) >= tolerance["split_forward_logits_cosine_min"],
        "full_official_hidden_cosine": min(row["full_vs_official_hidden"]["min_cosine"] for row in records) >= tolerance["split_forward_hidden_cosine_min"],
        "full_manual_hidden_cosine": min(row["full_vs_manual_hidden"]["min_cosine"] for row in records) >= tolerance["split_forward_hidden_cosine_min"],
        "official_choice_top1_match": sum(row["choice_prediction_match"]["official"] for row in records) / len(records) >= tolerance["split_forward_choice_top1_match_rate_min"],
        "manual_choice_top1_match": sum(row["choice_prediction_match"]["manual"] for row in records) / len(records) >= tolerance["split_forward_choice_top1_match_rate_min"],
        "official_manual_logits_exact": max(row["official_vs_manual_logits"]["max_abs"] for row in records) <= tolerance["official_manual_logits_max_abs"],
        "official_manual_post_rope_k_exact": max(row["cache"]["post_rope_k_max_abs"] for row in records) <= tolerance["official_manual_cache_max_abs"],
        "official_manual_native_v_exact": max(row["cache"]["native_v_max_abs"] for row in records) <= tolerance["official_manual_cache_max_abs"],
        "options_first_accuracy_retention": options_gate,
    }
    root = Path(cfg["work_dir"]) / "artifacts/protocol_audit"
    write_jsonl(root / "per_sample.jsonl", records)
    save_json(root / "summary.json", {
        "passed": all(checks.values()), "checks": checks, "tolerances": tolerance,
        "sample_count": len(records), "standard_question_first_accuracy": standard_accuracy,
        "options_first_accuracy": options_accuracy, "options_first_relative_accuracy": relative,
        "gold_used_only_for_protocol_feasibility_measurement": True,
    })
    if not all(checks.values()):
        raise RuntimeError(f"MMLU-Pro Full-KV protocol audit failed: {checks}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()
