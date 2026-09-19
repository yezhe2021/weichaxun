from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from common import answer_f1, cuda, load_json, load_model, load_tokenizer, normalize_answer, progress, read_jsonl, save_json, seed_all, write_jsonl
from kv_protocol import apply_receiver_rope, capture_full_native, official_cache_tensors, validate_native_shapes
from receiver import answer_logits, generate, suffix_continuation_logits


def comparison(reference: torch.Tensor, other: torch.Tensor, gold: torch.Tensor):
    return {
        "logits_max_abs": (reference - other).abs().max().item(),
        "logits_mean_abs": (reference - other).abs().mean().item(),
        "logits_cosine": F.cosine_similarity(reference.flatten(), other.flatten(), dim=0).item(),
        "top1_match_rate": (reference.argmax(-1) == other.argmax(-1)).float().mean().item(),
        "answer_nll_reference": F.cross_entropy(reference, gold).item(),
        "answer_nll_other": F.cross_entropy(other, gold).item(),
        "answer_nll_abs_diff": abs(F.cross_entropy(reference, gold).item() - F.cross_entropy(other, gold).item()),
    }


def generation_row(sample, condition, prediction, token_ids, stop_reason, nll):
    return {
        "sample_id": sample["id"],
        "type": sample["type"],
        "condition": condition,
        "question": sample["question"],
        "gold_answer": sample["answer"],
        "prediction": prediction,
        "generated_token_ids": token_ids,
        "generation_stop_reason": stop_reason,
        "em": float(normalize_answer(prediction) == normalize_answer(sample["answer"])),
        "f1": answer_f1(prediction, sample["answer"]),
        "answer_nll": nll,
    }


def run(cfg):
    samples = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "manifests" / "test.jsonl")
    samples = samples[: cfg["protocol_audit_samples"]]
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    tokenizer = load_tokenizer(cfg["model_4b"])
    records, generations = [], []
    try:
        for index, sample in enumerate(samples, 1):
            pre_key, native_v, official_cache, official_first_suffix_logits = capture_full_native(
                model, sample["prefix_token_ids"], cfg["target_layers"], cuda()
            )
            validate_native_shapes(pre_key, native_v, cfg["target_layers"], sample["context_length"], cfg["num_kv_heads"], cfg["head_dim"])
            pre_key = pre_key.to(cuda())
            native_v = native_v.to(cuda())

            text_logits, gold = answer_logits(model, sample)
            official_logits, _ = answer_logits(model, sample, official_cache=copy.deepcopy(official_cache))
            manual_logits, _ = answer_logits(model, sample, pre_key=pre_key, value=native_v)
            text_suffix = suffix_continuation_logits(model, sample)
            official_suffix = suffix_continuation_logits(model, sample, cache=copy.deepcopy(official_cache))
            manual_suffix = suffix_continuation_logits(model, sample, pre_key=pre_key, value=native_v)

            full_ids = torch.tensor([sample["full_prompt_ids"]], dtype=torch.long, device=cuda())
            full_positions = torch.arange(full_ids.shape[1], device=cuda()).unsqueeze(0)
            with torch.no_grad():
                full_output = model(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    position_ids=full_positions,
                    use_cache=False,
                )
            # The logit at prefix_length-1 predicts the first later Question token.
            full_first_suffix_logits = full_output.logits[0, sample["context_length"] - 1].float().cpu()

            official_k, official_v = official_cache_tensors(official_cache)
            manual_post_k = apply_receiver_rope(model, pre_key).detach().cpu()
            cache_comparison = {
                "post_rope_k_max_abs": (official_k.float() - manual_post_k.float()).abs().max().item(),
                "native_v_max_abs": (official_v.float() - native_v.detach().cpu().float()).abs().max().item(),
                "post_rope_k_cosine": F.cosine_similarity(official_k.double().flatten(), manual_post_k.double().flatten(), dim=0).item(),
                "native_v_cosine": F.cosine_similarity(official_v.double().flatten(), native_v.detach().cpu().double().flatten(), dim=0).item(),
            }
            record = {
                "sample_id": sample["id"],
                "context_length": sample["context_length"],
                "full_text_vs_official": comparison(text_logits, official_logits, gold),
                "full_text_vs_manual": comparison(text_logits, manual_logits, gold),
                "official_vs_manual": comparison(official_logits, manual_logits, gold),
                "suffix_after_first_token_vs_official": {
                    "max_abs": (text_suffix - official_suffix).abs().max().item(),
                    "cosine": F.cosine_similarity(text_suffix.flatten(), official_suffix.flatten(), dim=0).item(),
                },
                "suffix_after_first_token_vs_manual": {
                    "max_abs": (text_suffix - manual_suffix).abs().max().item(),
                    "cosine": F.cosine_similarity(text_suffix.flatten(), manual_suffix.flatten(), dim=0).item(),
                },
                "full_text_first_suffix_token_vs_official_prefill": {
                    "max_abs": (full_first_suffix_logits - official_first_suffix_logits[0]).abs().max().item(),
                    "cosine": F.cosine_similarity(full_first_suffix_logits, official_first_suffix_logits[0], dim=0).item(),
                },
                "cache": cache_comparison,
            }
            records.append(record)
            for condition, kwargs, logits in (
                ("4b_full_text", {}, text_logits),
                ("4b_official_native_full_kv", {"official_cache": official_cache}, official_logits),
                ("4b_manual_native_full_kv", {"pre_key": pre_key, "value": native_v}, manual_logits),
            ):
                prediction, ids, stop = generate(model, tokenizer, sample, cfg, **kwargs)
                generations.append(generation_row(sample, condition, prediction, ids, stop, F.cross_entropy(logits, gold).item()))
            progress(f"protocol audit {index}/{len(samples)}")
    finally:
        del model
        torch.cuda.empty_cache()

    tolerance = cfg["protocol_tolerances"]
    checks = {
        "full_text_official_split_logits_max_abs": max(row["full_text_vs_official"]["logits_max_abs"] for row in records) <= tolerance["split_forward_logits_max_abs"],
        "full_text_manual_split_logits_max_abs": max(row["full_text_vs_manual"]["logits_max_abs"] for row in records) <= tolerance["split_forward_logits_max_abs"],
        "full_text_official_split_logits_cosine": min(row["full_text_vs_official"]["logits_cosine"] for row in records) >= tolerance["split_forward_logits_cosine_min"],
        "full_text_manual_split_logits_cosine": min(row["full_text_vs_manual"]["logits_cosine"] for row in records) >= tolerance["split_forward_logits_cosine_min"],
        "full_text_official_split_top1": min(row["full_text_vs_official"]["top1_match_rate"] for row in records) >= tolerance["split_forward_top1_match_rate_min"],
        "full_text_manual_split_top1": min(row["full_text_vs_manual"]["top1_match_rate"] for row in records) >= tolerance["split_forward_top1_match_rate_min"],
        "full_text_official_split_answer_nll": max(row["full_text_vs_official"]["answer_nll_abs_diff"] for row in records) <= tolerance["split_forward_answer_nll_abs_diff_max"],
        "full_text_manual_split_answer_nll": max(row["full_text_vs_manual"]["answer_nll_abs_diff"] for row in records) <= tolerance["split_forward_answer_nll_abs_diff_max"],
        "official_manual_logits_exact": max(row["official_vs_manual"]["logits_max_abs"] for row in records) <= tolerance["official_manual_logits_max_abs"],
        "official_manual_answer_nll_exact": max(row["official_vs_manual"]["answer_nll_abs_diff"] for row in records) <= tolerance["official_manual_answer_nll_abs_diff_max"],
        "official_manual_post_rope_k_exact": max(row["cache"]["post_rope_k_max_abs"] for row in records) <= tolerance["official_manual_cache_max_abs"],
        "official_manual_native_v_exact": max(row["cache"]["native_v_max_abs"] for row in records) <= tolerance["official_manual_cache_max_abs"],
    }
    if tolerance["greedy_match_required"]:
        text = {row["sample_id"]: row["generated_token_ids"] for row in generations if row["condition"] == "4b_full_text"}
        checks["official_generation_match"] = all(row["generated_token_ids"] == text[row["sample_id"]] for row in generations if row["condition"] == "4b_official_native_full_kv")
        checks["manual_generation_match"] = all(row["generated_token_ids"] == text[row["sample_id"]] for row in generations if row["condition"] == "4b_manual_native_full_kv")
    root = Path(cfg["work_dir"]) / "artifacts" / "protocol_audit"
    write_jsonl(root / "per_sample.jsonl", records)
    write_jsonl(root / "generations.jsonl", generations)
    save_json(root / "summary.json", {"passed": all(checks.values()), "checks": checks, "tolerances": tolerance, "sample_count": len(samples)})
    if not all(checks.values()):
        raise RuntimeError(f"Full-cache protocol audit failed: {checks}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()
