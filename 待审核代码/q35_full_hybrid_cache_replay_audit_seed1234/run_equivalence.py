from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import DynamicCache

from common import (
    answer_f1, append_jsonl, cache_tensors_equal, clone_cache, collect_cache_tensors,
    distribution, forward_logits, generate, load_json, load_model, load_tokenizer,
    normalize_answer, nll, prefill, progress, read_jsonl, selected_samples, target_ids,
)


def generation_row(condition, sample, prediction, ids, nll_value):
    return {
        "sample_id": sample["id"], "type": sample["type"], "condition": condition,
        "answer": sample["answer"], "prediction": prediction,
        "em": float(normalize_answer(prediction) == normalize_answer(sample["answer"])),
        "f1": answer_f1(prediction, sample["answer"]), "nll": nll_value,
        "output_tokens": len(ids),
    }


def process(model, tok, cfg, sample):
    prefix_ids = sample["context_input_ids"]
    question_ids = sample["question_input_ids"]
    full_ids = sample["input_ids"]
    prefix_len = len(prefix_ids)
    target = target_ids(tok, sample["answer"], cfg["max_answer_tokens"])

    base_cache = prefill(model, prefix_ids)
    snapshot_base = collect_cache_tensors(base_cache)

    # independent clones, BEFORE any forward
    cache_b = clone_cache(base_cache, model.config)
    cache_c = clone_cache(base_cache, model.config)
    _, clone_b_max_abs, clone_b_nmse = cache_tensors_equal(
        collect_cache_tensors(cache_b), snapshot_base)
    _, clone_c_max_abs, clone_c_nmse = cache_tensors_equal(
        collect_cache_tensors(cache_c), snapshot_base)
    clone_max_abs = max(clone_b_max_abs, clone_c_max_abs)
    clone_nmse = max(clone_b_nmse, clone_c_nmse)

    # Path A0: continuous, no cache. Path A1: continuous, empty DynamicCache.
    logits_A0 = forward_logits(model, full_ids, target)
    logits_A1 = forward_logits(model, full_ids, target, cache=DynamicCache(config=model.config))

    # Path B/C: replay from the two clones
    logits_B = forward_logits(model, question_ids, target, prefix=prefix_len, cache=cache_b)
    logits_C = forward_logits(model, question_ids, target, prefix=prefix_len, cache=cache_c)

    # attribution chain:
    #   1. A0 vs A1  -> enabling the cache does not change the output
    #   2. B vs C    -> clone / re-injection is lossless
    #   3. A1 vs B   -> arbitrary-split replay equals continuous
    m_A0A1 = distribution(logits_A0, logits_A1, target)
    m_BvC = distribution(logits_B, logits_C, target)
    m_A1B = distribution(logits_A1, logits_B, target)
    m_A0B = distribution(logits_A0, logits_B, target)

    # after forward: B and C must have been updated identically, base untouched
    _, post_BC_max_abs, post_BC_nmse = cache_tensors_equal(
        collect_cache_tensors(cache_b), collect_cache_tensors(cache_c))
    _, base_max_abs, base_nmse = cache_tensors_equal(
        snapshot_base, collect_cache_tensors(base_cache))

    # generations (decode through the fused per-step kernel on both paths)
    gen_continuous, ids_cont = generate(model, tok, full_ids, cfg)
    gen_replay, ids_replay = generate(model, tok, question_ids, cfg,
                                      prefix=prefix_len, cache=clone_cache(base_cache, model.config))
    match = gen_continuous == gen_replay

    return {
        "sample_id": sample["id"], "type": sample["type"],
        "prefix_len": prefix_len, "prefix_mod64": prefix_len % 64,
        "question_len": len(question_ids),
        "clone_gate": {
            "clone_pre_forward_max_abs": clone_max_abs,
            "clone_pre_forward_nmse": clone_nmse,
            "post_forward_B_vs_C_max_abs": post_BC_max_abs,
            "post_forward_B_vs_C_nmse": post_BC_nmse,
            "base_cache_unmutated_max_abs": base_max_abs,
            "base_cache_unmutated_nmse": base_nmse,
            **m_BvC,
        },
        "continuity": {
            "a0_vs_a1": m_A0A1,
            "a1_vs_b": m_A1B,
            "a0_vs_b": m_A0B,
        },
        "generation": [
            generation_row("continuous", sample, gen_continuous, ids_cont, nll(logits_A0, target)),
            generation_row("full_hybrid_replay", sample, gen_replay, ids_replay, nll(logits_B, target)),
            {"sample_id": sample["id"], "exact_generation_match": match,
             "continuous": gen_continuous, "replay": gen_replay},
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    model, tok = load_model(cfg), load_tokenizer(cfg)
    samples = selected_samples(cfg, args.mode)
    out_path = Path(cfg["work_dir"]) / "outputs" / args.mode / "equivalence_samples.jsonl"
    done = {r["sample_id"] for r in read_jsonl(out_path)}

    for index, sample in enumerate(samples, 1):
        if sample["id"] in done:
            progress(f"equivalence: resume skip {index}/{len(samples)}")
            continue
        record = process(model, tok, cfg, sample)
        append_jsonl(out_path, record)
        progress(f"equivalence {index}/{len(samples)} {sample['id']}: "
                 f"clone_pre_nmse={record['clone_gate']['clone_pre_forward_nmse']:.3e} "
                 f"A1B_kl={record['continuity']['a1_vs_b']['mean_kl']:.3e} "
                 f"A0A1_kl={record['continuity']['a0_vs_a1']['mean_kl']:.3e} "
                 f"gen_match={record['generation'][-1]['exact_generation_match']}")


if __name__ == "__main__":
    main()
