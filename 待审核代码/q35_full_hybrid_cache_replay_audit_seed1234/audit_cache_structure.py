from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    cache_manifest, cache_tensors_equal, clone_cache, collect_cache_tensors,
    device, forward_logits, load_json, load_model, load_tokenizer, prefill,
    progress, save_json, selected_samples, target_ids, distribution,
)


def forward_logits_range(model, prompt_ids, prefix=0, cache=None):
    """Logits at every position of `prompt_ids` (used by the split diagnostic).

    Path continuous: prompt_ids = full ids, cache=None -> use_cache=False.
    Path replay: prompt_ids = suffix ids, cache = prefix cache -> use_cache=True.
    """
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device())
    mask = torch.ones(1, prefix + len(prompt_ids), dtype=torch.long, device=device())
    with torch.no_grad():
        output = model(
            input_ids=ids, attention_mask=mask,
            past_key_values=cache, use_cache=cache is not None,
        )
    return output.logits[0].float().cpu()


def split_remainder(model, full_ids, context_end, target, r):
    """Split full_ids at the largest position <= context_end with (k mod 64)==r,
    prefill [0:k), replay [k:), compare with continuous logits at [k:)."""
    k = context_end - ((context_end - r) % 64)
    if k < 64 or k >= len(full_ids) - 1:
        return {"remainder": r, "split": k, "skipped": True}
    prefix_ids, suffix_ids = full_ids[:k], full_ids[k:]
    cache = prefill(model, prefix_ids)
    logits_replay = forward_logits_range(model, suffix_ids, prefix=k, cache=cache)
    logits_cont = forward_logits_range(model, full_ids)
    logits_cont_slice = logits_cont[k:]
    m = distribution(logits_cont_slice, logits_replay, gold=None)
    return {
        "remainder": r,
        "split": k,
        "split_mod_64": k % 64,
        "prefix_len": k,
        "suffix_len": len(suffix_ids),
        "skipped": False,
        **m,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split-diag", action="store_true",
                        help="run Gate-2 arbitrary-split diagnostic (5 prefills)")
    args = parser.parse_args()

    cfg = load_json(args.config)
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    model, tok = load_model(cfg), load_tokenizer(cfg)
    root = Path(cfg["work_dir"])
    samples = selected_samples(cfg, "smoke")
    sample = samples[args.sample_index]

    if not sample.get("split_equal"):
        raise RuntimeError("prefix+suffix does not reconstruct full ids for sample")
    prefix_ids = sample["context_input_ids"]
    question_ids = sample["question_input_ids"]
    full_ids = sample["input_ids"]
    prefix_len = len(prefix_ids)
    target = target_ids(tok, sample["answer"], cfg["max_answer_tokens"])

    progress(f"sample {sample['id']}: prefix_len={prefix_len} question_len={len(question_ids)} "
             f"full_len={len(full_ids)} prefix_mod64={prefix_len % 64}")

    # ---- Smoke-1.a: manifest ----
    base_cache = prefill(model, prefix_ids)
    manifest = cache_manifest(base_cache, model.config, prefix_len)
    save_json(root / "cache_manifest.json", manifest)
    checks = {
        "manifest_seq_length": manifest["get_seq_length"] == prefix_len,
        "has_previous_state": manifest["has_previous_state"],
    }
    n_linear_initialized = sum(
        l["layer_type"] == "linear_attention" and "recurrent_shape" in l
        for l in manifest["layers"]
    )
    n_fa_initialized = sum(
        l["layer_type"] == "full_attention" and "key_shape" in l
        for l in manifest["layers"]
    )
    checks["n_linear_recurrent_initialized"] = n_linear_initialized == 24
    checks["n_fa_kv_initialized"] = n_fa_initialized == 8

    # ---- Smoke-1.b: Gate 1, cache-copy / re-injection ----
    snapshot_base = collect_cache_tensors(base_cache)
    cache_b = clone_cache(base_cache, model.config)
    cache_c = clone_cache(base_cache, model.config)
    clone_eq, clone_max_abs, clone_max_nmse = cache_tensors_equal(
        collect_cache_tensors(cache_b), snapshot_base)
    checks["clone_tensors_equal"] = clone_eq

    logits_B = forward_logits(model, question_ids, target, prefix=prefix_len, cache=cache_b)
    logits_C = forward_logits(model, question_ids, target, prefix=prefix_len, cache=cache_c)
    logits_A = forward_logits(model, full_ids, target)
    m_BvC = distribution(logits_B, logits_C, target)
    m_AvB = distribution(logits_A, logits_B, target)

    # after replay: B and C were mutated identically; base_cache must be untouched
    eq_BC_post, _, _ = cache_tensors_equal(collect_cache_tensors(cache_b),
                                            collect_cache_tensors(cache_c))
    eq_base_post, _, _ = cache_tensors_equal(snapshot_base, collect_cache_tensors(base_cache))
    checks["post_forward_B_equals_C"] = eq_BC_post
    checks["base_cache_unmutated"] = eq_base_post

    result = {
        "sample_id": sample["id"],
        "prefix_len": prefix_len,
        "question_len": len(question_ids),
        "clone_gate": {"tensors_equal": clone_eq, "max_abs": clone_max_abs,
                       "max_nmse": clone_max_nmse, **m_BvC},
        "continuity_gate": m_AvB,
        "manifest_checks": checks,
        "split_diag": [],
    }

    # ---- Smoke-1.c: Gate 2, arbitrary-split replay ----
    if args.split_diag:
        for r in cfg["split_remainders"]:
            result["split_diag"].append(split_remainder(model, full_ids, prefix_len, target, r))

    save_json(root / "metrics" / "smoke1_audit.json", result)
    progress(f"clone_gate max_abs={clone_max_abs:.3e} | continuity mean_kl={m_AvB['mean_kl']:.3e} "
             f"max_abs={m_AvB['logits_max_absolute_error']:.3e} | checks={checks}")
    if cfg["enforce_hard_gates"] and not all(v is True for v in checks.values()):
        raise RuntimeError(f"Smoke-1 structural checks failed: "
                           f"{[k for k, v in checks.items() if v is not True]}")


if __name__ == "__main__":
    main()
