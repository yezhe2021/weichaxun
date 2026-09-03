from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    answer_f1, append_jsonl, clone_cache, forward_logits, generate, load_json,
    load_model, load_tokenizer, normalize_answer, nll, prefill, progress,
    read_jsonl, selected_samples, target_ids, zero_components,
)


def closest_shuffle(samples):
    mapping = {}
    for sample in samples:
        choices = [x for x in samples if x["id"] != sample["id"]
                   and normalize_answer(x["answer"]) != normalize_answer(sample["answer"])]
        mapping[sample["id"]] = min(
            choices, key=lambda x: abs(len(x["context_input_ids"]) - len(sample["context_input_ids"])))
    return mapping


def conditioned_cache(base_cache, config, zero_fa=False, zero_recurrent=False, zero_conv=False):
    """Fresh clone + same-shape zeroing, per protocol section 7. Never reuses a
    mutated cache: the forward pass updates conv/recurrent in place, so every
    logits call and every generation must start from its own clone."""
    cache = clone_cache(base_cache, config)
    zero_components(cache, config, zero_fa=zero_fa, zero_recurrent=zero_recurrent, zero_conv=zero_conv)
    return cache


def condition_row(sample, name, logits, gen, ids, gold_nll_target):
    return {
        "sample_id": sample["id"], "type": sample["type"], "condition": name,
        "prediction": gen, "em": float(normalize_answer(gen) == normalize_answer(sample["answer"])),
        "f1": answer_f1(gen, sample["answer"]), "nll": nll(logits, gold_nll_target),
        "output_tokens": len(ids),
    }


def process(model, tok, cfg, sample, donors):
    prefix_ids = sample["context_input_ids"]
    question_ids = sample["question_input_ids"]
    full_ids = sample["input_ids"]
    q_only_ids = sample["question_only_input_ids"]
    prefix_len = len(prefix_ids)
    target = target_ids(tok, sample["answer"], cfg["max_answer_tokens"])

    base_cache = prefill(model, prefix_ids)
    rows = []

    # continuous (no cache)
    logits = forward_logits(model, full_ids, target)
    gen, ids = generate(model, tok, full_ids, cfg)
    rows.append(condition_row(sample, "continuous", logits, gen, ids, target))

    # full hybrid replay
    logits = forward_logits(model, question_ids, target, prefix=prefix_len,
                            cache=conditioned_cache(base_cache, model.config))
    gen, ids = generate(model, tok, question_ids, cfg, prefix=prefix_len,
                        cache=conditioned_cache(base_cache, model.config))
    rows.append(condition_row(sample, "full_hybrid_replay", logits, gen, ids, target))

    # FA only: keep FA KV, drop recurrent + conv
    logits = forward_logits(model, question_ids, target, prefix=prefix_len,
                            cache=conditioned_cache(base_cache, model.config,
                                                    zero_recurrent=True, zero_conv=True))
    gen, ids = generate(model, tok, question_ids, cfg, prefix=prefix_len,
                        cache=conditioned_cache(base_cache, model.config,
                                                zero_recurrent=True, zero_conv=True))
    rows.append(condition_row(sample, "fa_only", logits, gen, ids, target))

    # FA + Recurrent: keep FA KV + recurrent, drop conv
    logits = forward_logits(model, question_ids, target, prefix=prefix_len,
                            cache=conditioned_cache(base_cache, model.config, zero_conv=True))
    gen, ids = generate(model, tok, question_ids, cfg, prefix=prefix_len,
                        cache=conditioned_cache(base_cache, model.config, zero_conv=True))
    rows.append(condition_row(sample, "fa_recurrent", logits, gen, ids, target))

    # question only (no prefix cache)
    logits = forward_logits(model, q_only_ids, target)
    gen, ids = generate(model, tok, q_only_ids, cfg)
    rows.append(condition_row(sample, "question_only", logits, gen, ids, target))

    # shuffled hybrid: wrong-context prefix cache + original question
    donor = donors[sample["id"]]
    donor_cache = prefill(model, donor["context_input_ids"])
    donor_prefix = len(donor["context_input_ids"])
    logits = forward_logits(model, question_ids, target, prefix=donor_prefix,
                            cache=conditioned_cache(donor_cache, model.config))
    gen, ids = generate(model, tok, question_ids, cfg, prefix=donor_prefix,
                        cache=conditioned_cache(donor_cache, model.config))
    row = condition_row(sample, "shuffled", logits, gen, ids, target)
    row["donor_id"] = donor["id"]
    row["donor_prefix_len"] = donor_prefix
    rows.append(row)

    return {"sample_id": sample["id"], "type": sample["type"],
            "prefix_len": prefix_len, "question_len": len(question_ids),
            "conditions": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    model, tok = load_model(cfg), load_tokenizer(cfg)
    samples = selected_samples(cfg, args.mode)
    donors = closest_shuffle(samples)
    out_path = Path(cfg["work_dir"]) / "outputs" / args.mode / "ablation_samples.jsonl"
    done = {r["sample_id"] for r in read_jsonl(out_path)}

    for index, sample in enumerate(samples, 1):
        if sample["id"] in done:
            progress(f"ablations: resume skip {index}/{len(samples)}")
            continue
        record = process(model, tok, cfg, sample, donors)
        append_jsonl(out_path, record)
        summary = {c["condition"]: round(c["f1"], 3) for c in record["conditions"]}
        progress(f"ablations {index}/{len(samples)} {sample['id']}: {summary}")


if __name__ == "__main__":
    main()
