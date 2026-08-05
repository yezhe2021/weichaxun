"""阶段1：四个模型自身能力基线 + Cache Gate（方案 §五）。

对每个模型 M 独立运行：
  1. Question only              → F1_M^QOnly
  2. Full-context text          → F1_M^FullText（完整 10 段 Context + Question）
  3. Official Native Cache      → 官方 cache 机制前向（等价性验证用）
  4. Manual pre-RoPE Cache      → pre-RoPE 提取 + 重新注入（等价性验证用）
  5. Shuffled Native Cache      → 另一条样本的 Context KV（donor），Question 不变 → F1_M^Shuffled

Cache Gate（必须达到）：FullText ≈ Official ≈ Manual
  自由生成一致率 100%；answer-token top-1 ≥ 99.5%；mean logit KL < 1e-3。

输出 F1_M^QOnly / FullText / Shuffled 以及 SelfGain_M = F1_M^FullText − F1_M^QOnly（方案 §五）。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from protocol import (
    answer_logits, capture_native, cuda, dynamic_cache, generate, load_json, load_model,
    progress, save_json, token_f1, tokenizer, write_jsonl, seed_all,
)


def manual_kv(model, sample):
    """提取样本自身的 context-only pre-RoPE KV（方案 §二）。"""
    return capture_native(model, sample["context_input_ids"], model.config.num_hidden_layers)


@torch.no_grad()
def logit_kl(native, test):
    n = min(native.shape[0], test.shape[0])
    native, test = native[:n].float(), test[:n].float()
    log_test = F.log_softmax(test, dim=-1)
    p_native = F.softmax(native, dim=-1)
    return max(0.0, F.kl_div(log_test, p_native, reduction="batchmean").item() / n)


@torch.no_grad()
def run_cache_gate(model, sample):
    """比较 FullText / Official Native / Manual pre-RoPE 三种路径的 answer-token logits。"""
    answer = sample["answer_token_ids"]
    n = len(answer) - 1
    # 1) FullText：非 cache 完整前向（基准）
    current = sample["full_input_ids"] + answer[:-1]
    ids = torch.tensor([current], dtype=torch.long, device=cuda())
    positions = torch.arange(len(current), device=cuda()).unsqueeze(0)
    mask = torch.ones(1, len(current), dtype=torch.long, device=cuda())
    start = len(sample["full_input_ids"]) - 1
    full_logits = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).logits[0, start:start + n]

    # 2) Official Native Cache：官方 cache 机制（use_cache=True，同样完整前向）
    off_logits = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True).logits[0, start:start + n]

    # 3) Manual pre-RoPE Cache：context-only pre-RoPE KV 提取 → 注入 → 以 question 继续
    kv = manual_kv(model, sample)
    prompt = sample["question_suffix_ids"]
    current_manual = prompt + answer[:-1]
    prefix = kv[0].shape[1]
    ids_manual = torch.tensor([current_manual], dtype=torch.long, device=cuda())
    pos_manual = torch.arange(prefix, prefix + len(current_manual), device=cuda()).unsqueeze(0)
    mask_manual = torch.ones(1, prefix + len(current_manual), dtype=torch.long, device=cuda())
    cache = dynamic_cache(model, kv[0], kv[1])
    manual_logits = model(
        input_ids=ids_manual, attention_mask=mask_manual, position_ids=pos_manual,
        past_key_values=cache, use_cache=False,
    ).logits[0, len(prompt) - 1:len(prompt) - 1 + n]
    del cache, ids, ids_manual
    torch.cuda.empty_cache()

    pairs = {
        "fulltext_vs_official": (full_logits, off_logits),
        "fulltext_vs_manual": (full_logits, manual_logits),
        "official_vs_manual": (off_logits, manual_logits),
    }
    gate = {}
    for name, (a, b) in pairs.items():
        gate[name] = {
            "top1_match": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
            "mean_logit_kl": logit_kl(a, b),
            "max_abs_logit": (a - b).abs().max().item(),
        }
    return gate


@torch.no_grad()
def run_sample(model, tok, sample, cfg, donor_kv):
    """计算单个样本的各条件 F1 与 cache gate。"""
    qonly = generate(model, tok, sample, cfg, prompt_kind="qonly")[0]
    full = generate(model, tok, sample, cfg, prompt_kind="full")[0]
    own_kv = manual_kv(model, sample)
    manual = generate(model, tok, sample, cfg, key=own_kv[0], value=own_kv[1])[0]
    shuffled = generate(model, tok, sample, cfg, key=donor_kv[0], value=donor_kv[1])[0]
    gate = run_cache_gate(model, sample)
    answer = sample["answer"]
    row = {
        "id": sample["id"],
        "type": sample["type"],
        "answer": answer,
        "qonly_prediction": qonly,
        "fulltext_prediction": full,
        "manual_prediction": manual,
        "shuffled_prediction": shuffled,
        "qonly_f1": token_f1(qonly, answer),
        "fulltext_f1": token_f1(full, answer),
        "manual_f1": token_f1(manual, answer),
        "shuffled_f1": token_f1(shuffled, answer),
        "gate": gate,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="阶段1：模型自身能力基线 + Cache Gate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workdir", required=True, help="实验目录（含 artifacts/manifest.json）")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--out", required=True, help="输出前缀（生成 {out}_per_sample.jsonl 与 {out}_summary.json）")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--receiver-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    seed_all(args.seed)
    cfg = {"receiver_dtype": args.receiver_dtype, "max_new_tokens": args.max_new_tokens}
    tok = tokenizer(args.model)
    model = load_model(args.model, cfg)
    manifest = load_json(Path(args.workdir) / "artifacts" / "manifest.json")[args.split]
    samples = manifest if args.max_samples <= 0 else manifest[: args.max_samples]
    id_map = {sample["id"]: sample for sample in samples}

    per_sample = []
    for index, sample in enumerate(samples, 1):
        donor = id_map[sample["shuffle_id"]]
        donor_kv = manual_kv(model, donor)
        row = run_sample(model, tok, sample, cfg, donor_kv)
        per_sample.append(row)
        progress(f"{args.split} {index}/{len(samples)}")

    keys = ("qonly_f1", "fulltext_f1", "manual_f1", "shuffled_f1")
    summary = {
        "model": args.model,
        "split": args.split,
        "n": len(per_sample),
        "f1": {key: float(sum(row[key] for row in per_sample) / len(per_sample)) for key in keys},
        "f1_by_type": {
            type_name: {
                key: float(sum(row[key] for row in per_sample if row["type"] == type_name) / max(1, sum(1 for row in per_sample if row["type"] == type_name)))
                for key in keys
            }
            for type_name in ("bridge", "comparison")
        },
    }
    summary["self_gain"] = summary["f1"]["fulltext_f1"] - summary["f1"]["qonly_f1"]

    # Cache gate 汇总（方案 §五：top1 ≥ 99.5%，KL < 1e-3，自由生成 100%）
    gate_keys = list(per_sample[0]["gate"]) if per_sample else []
    gate_summary = {}
    for name in gate_keys:
        values = [row["gate"][name] for row in per_sample]
        gate_summary[name] = {
            "mean_top1_match": float(sum(v["top1_match"] for v in values) / len(values)),
            "mean_mean_logit_kl": float(sum(v["mean_logit_kl"] for v in values) / len(values)),
            "mean_max_abs_logit": float(sum(v["max_abs_logit"] for v in values) / len(values)),
        }
    full_manual_agree = sum(1 for row in per_sample if row["fulltext_prediction"] == row["manual_prediction"]) / len(per_sample)
    gate_summary["free_generation_fulltext_vs_manual_agreement"] = full_manual_agree
    gate_summary["cache_gate_passed"] = (
        all(gate_summary[name]["mean_top1_match"] >= 0.995 and gate_summary[name]["mean_mean_logit_kl"] < 1e-3 for name in gate_keys)
        and full_manual_agree == 1.0
    )
    summary["cache_gate"] = gate_summary
    save_json(args.out + "_summary.json", summary)
    write_jsonl(args.out + "_per_sample.jsonl", per_sample)
    print(summary)


if __name__ == "__main__":
    main()
