"""Alignment-only 消融评估：比较 identity / alignment-only / alignment+CE 的功能差异。

设计（用户需求）：
  writer-mode=identity     W=I（LinearWriter 不训练，仅 RMS scale 交换）→ Native 功能基线
  writer-mode=checkpoint   加载 stage_a（alignment-only）或 stage_b（align+CE）best.pt

评估条件（每方向统一）：
  correct           当前样本 Sender KV（经 writer）注入
  shuffled          另一条样本 Sender KV（donor）注入
  zero              全 0 KV 注入（无样本信息的控制）
  learned_constant  固定 KV（训练集第 1 个样本的 Receiver native KV）注入（无样本特定信息的控制）
  qonly             Receiver 只看 Question
  receiver_native   Receiver 自己的 context KV 注入（上限）

核心差值（S_align / S_CE / Δ_CE）由结果文件衍生，本脚本输出全条件 F1/EM。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protocol import (
    Store, cuda, dynamic_cache, generate, load_json, load_model, progress, save_json,
    seed_all, token_f1, tokenizer, write_jsonl,
)
from writer import LinearWriter, load_writer


@torch.no_grad()
def zero_kv(model, num_layers, context_length):
    device = next(model.parameters()).device
    return (
        torch.zeros(num_layers, context_length, 8, 128, dtype=torch.float16, device=device),
        torch.zeros(num_layers, context_length, 8, 128, dtype=torch.float16, device=device),
    )


@torch.no_grad()
def run_conditions(cfg, model, tok, store, samples, writer, constant_kv):
    writer.eval()
    rows = []
    for index, sample in enumerate(samples, 1):
        answer = sample["answer"]
        source = store.source("test", sample["id"])
        key, value = writer(source["pre_key"].to(cuda()), source["value"].to(cuda()))
        pred_correct, _ = generate(model, tok, sample, cfg, key, value)
        donor = store.source("test", sample["shuffle_id"])
        dkey, dvalue = writer(donor["pre_key"].to(cuda()), donor["value"].to(cuda()))
        pred_shuffled, _ = generate(model, tok, sample, cfg, dkey, dvalue)
        zkey, zvalue = zero_kv(model, cfg["num_layers"], len(sample["context_input_ids"]))
        pred_zero, _ = generate(model, tok, sample, cfg, zkey, zvalue)
        pred_const, _ = generate(model, tok, sample, cfg, constant_kv[0], constant_kv[1])
        pred_qonly, _ = generate(model, tok, sample, cfg, prompt_kind="qonly")
        native = store.target("test", sample["id"])
        pred_native, _ = generate(model, tok, sample, cfg, native["pre_key"].to(cuda()), native["value"].to(cuda()))
        rows.append({
            "id": sample["id"],
            "type": sample["type"],
            "answer": answer,
            "f1_correct": token_f1(pred_correct, answer),
            "f1_shuffled": token_f1(pred_shuffled, answer),
            "f1_zero": token_f1(pred_zero, answer),
            "f1_learned_constant": token_f1(pred_const, answer),
            "f1_qonly": token_f1(pred_qonly, answer),
            "f1_receiver_native": token_f1(pred_native, answer),
        })
        if index % 16 == 0:
            progress(f"ablation {index}/{len(samples)}")
    return rows


def mean_f1(rows, key):
    return float(sum(r[key] for r in rows) / len(rows))


def main():
    parser = argparse.ArgumentParser(description="Alignment-only 消融评估")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--writer-mode", choices=("identity", "checkpoint"), required=True)
    parser.add_argument("--writer-checkpoint", help="writer-mode=checkpoint 时的 checkpoint 路径")
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--out", required=True)
    parser.add_argument("--receiver-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    seed_all(args.seed)
    cfg = {
        "work_dir": args.workdir,
        "receiver_dtype": args.receiver_dtype,
        "num_layers": args.num_layers,
        "feature_dim": args.feature_dim,
        "source_dir": args.source_dir,
        "target_dir": args.target_dir,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    manifest = load_json(Path(args.workdir) / "artifacts" / "manifest.json")
    test_samples = manifest["test"]
    constant_sample = manifest["train"][0]
    store = Store(cfg, args.mode, {
        "test": test_samples,
        "train": [constant_sample],
        "validation": [],
    })

    model = load_model(args.receiver_model, cfg)
    tok = tokenizer(args.receiver_model)

    if args.writer_mode == "identity":
        scales = torch.load(
            Path(args.workdir) / "artifacts" / args.mode / args.direction / "scales.pt",
            map_location="cpu", weights_only=False,
        )
        writer = LinearWriter(scales, cfg).to(cuda()).eval()
    else:
        if not args.writer_checkpoint:
            raise ValueError("writer-mode=checkpoint requires --writer-checkpoint")
        writer, _ = load_writer(args.writer_checkpoint, map_location="cuda")
        writer = writer.to(cuda()).eval()

    constant_kv = store.target("train", constant_sample["id"])
    constant_kv = (constant_kv["pre_key"].to(cuda()), constant_kv["value"].to(cuda()))

    rows = run_conditions(cfg, model, tok, store, test_samples, writer, constant_kv)
    conditions = {
        name: {"f1": mean_f1(rows, key), "by_type": {
            t: float(sum(r[key] for r in rows if r["type"] == t) / max(1, sum(1 for r in rows if r["type"] == t)))
            for t in ("bridge", "comparison")
        }}
        for name, key in (
            ("correct", "f1_correct"),
            ("shuffled", "f1_shuffled"),
            ("zero", "f1_zero"),
            ("learned_constant", "f1_learned_constant"),
            ("question_only", "f1_qonly"),
            ("receiver_native", "f1_receiver_native"),
        )
    }
    result = {
        "sender": None,
        "receiver": args.receiver_model,
        "direction": args.direction,
        "writer_mode": args.writer_mode,
        "writer_checkpoint": args.writer_checkpoint,
        "mode": args.mode,
        "n": len(rows),
        "conditions": conditions,
        "derived": {
            "specificity": conditions["correct"]["f1"] - conditions["shuffled"]["f1"],
            "correct_minus_qonly": conditions["correct"]["f1"] - conditions["question_only"]["f1"],
            "correct_minus_zero": conditions["correct"]["f1"] - conditions["zero"]["f1"],
            "correct_minus_constant": conditions["correct"]["f1"] - conditions["learned_constant"]["f1"],
        },
    }
    save_json(args.out, result)
    write_jsonl(str(Path(args.out).with_suffix("")) + "_per_sample.jsonl", rows)
    print({"direction": args.direction, "writer_mode": args.writer_mode, "derived": result["derived"]})


if __name__ == "__main__":
    main()
