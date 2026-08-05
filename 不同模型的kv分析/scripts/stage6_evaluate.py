"""阶段6：最终测试条件 + 核心指标（方案 §十一 / §十二）。

每个方向选择唯一 validation checkpoint 后，在 test 上固定运行：
  correct           当前样本的 Sender KV（经 Writer）
  shuffled          另一条样本的 Sender KV（donor：type 匹配、长度尽量接近、答案不同）
  no_memory         不注入任何外部 KV，只让 Receiver 看 Question（= question_only）
  receiver_fulltext Receiver 直接读取完整文本
  receiver_native   Receiver 自己的正确 Native KV（context-only，不经 Writer，确认目标上限）
  sender_fulltext   Sender 自己读取完整文本（用于 SelfGain）
  sender_qonly      Sender 只看 Question（用于 SelfGain）

correct / shuffled / no_memory 必须用同一个 Writer checkpoint（方案 §十一.3）。

派生指标（方案 §十二）：
  CrossGain         = F1_correct − F1_R^QOnly
  SelfGain_S        = F1_S^FullText − F1_S^QOnly
  ReleaseDelta      = CrossGain − SelfGain_S
  ReceiverRecovery  = (F1_correct − F1_R^QOnly) / (F1_R^FullText − F1_R^QOnly)
  Specificity       = F1_correct − F1_shuffled
全部指标按总体 / Bridge / Comparison 分别报告。

输出符合方案 §十六 的结果 JSON schema。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protocol import (
    Store, capture_native, cuda, dynamic_cache, generate, load_json, load_model, normalize_answer,
    progress, save_json, token_f1, tokenizer, write_jsonl, seed_all,
)
from writer import load_writer


def exact_match(prediction, answer):
    """Normalized EM（问题 9）：统一大小写/标点/冠词/空格后比较。"""
    return float(normalize_answer(prediction) == normalize_answer(answer))


def _em(rows, pred_key):
    return float(sum(exact_match(row[pred_key], row["answer"]) for row in rows) / len(rows))


@torch.no_grad()
def run_receiver_conditions(cfg, model, tok, store, writer, samples):
    """correct / shuffled / no_memory / receiver_fulltext / receiver_native。"""
    writer.eval()
    rows = []
    for index, sample in enumerate(samples, 1):
        answer = sample["answer"]
        # correct：Writer(Sender KV) 注入
        key, value = writer(store.source("test", sample["id"])["pre_key"].to(cuda()),
                            store.source("test", sample["id"])["value"].to(cuda()))
        pred_correct, _ = generate(model, tok, sample, cfg, key, value)
        # shuffled：Writer(donor Sender KV) 注入，Question 不变
        donor = store.source("test", sample["shuffle_id"])
        dkey, dvalue = writer(donor["pre_key"].to(cuda()), donor["value"].to(cuda()))
        pred_shuffled, _ = generate(model, tok, sample, cfg, dkey, dvalue)
        # no_memory：不注入外部 KV，只看 Question
        pred_qonly, _ = generate(model, tok, sample, cfg, prompt_kind="qonly")
        # receiver_fulltext：直接读完整文本
        pred_full, _ = generate(model, tok, sample, cfg, prompt_kind="full")
        # receiver_native：Receiver 自己的 context KV（不经 Writer）→ 上限
        native = store.target("test", sample["id"])
        pred_native, _ = generate(model, tok, sample, cfg, native["pre_key"].to(cuda()), native["value"].to(cuda()))
        rows.append({
            "id": sample["id"],
            "type": sample["type"],
            "answer": answer,
            "correct_prediction": pred_correct,
            "shuffled_prediction": pred_shuffled,
            "qonly_prediction": pred_qonly,
            "receiver_fulltext_prediction": pred_full,
            "receiver_native_prediction": pred_native,
            "f1_correct": token_f1(pred_correct, answer),
            "f1_shuffled": token_f1(pred_shuffled, answer),
            "f1_qonly": token_f1(pred_qonly, answer),
            "f1_receiver_fulltext": token_f1(pred_full, answer),
            "f1_receiver_native": token_f1(pred_native, answer),
        })
        if index % 16 == 0:
            progress(f"receiver conditions {index}/{len(samples)}")
    writer.train()
    return rows


@torch.no_grad()
def run_sender_conditions(cfg, model, tok, samples):
    """sender_fulltext / sender_qonly：Sender 自己读全文 / 只看 Question。"""
    rows = []
    for index, sample in enumerate(samples, 1):
        answer = sample["answer"]
        pred_full, _ = generate(model, tok, sample, cfg, prompt_kind="full")
        pred_qonly, _ = generate(model, tok, sample, cfg, prompt_kind="qonly")
        rows.append({
            "id": sample["id"],
            "type": sample["type"],
            "answer": answer,
            "sender_fulltext_prediction": pred_full,
            "sender_qonly_prediction": pred_qonly,
            "f1_sender_fulltext": token_f1(pred_full, answer),
            "f1_sender_qonly": token_f1(pred_qonly, answer),
        })
        if index % 16 == 0:
            progress(f"sender conditions {index}/{len(samples)}")
    return rows


def mean_f1(rows, key):
    return float(sum(row[key] for row in rows) / len(rows))


def type_f1(rows, key):
    return {t: float(sum(row[key] for row in rows if row["type"] == t) / max(1, sum(1 for row in rows if row["type"] == t))) for t in ("bridge", "comparison")}


def merge(receiver_rows, sender_rows):
    sender_by_id = {row["id"]: row for row in sender_rows}
    merged = []
    for row in receiver_rows:
        s = sender_by_id[row["id"]]
        merged.append({**row, **{k: v for k, v in s.items() if k not in ("id", "type", "answer")}})
    return merged


def main():
    parser = argparse.ArgumentParser(description="阶段6：最终测试条件")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--writer-checkpoint", required=True, help="validation 选出的唯一 checkpoint")
    parser.add_argument("--training-path", required=True, choices=("f1_ce", "stage_a_then_ce", "self"))
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--out", required=True, help="结果 JSON 路径")
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
    manifest = load_json(Path(args.workdir) / "artifacts" / "manifest.json")["test"]
    smoke = {"smoke": 2, "development": 10**9}[args.mode]
    samples = manifest[:smoke]
    store = Store(cfg, args.mode, {"test": samples, "train": [], "validation": []})

    # Receiver 条件
    receiver = load_model(args.receiver_model, cfg)
    tok = tokenizer(args.receiver_model)
    writer, writer_meta = load_writer(args.writer_checkpoint, map_location="cuda")
    writer = writer.to(cuda()).eval()
    receiver_rows = run_receiver_conditions(cfg, receiver, tok, store, writer, samples)
    del receiver, writer
    torch.cuda.empty_cache()

    # Sender 条件（Sender 模型单独加载，避免同时占用显存）
    sender = load_model(args.sender_model, cfg)
    sender_tok = tokenizer(args.sender_model)
    sender_rows = run_sender_conditions(cfg, sender, sender_tok, samples)
    del sender
    torch.cuda.empty_cache()

    merged = merge(receiver_rows, sender_rows)
    conditions = {
        "correct": {
            "f1": mean_f1(merged, "f1_correct"),
            "em": _em(merged, "correct_prediction"),
            "by_type": type_f1(merged, "f1_correct"),
        },
        "shuffled": {
            "f1": mean_f1(merged, "f1_shuffled"),
            "em": _em(merged, "shuffled_prediction"),
            "by_type": type_f1(merged, "f1_shuffled"),
        },
        "question_only": {
            "f1": mean_f1(merged, "f1_qonly"),
            "em": _em(merged, "qonly_prediction"),
            "by_type": type_f1(merged, "f1_qonly"),
        },
        "receiver_full_text": {
            "f1": mean_f1(merged, "f1_receiver_fulltext"),
            "em": _em(merged, "receiver_fulltext_prediction"),
            "by_type": type_f1(merged, "f1_receiver_fulltext"),
        },
        "receiver_native": {
            "f1": mean_f1(merged, "f1_receiver_native"),
            "em": _em(merged, "receiver_native_prediction"),
            "by_type": type_f1(merged, "f1_receiver_native"),
        },
        "sender_full_text": {
            "f1": mean_f1(merged, "f1_sender_fulltext"),
            "em": _em(merged, "sender_fulltext_prediction"),
            "by_type": type_f1(merged, "f1_sender_fulltext"),
        },
        "sender_question_only": {
            "f1": mean_f1(merged, "f1_sender_qonly"),
            "em": _em(merged, "sender_qonly_prediction"),
            "by_type": type_f1(merged, "f1_sender_qonly"),
        },
    }
    cross_gain = conditions["correct"]["f1"] - conditions["question_only"]["f1"]
    self_gain = conditions["sender_full_text"]["f1"] - conditions["sender_question_only"]["f1"]
    recovery_denom = conditions["receiver_full_text"]["f1"] - conditions["question_only"]["f1"]
    def _type_derived(t):
        """按类型计算完整派生指标（问题 13：补 ReleaseDelta_t 与 ReceiverRecovery_t）。"""
        f1_correct = type_f1(merged, "f1_correct")[t]
        f1_shuffled = type_f1(merged, "f1_shuffled")[t]
        f1_qonly = type_f1(merged, "f1_qonly")[t]
        f1_fulltext = type_f1(merged, "f1_receiver_fulltext")[t]
        f1_sender_full = type_f1(merged, "f1_sender_fulltext")[t]
        f1_sender_qonly = type_f1(merged, "f1_sender_qonly")[t]
        cross = f1_correct - f1_qonly
        self_gain = f1_sender_full - f1_sender_qonly
        denom = f1_fulltext - f1_qonly
        return {
            "cross_gain": cross,
            "self_gain": self_gain,
            "release_delta": cross - self_gain,
            "receiver_recovery": (cross / denom) if abs(denom) > 1e-12 else float("nan"),
            "specificity": f1_correct - f1_shuffled,
        }

    result = {
        "sender": args.sender_model,
        "receiver": args.receiver_model,
        "training_path": args.training_path,
        "selected_checkpoint": args.writer_checkpoint,
        "selection_split": "validation",
        "mode": args.mode,
        "n": len(merged),
        "conditions": conditions,
        "derived": {
            "sender_self_gain": self_gain,
            "cross_gain": cross_gain,
            "release_delta": cross_gain - self_gain,
            "receiver_recovery": (cross_gain / recovery_denom) if abs(recovery_denom) > 1e-12 else float("nan"),
            "specificity": conditions["correct"]["f1"] - conditions["shuffled"]["f1"],
        },
        "by_type": {t: _type_derived(t) for t in ("bridge", "comparison")},
    }
    save_json(args.out, result)
    write_jsonl(str(Path(args.out).with_suffix("")) + "_per_sample.jsonl", merged)
    print({k: v for k, v in result["derived"].items()})


if __name__ == "__main__":
    main()
