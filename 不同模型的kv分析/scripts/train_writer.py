"""Writer 训练（方案 §八 阶段3 / §九 阶段4 / §十 阶段5）。

--phase overfit  阶段4：16 样本功能过拟合 Gate（8 Bridge + 8 Comparison），损失 = answer CE，
                 Gate：TrainRecovery ≥ 0.8 且 F1_correct − F1_shuffled ≥ 20 且
                       NLL_correct < 0.5·NLL_update0（更新 0 即 Identity Writer 的 NLL）。
--phase direct   阶段5 路径 F1：从 Identity 初始化直接功能训练，L = answer CE。
--phase stage_a  阶段5 路径 F2 Stage A：表示对齐 L_A = L_K,NMSE + L_V,NMSE + 0.25(L_K,cos + L_V,cos)。
--phase stage_b  阶段5 路径 F2 Stage B：从 Stage A 唯一 checkpoint 出发，L = answer CE，
                 Stage B 不保留任何表示/Identity/层约束。

统一训练设置（方案 §九）：batch_size=1、grad_accum=8、lr=5e-5（Stage A/Direct/Overfit）或
1e-5（Stage B）、weight_decay=0、grad_clip=1.0、max_updates=800。

同模型控制（阶段3）与跨模型共用本脚本：source_dir == target_dir 即 Self Writer
（如 self_06：source 与 target 都是 0.6B 的 context KV），Identity 控制用 --init identity
并在 phase=direct 且 --updates 0 时等价。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from protocol import (
    Store, ce_loss, cuda, generate, load_json, load_model, progress, save_json, seed_all,
    representation_loss, sampled_positions, token_f1, tokenizer,
)
from writer import LinearWriter, parameter_report


def save_writer(path, writer, cfg, extra=None):
    """保存训练 checkpoint（与 stage6 的 writer.load_writer 兼容：{kind, cfg, state_dict, metadata}）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "linear",
        "cfg": cfg,
        "state_dict": {k: v.detach().cpu() for k, v in writer.state_dict().items()},
        "metadata": extra or {},
    }, path)


def scales_for(cfg, mode, direction):
    return torch.load(Path(cfg["work_dir"]) / "artifacts" / mode / direction / "scales.pt", map_location="cpu", weights_only=False)


def writer_for(cfg, mode, direction, checkpoint=None):
    writer = LinearWriter(scales_for(cfg, mode, direction), cfg).to(cuda())
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        writer.load_state_dict(state["writer"] if "writer" in state else state)
    return writer


def select_overfit_samples(rows):
    """固定 16 条：Bridge 8 + Comparison 8（方案 §九）。"""
    bridge = [x for x in rows if x["type"] == "bridge"][:8]
    comparison = [x for x in rows if x["type"] == "comparison"][:8]
    samples = bridge + comparison
    if len(samples) != 16:
        raise RuntimeError(f"overfit needs 8+8 samples, got {len(bridge)} bridge + {len(comparison)} comparison")
    return samples


@torch.no_grad()
def full_source(cfg, store, writer, split, sample, source_id=None):
    record = store.source(split, source_id or sample["id"])
    return writer(record["pre_key"].to(cuda()), record["value"].to(cuda()))


@torch.no_grad()
def validation_generation(cfg, model, tok, writer, store, samples):
    writer.eval()
    scores = []
    for sample in samples:
        key, value = full_source(cfg, store, writer, "validation", sample)
        prediction, _ = generate(model, tok, sample, cfg, key, value)
        scores.append(token_f1(prediction, sample["answer"]))
    writer.train()
    return sum(scores) / len(scores)


@torch.no_grad()
def validation_nll(cfg, model, writer, store, samples):
    writer.eval()
    values = []
    for sample in samples:
        key, value = full_source(cfg, store, writer, "validation", sample)
        values.append(ce_loss(model, sample, key, value)[0].item())
    writer.train()
    return sum(values) / len(values)


@torch.no_grad()
def representation_validation(cfg, store, writer, samples):
    writer.eval()
    losses = []
    for sample in samples:
        source = store.source("validation", sample["id"])
        target = store.target("validation", sample["id"])
        positions = sampled_positions(target["pre_key"].shape[1], cfg["sampled_tokens"])
        sk = source["pre_key"][:, positions].to(cuda())
        sv = source["value"][:, positions].to(cuda())
        tk = target["pre_key"][:, positions].to(cuda())
        tv = target["value"][:, positions].to(cuda())
        pk, pv = writer(sk, sv)
        loss, _ = representation_loss(pk, pv, tk, tv, cfg["stage_a_cosine_weight"])
        losses.append(loss.item())
    writer.train()
    return sum(losses) / len(losses)


def make_store(cfg, mode, rows):
    return Store(cfg, mode, rows)


def train_ce(cfg, mode, phase, store, rows):
    """overfit / direct / stage_b 共用：L = answer CE（方案 §九/§十）。"""
    samples = select_overfit_samples(rows["train"]) if phase == "overfit" else list(rows["train"])
    direction_dir = Path(cfg["work_dir"]) / "artifacts" / mode / cfg["direction"]
    initial = None if phase in ("overfit", "direct") else direction_dir / "stage_a" / "best.pt"
    writer = writer_for(cfg, mode, cfg["direction"], initial).train()
    model = load_model(cfg["receiver_model"], cfg)
    tok = tokenizer(cfg["receiver_model"])
    out = direction_dir / phase
    out.mkdir(parents=True, exist_ok=True)

    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["max_updates"]
    grad_acc = cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]
    lr = cfg["stage_b_lr"] if phase == "stage_b" else cfg["stage_a_lr"]
    optimizer = AdamW(writer.parameters(), lr=lr, weight_decay=cfg["weight_decay"])

    history, evaluations = [], []
    cursor = epoch = 0
    best_f1 = -1.0
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    # 更新 0 基线（Identity Writer 的 NLL，用于 gate 的相对改进）
    nll0 = None
    if phase == "overfit":
        nll0 = validation_nll(cfg, model, writer, store, samples)
        save_writer(out / "update_000.pt", writer, cfg, {"update": 0})

    for update in range(1, maximum + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            key, value = full_source(cfg, store, writer, "train", sample)
            loss = ce_loss(model, sample, key, value)[0]
            (loss / grad_acc).backward()
            batch.append(loss.item())
        norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]).item()
        optimizer.step()
        history.append({"update": update, "loss": sum(batch) / len(batch), "gradient_norm": norm})

        interval = 1 if mode == "smoke" else cfg["eval_every"][phase]
        if update % interval == 0 or update == maximum:
            gen = validation_generation(cfg, model, tok, writer, store, rows["validation"][: cfg["generation_eval_samples"]])
            nll = validation_nll(cfg, model, writer, store, rows["validation"])
            evaluations.append({"update": update, "validation_generation_f1": gen, "validation_nll": nll})
            if gen > best_f1:
                best_f1 = gen
                save_writer(out / "best.pt", writer, cfg, {"update": update, "validation_generation_f1": gen})
            save_writer(out / f"update_{update:03d}.pt", writer, cfg, {"update": update})
            progress(f"{mode}: {phase} {update}/{maximum} gen_f1={gen:.4f} nll={nll:.4f}")

    summary = {
        "phase": phase,
        "initialized_from": "structure_identity" if initial is None else str(initial),
        "best_validation_generation_f1": best_f1,
        "training_seconds": time.perf_counter() - start,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "parameters": parameter_report(writer),
    }
    if phase == "overfit":
        gate = compute_gate(cfg, model, tok, store, writer, samples, nll0)
        summary["gate"] = gate
        summary["gate_passed"] = gate["passed"]
        save_json(out / "gate.json", gate)
    save_json(out / "history.json", history)
    save_json(out / "evaluations.json", evaluations)
    save_json(out / "summary.json", summary)
    del model
    torch.cuda.empty_cache()
    if phase == "overfit" and mode != "smoke" and not summary["gate_passed"]:
        raise RuntimeError(f"overfit gate failed: {gate}")
    progress(f"{mode}: {phase} done gate_passed={summary.get('gate_passed')}")


@torch.no_grad()
def compute_gate(cfg, model, tok, store, writer, samples, nll0):
    """16 样本训练集上的 Receiver 恢复率（方案 §九）：

    TrainRecovery = (F1_correct − F1_qonly) / (F1_fulltext − F1_qonly)
    通过条件：TrainRecovery ≥ 0.8 且 F1_correct − F1_shuffled ≥ 20 且 NLL_correct < 0.5·NLL_update0。
    """
    writer.eval()
    f1_correct, f1_shuffled, f1_qonly, f1_fulltext, nll_correct = [], [], [], [], []
    for sample in samples:
        key, value = full_source(cfg, store, writer, "train", sample)
        pred_correct, _ = generate(model, tok, sample, cfg, key, value)
        donor_key, donor_value = full_source(cfg, store, writer, "train", sample, sample["shuffle_id"])
        pred_shuffled, _ = generate(model, tok, sample, cfg, donor_key, donor_value)
        pred_qonly, _ = generate(model, tok, sample, cfg, prompt_kind="qonly")
        pred_full, _ = generate(model, tok, sample, cfg, prompt_kind="full")
        answer = sample["answer"]
        f1_correct.append(token_f1(pred_correct, answer))
        f1_shuffled.append(token_f1(pred_shuffled, answer))
        f1_qonly.append(token_f1(pred_qonly, answer))
        f1_fulltext.append(token_f1(pred_full, answer))
        nll_correct.append(ce_loss(model, sample, key, value)[0].item())
    writer.train()
    mean = lambda values: sum(values) / len(values)
    recovery_denom = mean(f1_fulltext) - mean(f1_qonly)
    train_recovery = (mean(f1_correct) - mean(f1_qonly)) / recovery_denom if abs(recovery_denom) > 1e-12 else float("nan")
    specificity = mean(f1_correct) - mean(f1_shuffled)
    nll_ratio = mean(nll_correct) / nll0 if nll0 else float("inf")
    passed = (
        train_recovery >= cfg["overfit_required_ratio"]
        and specificity >= 0.20
        and nll_ratio < 0.5
    )
    return {
        "f1_correct": mean(f1_correct),
        "f1_shuffled": mean(f1_shuffled),
        "f1_qonly": mean(f1_qonly),
        "f1_receiver_fulltext": mean(f1_fulltext),
        "specificity": specificity,
        "train_recovery": train_recovery,
        "nll_correct": mean(nll_correct),
        "nll_update0": nll0,
        "nll_ratio": nll_ratio,
        "required_train_recovery": cfg["overfit_required_ratio"],
        "required_specificity": 0.20,
        "required_nll_ratio": 0.5,
        "passed": passed,
    }


def train_stage_a(cfg, mode, store, rows):
    """阶段5 路径 F2 Stage A：表示对齐（方案 §十），不需要加载 Receiver 模型。"""
    writer = writer_for(cfg, mode, cfg["direction"]).train()
    out = Path(cfg["work_dir"]) / "artifacts" / mode / cfg["direction"] / "stage_a"
    out.mkdir(parents=True, exist_ok=True)
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["stage_a_updates"]
    grad_acc = cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]
    optimizer = AdamW(writer.parameters(), lr=cfg["stage_a_lr"], weight_decay=cfg["weight_decay"])
    samples = list(rows["train"])
    validation = rows["validation"]
    history, evaluations = [], []
    cursor = epoch = 0
    best = float("inf")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for update in range(1, maximum + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            source = store.source("train", sample["id"])
            target = store.target("train", sample["id"])
            positions = sampled_positions(target["pre_key"].shape[1], cfg["sampled_tokens"])
            sk = source["pre_key"][:, positions].to(cuda())
            sv = source["value"][:, positions].to(cuda())
            tk = target["pre_key"][:, positions].to(cuda())
            tv = target["value"][:, positions].to(cuda())
            pk, pv = writer(sk, sv)
            loss, _ = representation_loss(pk, pv, tk, tv, cfg["stage_a_cosine_weight"])
            (loss / grad_acc).backward()
            batch.append(loss.item())
        norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]).item()
        optimizer.step()
        history.append({"update": update, "loss": sum(batch) / len(batch), "gradient_norm": norm})
        interval = 1 if mode == "smoke" else cfg["stage_a_eval_every"]
        if update % interval == 0 or update == maximum:
            value = representation_validation(cfg, store, writer, validation)
            selected = value < best
            evaluations.append({"update": update, "validation_representation_loss": value, "selected": selected})
            if selected:
                best = value
                save_writer(out / "best.pt", writer, cfg, {"update": update, "validation_representation_loss": value})
            progress(f"{mode}: stage_a {update}/{maximum} val={value:.6f}")
    save_json(out / "history.json", history)
    save_json(out / "evaluations.json", evaluations)
    save_json(out / "summary.json", {
        "phase": "stage_a",
        "best_validation_representation_loss": best,
        "training_seconds": time.perf_counter() - start,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "parameters": parameter_report(writer),
    })


def main():
    parser = argparse.ArgumentParser(description="Writer 训练（阶段3/4/5）")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--phase", choices=("overfit", "direct", "stage_a", "stage_b"), required=True)
    parser.add_argument("--direction", default="main", help="产物子目录名，如 self_06 / 06_to_17 / 17_to_06")
    parser.add_argument("--receiver-model", help="Receiver 模型路径（overfit/direct/stage_b 需要）")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--sampled-tokens", type=int, default=128)
    parser.add_argument("--max-updates", type=int, default=800)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--stage-a-lr", type=float, default=5e-5)
    parser.add_argument("--stage-b-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--stage-a-cosine-weight", type=float, default=0.25)
    parser.add_argument("--stage-a-eval-every", type=int, default=50)
    parser.add_argument("--eval-every", default="50,50,50", help="overfit,direct,stage_b 的 eval 间隔")
    parser.add_argument("--generation-eval-samples", type=int, default=64)
    parser.add_argument("--overfit-required-ratio", type=float, default=0.8)
    parser.add_argument("--receiver-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    seed_all(args.seed)
    cfg = {
        "work_dir": args.workdir,
        "direction": args.direction,
        "receiver_model": args.receiver_model,
        "source_dir": args.source_dir,
        "target_dir": args.target_dir,
        "num_layers": args.num_layers,
        "feature_dim": args.feature_dim,
        "sampled_tokens": args.sampled_tokens,
        "max_updates": args.max_updates,
        "stage_a_updates": args.max_updates,
        "gradient_accumulation": args.gradient_accumulation,
        "smoke_gradient_accumulation": 1,
        "smoke_updates": 2,
        "stage_a_lr": args.stage_a_lr,
        "stage_b_lr": args.stage_b_lr,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "stage_a_cosine_weight": args.stage_a_cosine_weight,
        "stage_a_eval_every": args.stage_a_eval_every,
        "generation_eval_samples": args.generation_eval_samples,
        "overfit_required_ratio": args.overfit_required_ratio,
        "receiver_dtype": args.receiver_dtype,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    eval_every = [int(x) for x in args.eval_every.split(",")]
    if len(eval_every) != 3:
        raise ValueError("--eval-every needs 3 values for overfit,direct,stage_b")
    cfg["eval_every"] = {"overfit": eval_every[0], "direct": eval_every[1], "stage_b": eval_every[2]}

    manifest = load_json(Path(args.workdir) / "artifacts" / "manifest.json")
    smoke = {"smoke": 4, "development": 10**9}[args.mode]
    rows = {split: values[:smoke] for split, values in manifest.items()}
    store = make_store(cfg, args.mode, rows)

    if args.phase == "stage_a":
        train_stage_a(cfg, args.mode, store, rows)
    else:
        train_ce(cfg, args.mode, args.phase, store, rows)


if __name__ == "__main__":
    main()
