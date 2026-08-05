"""Writer 训练（方案 §八 阶段3 / §九 阶段4 / §十 阶段5）。

--phase overfit  阶段4：功能过拟合 Gate。
                 跨模型方向：TrainRecovery ≥ 0.8 且 F1_correct − F1_shuffled ≥ 20 且
                             NLL_correct < 0.5·NLL_update0（更新 0 即 Identity Writer 的 NLL）。
                 Self 方向（--self）：Update0 与 Native Cache 一致率 100%、Update0 KV max error ≈ 0、
                             训练后 F1 不下降超 2 点、Correct 显著高于 Shuffled。
--phase direct   阶段5 路径 F1：从 Identity 初始化直接功能训练，L = answer CE。
--phase stage_a  阶段5 路径 F2 Stage A：表示对齐 L_A = L_K,NMSE + L_V,NMSE + 0.25(L_K,cos + L_V,cos)。
--phase stage_b  阶段5 路径 F2 Stage B：从 Stage A 唯一 checkpoint 出发，L = answer CE，
                 Stage B 不保留任何表示/Identity/层约束。

统一训练设置（方案 §九）：batch_size=1、grad_accum=8、lr=5e-5（Stage A/Direct/Overfit）或
1e-5（Stage B）、weight_decay=0、grad_clip=1.0、max_updates=800。

实现要点（审查修复）：
- 训练路径上 Writer 前向不置于 no_grad 之下；每次 loss 校验 requires_grad，首步校验梯度非零。
- 过拟合 eval 固定在训练集 16 条上，不参与 validation checkpoint selection；
  gate 使用训练集 best checkpoint，该 checkpoint 只用于诊断，正式训练重新 Identity 初始化。
- Gate 失败不 raise：始终写出 gate.json，由 run_pipeline 收集并决定 formal 是否继续。
"""

from __future__ import annotations

import argparse
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
        writer.load_state_dict(state["state_dict"])
    return writer


def select_overfit_samples(rows, per_type=8):
    """固定 per_type×2 条（Development 8+8，Smoke 2+2）：Bridge/Comparison 各半（方案 §九）。"""
    bridge = [x for x in rows if x["type"] == "bridge"][:per_type]
    comparison = [x for x in rows if x["type"] == "comparison"][:per_type]
    samples = bridge + comparison
    if len(samples) != 2 * per_type:
        raise RuntimeError(f"overfit needs {per_type}+{per_type} samples, got {len(bridge)} bridge + {len(comparison)} comparison")
    return samples


def full_source(cfg, store, writer, split, sample, source_id=None):
    """Writer 前向（训练路径保持梯度；评估函数自带 no_grad）。"""
    record = store.source(split, source_id or sample["id"])
    return writer(record["pre_key"].to(cuda()), record["value"].to(cuda()))


@torch.no_grad()
def evaluate_generation(cfg, model, tok, writer, store, split, samples):
    writer.eval()
    scores = []
    for sample in samples:
        key, value = full_source(cfg, store, writer, split, sample)
        prediction, _ = generate(model, tok, sample, cfg, key, value)
        scores.append(token_f1(prediction, sample["answer"]))
    writer.train()
    return sum(scores) / len(scores)


@torch.no_grad()
def evaluate_nll(cfg, model, writer, store, split, samples):
    writer.eval()
    values = []
    for sample in samples:
        key, value = full_source(cfg, store, writer, split, sample)
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


@torch.no_grad()
def identity_baseline_self(cfg, model, tok, store, writer, samples):
    """Self 方向的 Update0 基线：Identity Writer 应与 Receiver Native Cache 严格等价。"""
    agreements, kv_errors, f1_native = [], [], []
    for sample in samples:
        key, value = full_source(cfg, store, writer, "train", sample)
        pred_update0, _ = generate(model, tok, sample, cfg, key, value)
        native = store.target("train", sample["id"])
        pred_native, _ = generate(model, tok, sample, cfg, native["pre_key"].to(cuda()), native["value"].to(cuda()))
        agreements.append(pred_update0 == pred_native)
        f1_native.append(token_f1(pred_native, sample["answer"]))
        target_k = native["pre_key"].float().to(cuda())
        target_v = native["value"].float().to(cuda())
        kv_errors.append((key.float() - target_k).abs().max().item())
        kv_errors.append((value.float() - target_v).abs().max().item())
    return {
        "update0_native_agreement": float(sum(agreements) / len(agreements)),
        "update0_kv_max_error": float(max(kv_errors)),
        "f1_native": float(sum(f1_native) / len(f1_native)),
    }


@torch.no_grad()
def compute_self_gate(cfg, model, tok, store, writer, samples, baseline):
    """Self Writer Gate（方案 §八）：Identity 一致 + 训练不破坏 Native 功能 + Correct > Shuffled。"""
    writer.eval()
    f1_correct, f1_shuffled = [], []
    for sample in samples:
        key, value = full_source(cfg, store, writer, "train", sample)
        pred_correct, _ = generate(model, tok, sample, cfg, key, value)
        donor_key, donor_value = full_source(cfg, store, writer, "train", sample, sample["shuffle_id"])
        pred_shuffled, _ = generate(model, tok, sample, cfg, donor_key, donor_value)
        f1_correct.append(token_f1(pred_correct, sample["answer"]))
        f1_shuffled.append(token_f1(pred_shuffled, sample["answer"]))
    writer.train()
    mean = lambda values: sum(values) / len(values)
    f1_trained = mean(f1_correct)
    specificity = f1_trained - mean(f1_shuffled)
    f1_drop = baseline["f1_native"] - f1_trained
    passed = (
        baseline["update0_native_agreement"] == 1.0
        and baseline["update0_kv_max_error"] <= cfg["self_kv_max_error"]
        and f1_drop <= 0.02
        and specificity >= 0.20
    )
    return {
        "f1_trained": f1_trained,
        "f1_shuffled": mean(f1_shuffled),
        "f1_native": baseline["f1_native"],
        "f1_drop": f1_drop,
        "specificity": specificity,
        "update0_native_agreement": baseline["update0_native_agreement"],
        "update0_kv_max_error": baseline["update0_kv_max_error"],
        "required_f1_drop": 0.02,
        "required_specificity": 0.20,
        "required_update0_agreement": 1.0,
        "passed": passed,
    }


@torch.no_grad()
def compute_cross_gate(cfg, model, tok, store, writer, samples, nll0):
    """跨模型方向 Gate（方案 §九）：

    TrainRecovery = (F1_correct − F1_qonly) / (F1_receiver_fulltext − F1_qonly)
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


def train_ce(cfg, mode, phase, store, rows):
    """overfit / direct / stage_b 共用：L = answer CE（方案 §九/§十）。"""
    per_type = 2 if mode == "smoke" else 8
    samples = select_overfit_samples(rows["train"], per_type) if phase == "overfit" else list(rows["train"])
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
    best_metric = -1.0 if phase != "overfit" else -1.0
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    # 过拟合阶段的 eval 固定在训练集（问题 12：不做 validation checkpoint selection）
    eval_split = "train" if phase == "overfit" else "validation"
    eval_samples = samples if phase == "overfit" else rows["validation"]

    # Update 0 基线（gate 相对改进的参照）
    baseline = None
    nll0 = None
    if phase == "overfit":
        if cfg["is_self"]:
            baseline = identity_baseline_self(cfg, model, tok, store, writer, samples)
            save_writer(out / "update_000.pt", writer, cfg, {"update": 0, **baseline})
        else:
            nll0 = evaluate_nll(cfg, model, writer, store, "train", samples)
            save_writer(out / "update_000.pt", writer, cfg, {"update": 0})

    gradient_checked = False
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
            if not loss.requires_grad:
                raise RuntimeError("CE loss is detached from graph; Writer forward must not be under no_grad")
            (loss / grad_acc).backward()
            batch.append(loss.item())
        # 首次 backward 后验证 Writer 确实收到梯度（问题 1）
        if not gradient_checked:
            non_zero = [p for p in writer.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0]
            if not non_zero:
                raise RuntimeError("no Writer parameter received gradient; training is broken")
            gradient_checked = True
        norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]).item()
        optimizer.step()
        history.append({"update": update, "loss": sum(batch) / len(batch), "gradient_norm": norm})

        interval = 1 if mode == "smoke" else cfg["eval_every"][phase]
        if update % interval == 0 or update == maximum:
            gen = evaluate_generation(cfg, model, tok, writer, store, eval_split, eval_samples[: cfg["generation_eval_samples"]])
            nll = evaluate_nll(cfg, model, writer, store, eval_split, eval_samples)
            evaluations.append({"update": update, f"{eval_split}_generation_f1": gen, f"{eval_split}_nll": nll})
            if gen > best_metric:
                best_metric = gen
                save_writer(out / "best.pt", writer, cfg, {"update": update, f"{eval_split}_generation_f1": gen})
            save_writer(out / f"update_{update:03d}.pt", writer, cfg, {"update": update})
            progress(f"{mode}: {phase} {update}/{maximum} {eval_split}_gen_f1={gen:.4f} nll={nll:.4f}")

    summary = {
        "phase": phase,
        "direction": cfg["direction"],
        "is_self": cfg["is_self"],
        "initialized_from": "structure_identity" if initial is None else str(initial),
        f"best_{eval_split}_generation_f1": best_metric,
        "training_seconds": time.perf_counter() - start,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "parameters": parameter_report(writer),
    }
    if phase == "overfit":
        # Gate 使用训练集 best checkpoint（问题 12）；正式训练重新 Identity 初始化，绝不复用它
        gate_writer = writer
        best_path = out / "best.pt"
        if best_path.exists():
            gate_writer = writer_for(cfg, mode, cfg["direction"], best_path)
        gate = compute_self_gate(cfg, model, tok, store, gate_writer, samples, baseline) if cfg["is_self"] else compute_cross_gate(cfg, model, tok, store, gate_writer, samples, nll0)
        summary["gate"] = gate
        summary["gate_passed"] = gate["passed"]
        save_json(out / "gate.json", gate)
        progress(f"{mode}: {phase} gate passed={gate['passed']}")
    save_json(out / "history.json", history)
    save_json(out / "evaluations.json", evaluations)
    save_json(out / "summary.json", summary)
    del model
    torch.cuda.empty_cache()
    progress(f"{mode}: {phase} done")


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
            if not loss.requires_grad:
                raise RuntimeError("Stage A loss is detached from graph")
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
        "direction": cfg["direction"],
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
    parser.add_argument("--self-kv-max-error", type=float, default=1e-3, help="Self Gate：Update0 KV max error 阈值")
    parser.add_argument("--receiver-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    seed_all(args.seed)
    cfg = {
        "work_dir": args.workdir,
        "direction": args.direction,
        "is_self": args.direction.startswith("self_"),
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
        "self_kv_max_error": args.self_kv_max_error,
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
