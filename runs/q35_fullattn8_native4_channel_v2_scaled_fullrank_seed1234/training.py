from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from v2_common import (
    Stores, accumulation, checkpoint_writer, cuda, load_json, load_reader,
    load_writer, loss_rows, normalized_target, predict, progress, rows_for,
    save_json, seed_all, validate_rep,
)


def mean_score(rows):
    return sum(x["representation_score"] for x in rows) / len(rows)


def sample_rep_loss(cfg, writer, store, split, sample, stage):
    source_k, source_v, mask = store.memory(split, "q35", sample)
    target_k, target_v, _ = store.memory(split, "4b", sample)
    source_k, source_v = source_k.to(cuda()), source_v.to(cuda())
    target_k, target_v = target_k.to(cuda()), target_v.to(cuda())
    gold_k, gold_v = normalized_target(writer, target_k, target_v)
    if stage == "a1":
        pred_k, pred_v = writer.features(source_k, source_v)
        gold_k = gold_k[cfg["target_depth_anchors"]]
        gold_v = gold_v[cfg["target_depth_anchors"]]
    else:
        pred_k, pred_v = writer.standardized_output(source_k, source_v)
    valid = mask[0].bool().to(cuda())
    rows = loss_rows(pred_k, pred_v, gold_k, gold_v, valid)
    losses = torch.stack([
        x["k_smooth_l1"] + x["v_smooth_l1"]
        + .1 * (1 - x["k_cosine"]) + .1 * (1 - x["v_cosine"])
        for x in rows
    ])
    return losses.mean()


def train_a1(cfg, mode, overfit=False):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    writer = load_writer(cfg, mode, "v1").train()
    optimizer = AdamW(
        writer.feature_parameters(), lr=cfg["a1_lr"],
        weight_decay=cfg["weight_decay"],
    )
    if mode == "smoke":
        maximum = cfg["smoke_updates"]
    else:
        maximum = cfg["overfit_updates"] if overfit else cfg["a1_updates"]
    train_rows = list(rows["train"][:cfg["overfit_samples"]]) if overfit else list(rows["train"])
    grad_acc = accumulation(cfg, mode)
    interval = 1 if mode == "smoke" else cfg["eval_interval"]
    name = "a1_overfit16" if overfit else "a1"
    out = Path(cfg["work_dir"]) / "artifacts" / mode / name
    out.mkdir(parents=True, exist_ok=True)
    best, history, evaluations, cursor, epoch = float("inf"), [], [], 0, 0
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        values = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + epoch).shuffle(train_rows)
                epoch += 1
            sample = train_rows[cursor]
            cursor = (cursor + 1) % len(train_rows)
            loss = sample_rep_loss(cfg, writer, store, "train", sample, "a1")
            (loss / grad_acc).backward()
            values.append(loss.detach().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            writer.feature_parameters(), cfg["gradient_clip"]
        ).item()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        history.append({
            "update": update, "loss": sum(values) / len(values),
            "grad_norm": grad_norm,
        })
        if update % interval == 0 or update == maximum:
            metrics = validate_rep(cfg, writer, store, rows["validation"], "a1")
            score = mean_score(metrics)
            selected = score < best
            if selected:
                best = score
                checkpoint_writer(
                    out / "best.pt", writer, update=update,
                    validation_score=score,
                )
                save_json(out / "per_layer_metrics.json", metrics)
            evaluations.append({
                "update": update, "validation_score": score,
                "selected": selected, "per_layer": metrics,
            })
            save_json(out / "history.json", history)
            save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}/{name}: {update}/{maximum}")
    save_json(out / "summary.json", {
        "completed": True, "best_validation_score": best,
        "hard_gate": None,
    })


@torch.no_grad()
def observe_nll(cfg, writer, store, reader, tok, r1, samples):
    writer.eval()
    values = []
    for sample in samples[:min(8, len(samples))]:
        output, mask = predict(cfg, writer, store, "validation", sample)
        values.append(r1.answer_loss(
            cfg, reader, tok, sample, output[0].half(), output[1].half(), mask
        ).item())
    writer.train()
    return sum(values) / len(values)


def train_a2(cfg, mode):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    a1 = Path(cfg["work_dir"]) / "artifacts" / mode / "a1" / "best.pt"
    writer = load_writer(cfg, mode, "v2")
    a1_state = torch.load(a1, map_location="cpu", weights_only=False)["writer"]
    own = writer.state_dict()
    own["feature_k"] = a1_state["feature_k"]
    own["feature_v"] = a1_state["feature_v"]
    writer.load_state_dict(own)
    optimizer = AdamW([
        {"params": writer.feature_parameters(), "lr": cfg["a2_feature_lr"]},
        {"params": writer.depth_parameters(), "lr": cfg["a2_depth_lr"]},
        {"params": writer.calibration_parameters(), "lr": cfg["a2_calibration_lr"]},
    ], weight_decay=cfg["weight_decay"])
    r1, reader, tok = load_reader(cfg, mode)
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["a2_updates"]
    grad_acc = accumulation(cfg, mode)
    interval = 1 if mode == "smoke" else cfg["eval_interval"]
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "a2"
    out.mkdir(parents=True, exist_ok=True)
    best, history, evaluations, samples, cursor, epoch = (
        float("inf"), [], [], list(rows["train"]), 0, 0
    )
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        values = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + 2000 + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            writer.train()
            loss = sample_rep_loss(cfg, writer, store, "train", sample, "a2")
            (loss / grad_acc).backward()
            values.append(loss.detach().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            writer.parameters(), cfg["gradient_clip"]
        ).item()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        history.append({
            "update": update, "loss": sum(values) / len(values),
            "grad_norm": grad_norm,
        })
        if update % interval == 0 or update == maximum:
            metrics = validate_rep(cfg, writer, store, rows["validation"], "a2")
            score = mean_score(metrics)
            nll = observe_nll(cfg, writer, store, reader, tok, r1, rows["validation"])
            selected = score < best
            if selected:
                best = score
                checkpoint_writer(
                    out / "best.pt", writer, update=update,
                    validation_score=score, teacher_forcing_nll=nll,
                )
                save_json(out / "per_layer_metrics.json", metrics)
            evaluations.append({
                "update": update, "validation_score": score,
                "teacher_forcing_nll_observation": nll,
                "selected": selected, "per_layer": metrics,
            })
            save_json(out / "history.json", history)
            save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}/a2: {update}/{maximum}")
    save_json(out / "summary.json", {
        "completed": True, "best_validation_score": best,
        "generation_loss_weight": 0.0, "hard_gate": None,
    })
    del reader
    torch.cuda.empty_cache()


def train_b(cfg, mode):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    checkpoint = Path(cfg["work_dir"]) / "artifacts" / mode / "a2" / "best.pt"
    writer = load_writer(cfg, mode, "v2", checkpoint).train()
    optimizer = AdamW([
        {"params": writer.feature_parameters(), "lr": cfg["b_feature_lr"]},
        {"params": writer.depth_parameters(), "lr": cfg["b_depth_calibration_lr"]},
        {"params": writer.calibration_parameters(), "lr": cfg["b_depth_calibration_lr"]},
    ], weight_decay=cfg["weight_decay"])
    r1, reader, tok = load_reader(cfg, mode)
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["b_updates"]
    grad_acc = accumulation(cfg, mode)
    interval = 1 if mode == "smoke" else cfg["eval_interval"]
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "b"
    out.mkdir(parents=True, exist_ok=True)
    best, history, evaluations, samples, cursor, epoch = (
        float("inf"), [], [], list(rows["train"]), 0, 0
    )
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        values = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + 4000 + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            writer.train()
            output, mask = predict(cfg, writer, store, "train", sample)
            answer_nll = r1.answer_loss(
                cfg, reader, tok, sample,
                output[0].half(), output[1].half(), mask,
            )
            (answer_nll / grad_acc).backward()
            values.append(answer_nll.detach().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            writer.parameters(), cfg["gradient_clip"]
        ).item()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        history.append({
            "update": update, "answer_nll": sum(values) / len(values),
            "grad_norm": grad_norm, "kv_loss_weight": 0.0,
            "distillation_weight": 0.0, "margin_weight": 0.0,
        })
        if update % interval == 0 or update == maximum:
            validation_nll = observe_nll(
                cfg, writer, store, reader, tok, r1, rows["validation"]
            )
            selected = validation_nll < best
            if selected:
                best = validation_nll
                checkpoint_writer(
                    out / "best.pt", writer, update=update,
                    validation_answer_nll=validation_nll,
                    selection_metric="validation_answer_nll",
                )
            evaluations.append({
                "update": update, "validation_answer_nll": validation_nll,
                "selected": selected,
            })
            save_json(out / "history.json", history)
            save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}/b answer-only: {update}/{maximum}")
    save_json(out / "summary.json", {
        "completed": True, "best_validation_answer_nll": best,
        "only_loss": "gold_answer_teacher_forcing_cross_entropy",
        "hard_gate": None,
    })


def cli(stage):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    if stage == "overfit":
        train_a1(cfg, args.mode, overfit=True)
    elif stage == "a1":
        train_a1(cfg, args.mode)
    elif stage == "a2":
        train_a2(cfg, args.mode)
    else:
        train_b(cfg, args.mode)
