from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from cache_store import cache_path, load_cache
from common import cuda, load_json, load_model, progress, read_jsonl, save_json, seed_all
from receiver import trajectory
from writers import load_writer, make_writer, save_writer


def cached_pair(cfg, split, sample, device, target_required=True):
    source = load_cache(cache_path(cfg, "source17", split, sample["id"]), sample)
    source_k = source["pre_key"].to(device)
    source_v = source["value"].to(device)
    if not target_required:
        return source_k, source_v, None, None
    target = load_cache(cache_path(cfg, "target4", split, sample["id"]), sample)
    return source_k, source_v, target["pre_key"].to(device), target["value"].to(device)


def token_subset(length: int, count: int, seed: int, device):
    if count <= 0 or count >= length:
        return torch.arange(length, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(length, generator=generator)[:count].sort().values.to(device)


def representation_loss(pred_k, pred_v, target_k, target_v, collect_metrics=False):
    families, total = [], pred_k.new_zeros((), dtype=torch.float32)
    for label, prediction, target in (("k", pred_k, target_k), ("v", pred_v, target_v)):
        prediction, target = prediction.float(), target.float()
        nmse = (prediction - target).square().mean(dim=(1, 2, 3)) / target.square().mean(dim=(1, 2, 3)).clamp_min(1e-8)
        cosine = F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)
        total = total + nmse.mean() + (1.0 - cosine).mean()
        families.append((label, nmse, cosine))
    metrics = []
    if collect_metrics:
        for layer in range(pred_k.shape[0]):
            metrics.append({
                "layer": layer,
                "k_nmse": families[0][1][layer].item(),
                "k_cosine": families[0][2][layer].item(),
                "v_nmse": families[1][1][layer].item(),
                "v_cosine": families[1][2][layer].item(),
            })
    summary = {
        "k_nmse": families[0][1].mean().item(),
        "k_cosine": families[0][2].mean().item(),
        "v_nmse": families[1][1].mean().item(),
        "v_cosine": families[1][2].mean().item(),
    }
    return total, metrics, summary


def hidden_trajectory_loss(student_hidden, teacher_hidden, collect_metrics=False):
    if len(student_hidden) != len(teacher_hidden):
        raise RuntimeError(f"hidden layer count mismatch: {len(student_hidden)} vs {len(teacher_hidden)}")
    total, metrics, layer_nmse, layer_cosine = student_hidden[0].new_zeros((), dtype=torch.float32), [], [], []
    for layer, (student, teacher) in enumerate(zip(student_hidden, teacher_hidden)):
        student, teacher = student.float(), teacher.float()
        if student.shape != teacher.shape:
            raise RuntimeError(f"hidden shape mismatch at layer {layer}: {student.shape} vs {teacher.shape}")
        nmse = (student - teacher).square().mean() / teacher.square().mean().clamp_min(1e-8)
        cosine = F.cosine_similarity(student, teacher, dim=-1).mean()
        total = total + nmse + (1.0 - cosine)
        layer_nmse.append(nmse)
        layer_cosine.append(cosine)
        if collect_metrics:
            metrics.append({"layer": layer, "nmse": nmse.item(), "cosine": cosine.item()})
    summary = {
        "hidden_nmse": torch.stack(layer_nmse).mean().item(),
        "hidden_cosine": torch.stack(layer_cosine).mean().item(),
    }
    return total / len(student_hidden), metrics, summary


def logit_kl_loss(student_logits, teacher_logits, settings):
    if student_logits.shape != teacher_logits.shape:
        raise RuntimeError(f"logit shape mismatch: {student_logits.shape} vs {teacher_logits.shape}")
    mode = settings["kl_token_mode"]
    if mode == "last_n":
        count = min(int(settings["kl_last_n_tokens"]), student_logits.shape[0])
        student_logits, teacher_logits = student_logits[-count:], teacher_logits[-count:]
    elif mode != "all":
        raise ValueError(f"unknown kl_token_mode={mode}")
    temperature = float(settings["kl_temperature"])
    student_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(student_log, teacher_log, reduction="batchmean", log_target=True) * temperature**2


def functional_components(cfg, model, writer, sample, split, reconstruction_seed, collect_metrics=False):
    source_k, source_v, target_k, target_v = cached_pair(cfg, split, sample, cuda(), target_required=True)
    predicted_k, predicted_v = writer(source_k, source_v)
    settings = cfg["stage_b"]
    positions = token_subset(
        source_k.shape[1], settings["reconstruction_token_sample_size"], reconstruction_seed, cuda()
    )
    kv, kv_layers, kv_summary = representation_loss(
        predicted_k[:, positions], predicted_v[:, positions],
        target_k[:, positions], target_v[:, positions], collect_metrics,
    )
    with torch.no_grad():
        teacher = trajectory(model, sample, pre_key=target_k, value=target_v, output_hidden_states=True)
    student = trajectory(model, sample, pre_key=predicted_k, value=predicted_v, output_hidden_states=True)
    hidden, hidden_layers, hidden_summary = hidden_trajectory_loss(student.hidden, teacher.hidden, collect_metrics)
    kl = logit_kl_loss(student.logits, teacher.logits, settings)
    return {"kv": kv, "hidden": hidden, "kl": kl}, {
        "kv_layers": kv_layers, "hidden_layers": hidden_layers,
        "summary": {**kv_summary, **hidden_summary},
    }


def weighted_total(components, settings):
    return (
        float(settings["lambda_kv"]) * components["kv"]
        + float(settings["lambda_hidden"]) * components["hidden"]
        + float(settings["lambda_kl"]) * components["kl"]
    )


@torch.no_grad()
def validate_representation(cfg, writer, samples, split="validation"):
    writer.eval()
    losses, layers = [], []
    for sample_index, sample in enumerate(samples):
        source_k, source_v, target_k, target_v = cached_pair(cfg, split, sample, cuda())
        positions = token_subset(source_k.shape[1], cfg["stage_a"]["token_sample_size"], cfg["seed"] + sample_index, cuda())
        pred_k, pred_v = writer(source_k[:, positions], source_v[:, positions])
        loss, metrics, _ = representation_loss(pred_k, pred_v, target_k[:, positions], target_v[:, positions], True)
        losses.append(loss.item())
        layers.append(metrics)
    aggregate = [{
        "layer": layer,
        **{name: sum(row[layer][name] for row in layers) / len(layers) for name in ("k_nmse", "k_cosine", "v_nmse", "v_cosine")},
    } for layer in range(cfg["target_layers"])]
    writer.train()
    return sum(losses) / len(losses), aggregate


def stage_a(cfg, writer_kind: str, overfit: bool = False):
    train = read_jsonl(Path(cfg["work_dir"]) / "artifacts/manifests/train.jsonl")
    validation = read_jsonl(Path(cfg["work_dir"]) / "artifacts/manifests/validation.jsonl")
    validation_split = "validation"
    if overfit:
        train = train[:cfg["overfit_samples"]]
        validation, validation_split = train, "train"
    writer = make_writer(writer_kind, cfg).to(cuda()).train()
    settings = cfg["stage_a"]
    updates = settings["overfit_updates"] if overfit else settings["updates"]
    optimizer = AdamW(writer.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    destination = Path(cfg["work_dir"]) / "checkpoints" / ("overfit" if overfit else "quick") / writer_kind / "stage_a"
    destination.mkdir(parents=True, exist_ok=True)
    initial, _ = validate_representation(cfg, writer, validation, validation_split)
    history, evaluations, best = [], [], float("inf")
    rng, started = random.Random(cfg["seed"]), time.perf_counter()
    for update in range(1, updates + 1):
        sample = train[rng.randrange(len(train))]
        source_k, source_v, target_k, target_v = cached_pair(cfg, "train", sample, cuda())
        positions = token_subset(source_k.shape[1], settings["token_sample_size"], cfg["seed"] + update, cuda())
        pred_k, pred_v = writer(source_k[:, positions], source_v[:, positions])
        loss, _, component_summary = representation_loss(pred_k, pred_v, target_k[:, positions], target_v[:, positions])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Stage A loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), settings["gradient_clip"])
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite Stage A gradient")
        optimizer.step()
        history.append({"update": update, "loss": loss.item(), "gradient_norm": gradient_norm.item(), **component_summary})
        if update % settings["validation_every"] == 0 or update == updates:
            score, per_layer = validate_representation(cfg, writer, validation, validation_split)
            selected = score < best
            evaluations.append({"update": update, "validation_loss": score, "selected": selected})
            if selected:
                best = score
                save_writer(str(destination / "best.pt"), writer, {"stage": "A", "update": update, "validation_loss": score})
                save_json(destination / "best_layer_metrics.json", per_layer)
            progress(f"Stage A {writer_kind} {update}/{updates} validation={score:.6f}")
    save_json(destination / "history.json", history)
    save_json(destination / "evaluations.json", evaluations)
    save_json(destination / "summary.json", {
        "writer": writer_kind, "overfit_code_validation": overfit,
        "initial_validation_loss": initial, "best_validation_loss": best,
        "loss_decreased": best < initial, "updates": updates,
        "seconds": time.perf_counter() - started, "gold_label_used": False,
    })
    del writer
    torch.cuda.empty_cache()


@torch.no_grad()
def validate_functional(cfg, model, writer, samples, split="validation"):
    writer.eval()
    totals = {name: 0.0 for name in (
        "total", "kv", "hidden", "kl", "k_nmse", "k_cosine", "v_nmse", "v_cosine",
        "hidden_nmse", "hidden_cosine",
    )}
    for index, sample in enumerate(samples):
        components, details = functional_components(cfg, model, writer, sample, split, cfg["seed"] + index)
        total = weighted_total(components, cfg["stage_b"])
        totals["total"] += total.item()
        for name in ("kv", "hidden", "kl"):
            totals[name] += components[name].item()
        for name, value in details["summary"].items():
            totals[name] += value
    writer.train()
    return {name: value / len(samples) for name, value in totals.items()}


def stage_b(cfg, writer_kind: str, overfit: bool = False):
    train = read_jsonl(Path(cfg["work_dir"]) / "artifacts/manifests/train.jsonl")
    validation = read_jsonl(Path(cfg["work_dir"]) / "artifacts/manifests/validation.jsonl")
    validation_split = "validation"
    if overfit:
        train = train[:cfg["overfit_samples"]]
        validation, validation_split = train, "train"
    root = Path(cfg["work_dir"]) / "checkpoints" / ("overfit" if overfit else "quick") / writer_kind
    writer = make_writer(writer_kind, cfg).to(cuda())
    load_writer(str(root / "stage_a/best.pt"), writer)
    writer.train()
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    settings = cfg["stage_b"]
    updates = settings["overfit_updates"] if overfit else settings["updates"]
    optimizer = AdamW(writer.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    destination = root / "stage_b"
    destination.mkdir(parents=True, exist_ok=True)
    initial = validate_functional(cfg, model, writer, validation, validation_split)
    history, evaluations, best = [], [], float("inf")
    rng, started = random.Random(cfg["seed"] + 17), time.perf_counter()
    for update in range(1, updates + 1):
        sample = train[rng.randrange(len(train))]
        components, details = functional_components(cfg, model, writer, sample, "train", cfg["seed"] + update)
        loss = weighted_total(components, settings)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Stage B loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen Receiver unexpectedly accumulated parameter gradients")
        missing = [name for name, parameter in writer.named_parameters() if parameter.requires_grad and parameter.grad is None]
        invalid = [name for name, parameter in writer.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        if missing or invalid:
            raise RuntimeError(f"invalid Writer gradients: missing={missing} nonfinite={invalid}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), settings["gradient_clip"])
        optimizer.step()
        history.append({
            "update": update, "total_loss": loss.item(),
            "kv_loss": components["kv"].item(), "hidden_loss": components["hidden"].item(),
            "kl_loss": components["kl"].item(), "gradient_norm": gradient_norm.item(),
            **details["summary"],
        })
        if update % settings["validation_every"] == 0 or update == updates:
            metrics = validate_functional(cfg, model, writer, validation, validation_split)
            selected = metrics["total"] < best
            evaluations.append({"update": update, **{f"validation_{key}": value for key, value in metrics.items()}, "selected": selected})
            if selected:
                best = metrics["total"]
                save_writer(str(destination / "best.pt"), writer, {"stage": "B", "update": update, "validation": metrics})
            progress(f"Stage B {writer_kind} {update}/{updates} total={metrics['total']:.6f} hidden={metrics['hidden']:.6f} kl={metrics['kl']:.6f}")
    save_json(destination / "history.json", history)
    save_json(destination / "evaluations.json", evaluations)
    save_json(destination / "summary.json", {
        "writer": writer_kind, "overfit_code_validation": overfit,
        "initial_validation": initial, "best_validation_total": best,
        "trajectory_loss_decreased": best < initial["total"], "updates": updates,
        "receiver_frozen": True, "all_36_receiver_layers_supervised": True,
        "all_suffix_tokens_hidden_supervised": True, "gold_label_used": False,
        "loss_weights": {key: settings[key] for key in ("lambda_kv", "lambda_hidden", "lambda_kl")},
        "seconds": time.perf_counter() - started,
    })
    del writer, model
    torch.cuda.empty_cache()


def _gradient_norm(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    squares = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    return math.sqrt(sum(value.item() for value in squares)) if squares else 0.0


def gradient_audit(cfg, writer_kind: str, checkpoint: str | None = None):
    samples = read_jsonl(Path(cfg["work_dir"]) / "artifacts/manifests/train.jsonl")[:cfg["gradient_audit_samples"]]
    writer = make_writer(writer_kind, cfg).to(cuda()).train()
    if checkpoint:
        load_writer(checkpoint, writer)
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    parameters = tuple(parameter for parameter in writer.parameters() if parameter.requires_grad)
    rows = []
    try:
        for index, sample in enumerate(samples):
            components, _ = functional_components(cfg, model, writer, sample, "train", cfg["seed"] + index)
            norms = {name: _gradient_norm(loss, parameters) for name, loss in components.items()}
            rows.append({
                "sample_id": sample["id"],
                "losses": {name: loss.item() for name, loss in components.items()},
                "gradient_norms": norms,
                "weighted_gradient_norms": {
                    "kv": norms["kv"] * cfg["stage_b"]["lambda_kv"],
                    "hidden": norms["hidden"] * cfg["stage_b"]["lambda_hidden"],
                    "kl": norms["kl"] * cfg["stage_b"]["lambda_kl"],
                },
            })
            progress(f"gradient scale audit {index + 1}/{len(samples)}")
    finally:
        del writer, model
        torch.cuda.empty_cache()
    averages = {
        group: {name: sum(row[group][name] for row in rows) / len(rows) for name in ("kv", "hidden", "kl")}
        for group in ("losses", "gradient_norms", "weighted_gradient_norms")
    }
    save_json(Path(cfg["work_dir"]) / "artifacts/gradient_audit/summary.json", {
        "writer": writer_kind, "checkpoint": checkpoint, "sample_count": len(rows),
        "rows": rows, "averages": averages, "weights_changed_automatically": False,
        "gold_label_used": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--writer", choices=("d0", "d1", "d2"), required=True)
    parser.add_argument("--stage", choices=("a", "b", "both", "gradient_audit"), default="both")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    if args.stage in {"a", "both"}:
        stage_a(cfg, args.writer, args.overfit)
    if args.stage in {"b", "both"}:
        stage_b(cfg, args.writer, args.overfit)
    if args.stage == "gradient_audit":
        gradient_audit(cfg, args.writer, args.checkpoint)


if __name__ == "__main__":
    main()
