from __future__ import annotations

import argparse
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


GOLD_FIELDS = frozenset({"gold_index", "gold_label", "answer", "answer_index", "cot_content"})


def training_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Return the only sample view allowed inside Stage A/B."""
    return {key: value for key, value in sample.items() if key not in GOLD_FIELDS}


def load_training_split(cfg, split):
    root = Path(cfg.get("manifest_dir", Path(cfg["work_dir"]) / "artifacts/manifests"))
    path = root / f"{split}.jsonl"
    return [training_sample(sample) for sample in read_jsonl(path)]


def epoch_permutation(size: int, seed: int, epoch: int) -> list[int]:
    order = list(range(size))
    random.Random(seed + 17 + epoch).shuffle(order)
    if len(order) != len(set(order)) or set(order) != set(range(size)):
        raise RuntimeError("epoch sampler did not produce a true permutation")
    return order


def gradient_was_clipped(pre_clip_norm: float, threshold: float) -> bool:
    return float(pre_clip_norm) > float(threshold)


def trainable_parameters(writer):
    parameters = tuple(parameter for parameter in writer.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("Writer has no trainable parameters")
    return parameters


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


def behavioral_kl_loss(student_logits, teacher_logits, settings, mode: str):
    if student_logits.shape != teacher_logits.shape:
        raise RuntimeError(f"logit shape mismatch: {student_logits.shape} vs {teacher_logits.shape}")
    if mode == "final":
        student_logits, teacher_logits = student_logits[-1:], teacher_logits[-1:]
    elif mode == "last_n":
        count = min(int(settings["functional_last_n_tokens"]), student_logits.shape[0])
        student_logits, teacher_logits = student_logits[-count:], teacher_logits[-count:]
    elif mode != "all":
        raise ValueError(f"unknown functional mode={mode}")
    temperature = float(settings["temperature"])
    student_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) * temperature**2 / student_logits.shape[0]


def functional_loss(cfg, model, writer, sample, split, mode):
    source_k, source_v, target_k, target_v = cached_pair(cfg, split, sample, cuda(), target_required=True)
    predicted_k, predicted_v = writer(source_k, source_v)
    with torch.no_grad():
        teacher = trajectory(model, sample, pre_key=target_k, value=target_v, output_hidden_states=False)
    student = trajectory(model, sample, pre_key=predicted_k, value=predicted_v, output_hidden_states=False)
    return behavioral_kl_loss(student.logits, teacher.logits, cfg["stage_b"], mode)


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
    train = load_training_split(cfg, "train")
    validation = load_training_split(cfg, "validation")
    validation_split = "validation"
    if overfit:
        train = train[:cfg["overfit_samples"]]
        validation, validation_split = train, "train"
    writer = make_writer(writer_kind, cfg).to(cuda()).train()
    settings = cfg["stage_a"]
    updates = settings["overfit_updates"] if overfit else settings["updates"]
    parameters = trainable_parameters(writer)
    optimizer = AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    destination = Path(cfg["work_dir"]) / "checkpoints" / ("overfit" if overfit else "quick") / writer_kind / "stage_a"
    destination.mkdir(parents=True, exist_ok=True)
    initial, _ = validate_representation(cfg, writer, validation, validation_split)
    history, evaluations, best = [], [{"update": 0, "validation_loss": initial, "selected": True}], initial
    save_writer(str(destination / "best.pt"), writer, {
        "stage": "A", "update": 0, "validation_loss": initial,
        "initialized_from": cfg["base_stage_a_checkpoint"],
    })
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
        missing = [name for name, parameter in writer.named_parameters() if parameter.requires_grad and parameter.grad is None]
        if missing:
            raise RuntimeError(f"Stage A missing trainable gradients: {missing}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"])
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
        "depth_projection_frozen": True,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "seconds": time.perf_counter() - started, "gold_label_used": False,
    })
    del writer
    torch.cuda.empty_cache()


@torch.no_grad()
def validate_functional(cfg, model, writer, samples, mode, split="validation"):
    writer.eval()
    losses = [functional_loss(cfg, model, writer, sample, split, mode).item() for sample in samples]
    writer.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def diagnose_writer(cfg, model, writer, samples, split="validation"):
    writer.eval()
    totals = {name: 0.0 for name in (
        "kv_loss", "k_nmse", "k_cosine", "v_nmse", "v_cosine",
        "hidden_loss", "hidden_nmse", "hidden_cosine", "final_kl", "all_kl",
    )}
    per_layer_kv, per_layer_hidden = [], []
    for sample in samples:
        source_k, source_v, target_k, target_v = cached_pair(cfg, split, sample, cuda())
        predicted_k, predicted_v = writer(source_k, source_v)
        kv_loss, kv_layers, kv_summary = representation_loss(predicted_k, predicted_v, target_k, target_v, True)
        teacher = trajectory(model, sample, pre_key=target_k, value=target_v, output_hidden_states=True)
        student = trajectory(model, sample, pre_key=predicted_k, value=predicted_v, output_hidden_states=True)
        hidden_loss, hidden_layers, hidden_summary = hidden_trajectory_loss(student.hidden, teacher.hidden, True)
        totals["kv_loss"] += kv_loss.item()
        totals["hidden_loss"] += hidden_loss.item()
        for key, value in {**kv_summary, **hidden_summary}.items():
            totals[key] += value
        totals["final_kl"] += behavioral_kl_loss(student.logits, teacher.logits, cfg["stage_b"], "final").item()
        totals["all_kl"] += behavioral_kl_loss(student.logits, teacher.logits, cfg["stage_b"], "all").item()
        per_layer_kv.append(kv_layers)
        per_layer_hidden.append(hidden_layers)
    count = len(samples)
    result = {name: value / count for name, value in totals.items()}
    result["sample_count"] = count
    result["kv_per_layer"] = [{
        "layer": layer,
        **{key: sum(rows[layer][key] for rows in per_layer_kv) / count for key in ("k_nmse", "k_cosine", "v_nmse", "v_cosine")},
    } for layer in range(cfg["target_layers"])]
    result["hidden_per_layer"] = [{
        "layer": layer,
        **{key: sum(rows[layer][key] for rows in per_layer_hidden) / count for key in ("nmse", "cosine")},
    } for layer in range(cfg["target_layers"])]
    writer.train()
    return result


def stage_b(cfg, writer_kind: str, mode: str, overfit: bool = False):
    if overfit:
        raise ValueError("epoch-sampler experiment does not define an overfit mode")
    if mode not in cfg["stage_b"]["modes"]:
        raise ValueError(f"functional mode {mode} is not enabled")
    train = load_training_split(cfg, "train")
    validation = load_training_split(cfg, "validation")
    validation_split = "validation"
    root = Path(cfg["work_dir"]) / "checkpoints/quick" / writer_kind
    stage_a_checkpoint = Path(cfg["stage_a_checkpoint"])
    writer = make_writer(writer_kind, cfg).to(cuda())
    load_writer(str(stage_a_checkpoint), writer)
    writer.train()
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    settings = cfg["stage_b"]
    if settings["micro_batch_size"] != 1:
        raise ValueError("this implementation requires micro_batch_size=1")
    accumulation = int(settings["gradient_accumulation_steps"])
    if accumulation != int(settings["effective_batch_size"]):
        raise ValueError("effective_batch_size must equal gradient_accumulation_steps for micro_batch_size=1")
    if len(train) % accumulation:
        raise ValueError(f"train size {len(train)} must be divisible by effective batch {accumulation}")
    optimizer_steps_per_epoch = len(train) // accumulation
    if int(settings["validation_every_epochs"]) != 1:
        raise ValueError("early stopping requires validation_every_epochs=1")
    parameters = trainable_parameters(writer)
    optimizer = AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    destination = root / f"stage_b_{mode}"
    destination.mkdir(parents=True, exist_ok=True)
    initial = validate_functional(cfg, model, writer, validation, mode, validation_split)
    history, evaluations, diagnostics, epoch_summaries = [], [], [], []
    best, best_epoch, epochs_without_improvement = initial, 0, 0
    total_optimizer_steps, total_clipped_steps, total_exposures = 0, 0, 0
    stopped_early, early_stopping_triggered, started = False, False, time.perf_counter()
    save_writer(str(destination / "best.pt"), writer, {
        "stage": "B", "functional_mode": mode, "epoch": 0,
        "validation_functional_kl": initial, "initialized_from": str(stage_a_checkpoint),
    })
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        order = epoch_permutation(len(train), cfg["seed"], epoch)
        epoch_losses, epoch_clipped = [], 0
        for batch_index in range(optimizer_steps_per_epoch):
            indices = order[batch_index * accumulation:(batch_index + 1) * accumulation]
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for sample_index in indices:
                loss = functional_loss(cfg, model, writer, train[sample_index], "train", mode)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite Stage B functional KL")
                batch_losses.append(loss.item())
                (loss / accumulation).backward()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("frozen Receiver unexpectedly accumulated parameter gradients")
            missing = [name for name, parameter in writer.named_parameters() if parameter.requires_grad and parameter.grad is None]
            invalid = [name for name, parameter in writer.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            if missing or invalid:
                raise RuntimeError(f"invalid Writer gradients: missing={missing} nonfinite={invalid}")
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"])
            if not torch.isfinite(pre_clip_norm):
                raise RuntimeError("non-finite accumulated gradient norm")
            clipped = gradient_was_clipped(pre_clip_norm.item(), settings["gradient_clip"])
            optimizer.step()
            total_optimizer_steps += 1
            total_exposures += len(indices)
            total_clipped_steps += int(clipped)
            epoch_clipped += int(clipped)
            epoch_losses.extend(batch_losses)
            history.append({
                "epoch": epoch, "optimizer_step": total_optimizer_steps,
                "batch_in_epoch": batch_index + 1,
                "sample_ids": [train[index]["id"] for index in indices],
                "mean_functional_kl": sum(batch_losses) / len(batch_losses),
                "pre_clip_gradient_norm": pre_clip_norm.item(),
                "gradient_clip_threshold": float(settings["gradient_clip"]),
                "clipped": clipped,
            })
        score = validate_functional(cfg, model, writer, validation, mode, validation_split)
        selected = score < best - float(settings["early_stopping_min_delta"])
        if selected:
            best, best_epoch, epochs_without_improvement = score, epoch, 0
            save_writer(str(destination / "best.pt"), writer, {
                "stage": "B", "functional_mode": mode, "epoch": epoch,
                "optimizer_step": total_optimizer_steps,
                "validation_functional_kl": score, "initialized_from": str(stage_a_checkpoint),
            })
        else:
            epochs_without_improvement += 1
        epoch_summary = {
            "epoch": epoch, "optimizer_steps": optimizer_steps_per_epoch,
            "sample_exposures": len(train), "unique_samples": len(set(order)),
            "duplicate_exposures_within_epoch": len(order) - len(set(order)),
            "mean_train_functional_kl": sum(epoch_losses) / len(epoch_losses),
            "validation_functional_kl": score, "selected": selected,
            "clipped_optimizer_steps": epoch_clipped,
            "clip_rate": epoch_clipped / optimizer_steps_per_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }
        epoch_summaries.append(epoch_summary)
        evaluations.append(epoch_summary.copy())
        progress(
            f"Stage B {mode} epoch={epoch}/{settings['max_epochs']} "
            f"validation={score:.6f} clip_rate={epoch_summary['clip_rate']:.4f}"
        )
        if epoch % int(settings["diagnostic_every_epochs"]) == 0:
            diagnostic = diagnose_writer(
                cfg, model, writer, validation[:settings["diagnostic_samples"]], validation_split
            )
            diagnostics.append({"epoch": epoch, "optimizer_step": total_optimizer_steps, **diagnostic})
        if epochs_without_improvement >= int(settings["early_stopping_patience_epochs"]):
            early_stopping_triggered = True
            stopped_early = epoch < int(settings["max_epochs"])
            progress(f"Stage B early stopping at epoch={epoch}; best_epoch={best_epoch}")
            break
    load_writer(str(destination / "best.pt"), writer)
    best_diagnostic = diagnose_writer(
        cfg, model, writer, validation[:settings["diagnostic_samples"]], validation_split
    )
    save_json(destination / "history.json", history)
    save_json(destination / "evaluations.json", evaluations)
    save_json(destination / "epoch_summaries.json", epoch_summaries)
    save_json(destination / "diagnostics.json", diagnostics)
    save_json(destination / "best_diagnostics.json", best_diagnostic)
    save_json(destination / "summary.json", {
        "writer": writer_kind, "functional_mode": mode,
        "initial_validation_functional_kl": initial, "best_validation_functional_kl": best,
        "best_epoch": best_epoch, "functional_kl_decreased": best < initial,
        "epochs_completed": len(epoch_summaries), "max_epochs": settings["max_epochs"],
        "stopped_early": stopped_early,
        "early_stopping_triggered": early_stopping_triggered,
        "early_stopping_patience_epochs": settings["early_stopping_patience_epochs"],
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "optimizer_steps": total_optimizer_steps,
        "sample_exposures": total_exposures,
        "effective_batch_size": settings["effective_batch_size"],
        "true_epoch_sampler_without_replacement": True,
        "clipped_optimizer_steps": total_clipped_steps,
        "clip_rate": total_clipped_steps / total_optimizer_steps,
        "receiver_frozen": True, "teacher_no_grad": True,
        "depth_projection_frozen": True,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "stage_b_loss": f"only_{mode}_full_vocabulary_teacher_to_student_kl",
        "kv_reconstruction_diagnostic_only": True,
        "hidden_trajectory_diagnostic_only": True,
        "initialized_from": str(stage_a_checkpoint),
        "gold_label_used": False, "seconds": time.perf_counter() - started,
    })
    del writer, model
    torch.cuda.empty_cache()


def _gradient_stats(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    present = [gradient for gradient in gradients if gradient is not None]
    return {
        "norm": sum(gradient.float().square().sum().item() for gradient in present) ** 0.5,
        "missing_parameter_count": len(gradients) - len(present),
        "nonfinite_parameter_count": sum(not torch.isfinite(gradient).all().item() for gradient in present),
        "zero_parameter_count": sum(gradient.abs().max().item() == 0.0 for gradient in present),
    }


def gradient_audit(cfg, writer_kind: str, checkpoint: str, functional_mode: str):
    modes = cfg["stage_b"]["modes"] if functional_mode == "both" else [functional_mode]
    samples = load_training_split(cfg, "train")[:cfg["gradient_audit_samples"]]
    writer = make_writer(writer_kind, cfg).to(cuda()).train()
    load_writer(checkpoint, writer)
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    parameters = tuple(parameter for parameter in writer.parameters() if parameter.requires_grad)
    rows = []
    try:
        for sample in samples:
            for mode in modes:
                loss = functional_loss(cfg, model, writer, sample, "train", mode)
                rows.append({
                    "sample_id": sample["id"], "functional_mode": mode,
                    "functional_kl": loss.item(), "gradient": _gradient_stats(loss, parameters),
                })
                progress(f"gradient audit {mode} sample={sample['id']}")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("gradient audit found Receiver parameter gradients")
        failures = [row for row in rows if row["gradient"]["missing_parameter_count"] or row["gradient"]["nonfinite_parameter_count"]]
        if failures:
            raise RuntimeError(f"functional gradient audit failed: {failures}")
    finally:
        del writer, model
        torch.cuda.empty_cache()
    save_json(Path(cfg["work_dir"]) / "artifacts/gradient_audit/summary.json", {
        "writer": writer_kind, "checkpoint": checkpoint, "sample_count": len(samples),
        "functional_modes": modes, "rows": rows, "receiver_gradient_count": 0,
        "kv_hidden_losses_present_in_training_graph": False, "gold_label_used": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--writer", choices=("d0", "d1", "d2", "full28", "full28_head8"), required=True)
    parser.add_argument("--stage", choices=("a", "b", "gradient_audit"), required=True)
    parser.add_argument("--functional-mode", choices=("final", "all", "both"))
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    if args.stage == "a":
        stage_a(cfg, args.writer, args.overfit)
    elif args.stage == "b":
        if args.functional_mode not in {"final", "all"}:
            raise ValueError("Stage B requires --functional-mode final or all")
        stage_b(cfg, args.writer, args.functional_mode, args.overfit)
    else:
        if not args.checkpoint or not args.functional_mode:
            raise ValueError("gradient_audit requires --checkpoint and --functional-mode")
        gradient_audit(cfg, args.writer, args.checkpoint, args.functional_mode)


if __name__ == "__main__":
    main()
