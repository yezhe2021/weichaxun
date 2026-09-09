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
from receiver import answer_ce
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


def representation_loss(pred_k, pred_v, target_k, target_v):
    rows, total = [], pred_k.new_zeros(())
    for label, prediction, target in (("k", pred_k, target_k), ("v", pred_v, target_v)):
        difference = (prediction.float() - target.float()).square().mean(dim=(1, 2, 3))
        denominator = target.float().square().mean(dim=(1, 2, 3)).clamp_min(1e-8)
        nmse = difference / denominator
        cosine = F.cosine_similarity(prediction.float().flatten(1), target.float().flatten(1), dim=1)
        total = total + nmse.mean() + (1.0 - cosine).mean()
        rows.append((label, nmse, cosine))
    metrics = []
    for layer in range(pred_k.shape[0]):
        metrics.append({
            "layer": layer,
            "k_nmse": rows[0][1][layer].item(),
            "k_cosine": rows[0][2][layer].item(),
            "v_nmse": rows[1][1][layer].item(),
            "v_cosine": rows[1][2][layer].item(),
        })
    return total, metrics


@torch.no_grad()
def validate_representation(cfg, writer, samples, split="validation"):
    writer.eval()
    losses, layers = [], []
    for sample_index, sample in enumerate(samples):
        source_k, source_v, target_k, target_v = cached_pair(cfg, split, sample, cuda())
        positions = token_subset(source_k.shape[1], cfg["stage_a"]["token_sample_size"], cfg["seed"] + sample_index, cuda())
        pred_k, pred_v = writer(source_k[:, positions], source_v[:, positions])
        loss, metrics = representation_loss(pred_k, pred_v, target_k[:, positions], target_v[:, positions])
        losses.append(loss.item())
        layers.append(metrics)
    aggregate = []
    for layer in range(cfg["target_layers"]):
        aggregate.append({
            "layer": layer,
            **{name: sum(row[layer][name] for row in layers) / len(layers) for name in ("k_nmse", "k_cosine", "v_nmse", "v_cosine")},
        })
    writer.train()
    return sum(losses) / len(losses), aggregate


def stage_a(cfg, writer_kind: str, overfit: bool = False):
    train = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "manifests" / "train.jsonl")
    validation = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "manifests" / "validation.jsonl")
    if overfit:
        train = train[: cfg["overfit_samples"]]
        validation, validation_split = train, "train"
    else:
        validation_split = "validation"
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
        loss, _ = representation_loss(pred_k, pred_v, target_k[:, positions], target_v[:, positions])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Stage A loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        invalid = [
            name for name, parameter in writer.named_parameters()
            if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
        ]
        if invalid:
            raise RuntimeError(f"Writer parameters missing finite Stage A gradients: {invalid}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), settings["gradient_clip"])
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite Stage A gradient")
        optimizer.step()
        history.append({"update": update, "loss": loss.item(), "gradient_norm": gradient_norm.item()})
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
        "writer": writer_kind,
        "overfit_code_validation": overfit,
        "initial_validation_loss": initial,
        "best_validation_loss": best,
        "loss_decreased": best < initial,
        "updates": updates,
        "seconds": time.perf_counter() - started,
        "hard_representation_gate_used": False,
    })
    del writer
    torch.cuda.empty_cache()


@torch.no_grad()
def validate_functional(cfg, model, writer, samples, split="validation"):
    writer.eval()
    values = []
    for sample in samples:
        source_k, source_v, _, _ = cached_pair(cfg, split, sample, cuda(), target_required=False)
        pred_k, pred_v = writer(source_k, source_v)
        loss, _, _ = answer_ce(model, sample, pred_k, pred_v)
        values.append(loss.item())
    writer.train()
    return sum(values) / len(values)


def stage_b(cfg, writer_kind: str, overfit: bool = False):
    train = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "manifests" / "train.jsonl")
    validation = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "manifests" / "validation.jsonl")
    if overfit:
        train = train[: cfg["overfit_samples"]]
        validation, validation_split = train, "train"
    else:
        validation_split = "validation"
    root = Path(cfg["work_dir"]) / "checkpoints" / ("overfit" if overfit else "quick") / writer_kind
    writer = make_writer(writer_kind, cfg).to(cuda())
    load_writer(str(root / "stage_a" / "best.pt"), writer)
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
        source_k, source_v, target_k, target_v = cached_pair(cfg, "train", sample, cuda(), target_required=True)
        pred_k, pred_v = writer(source_k, source_v)
        ce, _, _ = answer_ce(model, sample, pred_k, pred_v)
        positions = token_subset(source_k.shape[1], settings["reconstruction_token_sample_size"], cfg["seed"] + update, cuda())
        reconstruction, _ = representation_loss(pred_k[:, positions], pred_v[:, positions], target_k[:, positions], target_v[:, positions])
        loss = ce + settings["reconstruction_weight"] * reconstruction
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Stage B loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen Receiver unexpectedly accumulated parameter gradients")
        missing = [name for name, parameter in writer.named_parameters() if parameter.requires_grad and parameter.grad is None]
        if missing:
            raise RuntimeError(f"Writer parameters missing Stage B gradients: {missing}")
        invalid = [
            name for name, parameter in writer.named_parameters()
            if parameter.requires_grad and not torch.isfinite(parameter.grad).all()
        ]
        if invalid:
            raise RuntimeError(f"Writer parameters have non-finite Stage B gradients: {invalid}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), settings["gradient_clip"])
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite Stage B gradient")
        optimizer.step()
        history.append({
            "update": update,
            "total_loss": loss.item(),
            "answer_ce": ce.item(),
            "reconstruction": reconstruction.item(),
            "gradient_norm": gradient_norm.item(),
        })
        if update % settings["validation_every"] == 0 or update == updates:
            nll = validate_functional(cfg, model, writer, validation, validation_split)
            selected = nll < best
            evaluations.append({"update": update, "validation_answer_nll": nll, "selected": selected})
            if selected:
                best = nll
                save_writer(str(destination / "best.pt"), writer, {"stage": "B", "update": update, "validation_answer_nll": nll})
            progress(f"Stage B {writer_kind} {update}/{updates} validation_nll={nll:.6f}")
    save_json(destination / "history.json", history)
    save_json(destination / "evaluations.json", evaluations)
    save_json(destination / "summary.json", {
        "writer": writer_kind,
        "overfit_code_validation": overfit,
        "initial_validation_answer_nll": initial,
        "best_validation_answer_nll": best,
        "answer_nll_decreased": best < initial,
        "updates": updates,
        "receiver_frozen": True,
        "reader_lora_used": False,
        "seconds": time.perf_counter() - started,
    })
    del writer, model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--writer", choices=("d0", "d1", "d2"), required=True)
    parser.add_argument("--stage", choices=("a", "b", "both"), default="both")
    parser.add_argument("--overfit", action="store_true")
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    if args.stage in {"a", "both"}:
        stage_a(cfg, args.writer, args.overfit)
    if args.stage in {"b", "both"}:
        stage_b(cfg, args.writer, args.overfit)


if __name__ == "__main__":
    main()
