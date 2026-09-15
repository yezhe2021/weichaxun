from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from anchor_writer import make_writer
from common import (
    Stores, answer_loss, device, evaluate_conditions, generate, load_json,
    load_reader, native_memory, progress, rows_for, save_json, save_lora,
    seed_all, source_mode, summarize, token_f1, write_records, writer_memory,
)
from receiver_anchor_injection import AnchorInjection


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_assets(cfg, mode):
    v2 = Path(cfg["v2_dir"])
    r1 = Path(cfg["r1_dir"])
    return {
        "scales": v2 / "artifacts" / mode / "scales.pt",
        "a1": v2 / "artifacts" / mode / "a1" / "best.pt",
        "alignment": v2 / "artifacts" / mode / "alignment_metadata.json",
        "full36_reader": r1 / "artifacts" / source_mode(mode) / "sparse_reader" / "best.pt",
        "q35_cache": v2 / "cache" / mode,
        "native4_cache": r1 / "cache" / source_mode(mode),
    }


def audit(cfg, mode):
    assets = required_assets(cfg, mode)
    missing = [f"{key}: {path}" for key, path in assets.items() if not path.exists()]
    if missing:
        raise RuntimeError("missing reused assets: " + "; ".join(missing))
    if cfg["source_layers"] != [3, 7, 11, 15, 19, 23, 27, 31]:
        raise RuntimeError("unexpected Qwen3.5 full-attention layers")
    if cfg["anchor_layers"] != [3, 8, 12, 17, 21, 26, 30, 35]:
        raise RuntimeError("unexpected Qwen3-4B anchor layers")
    writer = make_writer(cfg, mode, "identity")
    if writer.zero_check() != 0.0:
        raise RuntimeError("AnchorWriter(0) != 0")
    if any(getattr(module, "bias", None) is not None for module in writer.modules()):
        raise RuntimeError("AnchorWriter contains bias")
    del writer
    report = {
        "passed": True,
        "asset_paths": {key: str(path) for key, path in assets.items()},
        "checkpoint_sha256": {
            key: sha256(path) for key, path in assets.items() if path.is_file() and path.suffix == ".pt"
        },
        "source_layers": cfg["source_layers"],
        "anchor_layers": cfg["anchor_layers"],
        "external_memory_layers": 8,
        "non_anchor_external_slots": 0,
        "writer_parameters": "16 independent bias-free 1024x1024 matrices only",
        "deleted_modules": [
            "36x8 depth mapper", "learned depth residual", "8-to-36 interpolation",
            "target-layer calibration", "non-anchor prediction", "dummy KV", "zero placeholders",
        ],
        "question_enters_sender": False,
        "hard_gates_enforced": False,
    }
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "protocol_audit.json", report)
    progress(f"{mode}: asset and protocol audit passed")


def controller_for(model):
    # Full36 is needed only for the explicit upper-bound control. Anchor conditions
    # still set exactly eight dictionary entries and assert exact usage.
    return AnchorInjection(model, range(36))


def upper_bound(cfg, mode):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    r1, model, tok, reader_path = load_reader(cfg, mode)
    controller = controller_for(model)
    f0 = make_writer(cfg, mode, "a1").eval()
    conditions = [
        {"key": "question_only", "compact": True, "lora": True},
        {"key": "native4_full36", "source": "native4", "anchor": False, "lora": True},
        {"key": "native4_anchor8", "source": "native4", "anchor": True, "lora": True},
        {"key": "native4_anchor8_shuffled", "source": "native4", "anchor": True, "kind": "shuffled", "lora": True},
        {"key": "native4_anchor8_lora_off", "source": "native4", "anchor": True, "lora": False},
        {"key": "q35_a1_anchor_f0", "source": "q35", "writer": f0, "lora": True},
        {"key": "q35_a1_anchor_f0_shuffled", "source": "q35", "writer": f0, "kind": "shuffled", "lora": True},
        {"key": "q35_a1_anchor_none", "compact": False, "lora": True},
    ]
    records = evaluate_conditions(cfg, rows, store, r1, model, tok, controller, conditions)
    summary = summarize(records)
    gap = summary["native4_anchor8"]["token_f1"] - summary["question_only"]["token_f1"]
    train_required = gap < cfg["reader_anchor_min_f1_gap"]
    if mode == "smoke":
        train_required = True  # exercise and verify the conditional training branch
    decision = {
        "reader_checkpoint": str(reader_path),
        "native4_anchor8_minus_question_only_f1": gap,
        "minimum_interpretable_gap": cfg["reader_anchor_min_f1_gap"],
        "train_anchor_reader": train_required,
        "smoke_forces_branch_test": mode == "smoke",
    }
    root = Path(cfg["work_dir"]) / "artifacts" / mode / "upper_bound"
    write_records(root, records, summary)
    save_json(root / "reader_decision.json", decision)
    progress(f"{mode}: Anchor8 upper bound completed; train_reader={train_required}")


@torch.no_grad()
def validation_nll(cfg, rows, store, r1, model, tok, controller, split="validation"):
    model.eval()
    values = []
    for sample in rows[split]:
        memory = native_memory(cfg, store, split, sample, anchor=True)
        values.append(answer_loss(cfg, model, tok, controller, sample, memory).item())
    return sum(values) / len(values)


def reader_generation_snapshot(cfg, rows, store, r1, model, tok, controller, update):
    result = []
    limit = min(cfg["generation_eval_samples"], len(rows["validation"]))
    for sample in rows["validation"][:limit]:
        for kind in ("correct", "shuffled"):
            memory = native_memory(cfg, store, "validation", sample, anchor=True, kind=kind)
            prediction = generate(cfg, model, tok, controller, sample, memory)
            result.append({
                "update": update, "id": sample["id"], "kind": kind,
                "answer": sample["answer"], "prediction": prediction,
                "token_f1": token_f1(prediction, sample["answer"]),
            })
    return result


def train_anchor_reader(cfg, mode):
    decision_path = Path(cfg["work_dir"]) / "artifacts" / mode / "upper_bound" / "reader_decision.json"
    decision = load_json(decision_path)
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "anchor_reader"
    out.mkdir(parents=True, exist_ok=True)
    if not decision["train_anchor_reader"]:
        save_json(out / "summary.json", {
            "completed": True, "trained": False,
            "reason": "current Full36 Reader retained an interpretable Native4 Anchor8 upper bound",
            "checkpoint": decision["reader_checkpoint"],
        })
        progress(f"{mode}: dedicated Anchor8 Reader not required")
        return
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    r1, model, tok, initial = load_reader(cfg, mode, trainable=True)
    controller = controller_for(model)
    parameters = r1.lora_parameters(model)
    optimizer = AdamW(parameters, lr=cfg["reader_lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["reader_updates"]
    grad_acc = cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]
    interval = 1 if mode == "smoke" else cfg["eval_interval"]
    samples, cursor, epoch = list(rows["train"]), 0, 0
    history, evaluations, generations = [], [], []
    best, stale = float("inf"), 0
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        losses = []
        model.train()
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + 1000 + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            memory = native_memory(cfg, store, "train", sample, anchor=True)
            loss = answer_loss(cfg, model, tok, controller, sample, memory)
            scaler.scale(loss / grad_acc).backward()
            losses.append(loss.detach().item())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"]).item()
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        history.append({"update": update, "answer_nll": sum(losses) / len(losses), "gradient_norm": grad_norm})
        if update % interval == 0 or update == maximum:
            value = validation_nll(cfg, rows, store, r1, model, tok, controller)
            selected = value < best
            if selected:
                best, stale = value, 0
                save_lora(r1, model, out / "best.pt", update=update, validation_answer_nll=value)
            else:
                stale += 1
            evaluations.append({"update": update, "validation_answer_nll": value, "selected": selected})
            save_json(out / "history.json", history)
            save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}: Anchor8 Reader {update}/{maximum}")
        if update % cfg["generation_interval"] == 0 or update == maximum:
            generations.extend(reader_generation_snapshot(cfg, rows, store, r1, model, tok, controller, update))
            save_json(out / "generation_snapshots.json", generations)
        if update >= cfg["early_stop_after"] and stale >= cfg["early_stop_patience"]:
            progress(f"{mode}: Anchor8 Reader early stop at {update}")
            break
    save_json(out / "summary.json", {
        "completed": True, "trained": True, "initialized_from": str(initial),
        "only_loss": "gold_answer_teacher_forcing_cross_entropy",
        "best_validation_answer_nll": best, "updates_completed": history[-1]["update"],
    })


def chosen_reader_checkpoint(cfg, mode):
    summary = load_json(Path(cfg["work_dir"]) / "artifacts" / mode / "anchor_reader" / "summary.json")
    if summary["trained"]:
        return Path(cfg["work_dir"]) / "artifacts" / mode / "anchor_reader" / "best.pt", "anchor8_reader"
    return Path(summary["checkpoint"]), "full36_reader"


def apply_rope(x, positions, theta):
    inverse = 1.0 / (
        float(theta) ** (torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1])
    )
    frequency = torch.outer(torch.tensor(positions, device=x.device).float(), inverse)
    embedding = torch.cat((frequency, frequency), -1)[:, None]
    half = x.shape[-1] // 2
    return x.float() * embedding.cos() + torch.cat((-x[..., half:], x[..., :half]), -1).float() * embedding.sin()


@torch.no_grad()
def representation_metrics(cfg, writer, store, samples, split="validation"):
    writer.eval()
    all_rows = []
    for sample in samples:
        source_k, source_v, mask = store.memory(split, "q35", sample)
        target_k, target_v, _ = store.memory(split, "4b", sample)
        pred_k, pred_v = writer(source_k.to(device()), source_v.to(device()))
        gold_k = target_k[cfg["anchor_layers"]].to(device()).float()
        gold_v = target_v[cfg["anchor_layers"]].to(device()).float()
        valid = mask[0].bool().to(device())
        query = store.query(split, sample["id"])
        sample_rows = []
        for slot, target_layer in enumerate(cfg["anchor_layers"]):
            pk, pv = pred_k[slot, valid].float(), pred_v[slot, valid].float()
            gk, gv = gold_k[slot, valid], gold_v[slot, valid]
            scale_k, scale_v = writer.scale_target_k[slot], writer.scale_target_v[slot]
            pks, gks = pk.reshape(-1, 1024) / scale_k, gk.reshape(-1, 1024) / scale_k
            pvs, gvs = pv.reshape(-1, 1024) / scale_v, gv.reshape(-1, 1024) / scale_v
            q = query["query"][target_layer].to(device())
            positions = [p for p, keep in zip(sample["selected_position_ids"], valid.tolist()) if keep]
            pkr, gkr = apply_rope(pk, positions, cfg["rope_theta"]), apply_rope(gk, positions, cfg["rope_theta"])
            pkr, gkr = pkr.repeat_interleave(4, 1), gkr.repeat_interleave(4, 1)
            q = apply_rope(q, query["query_position_ids"], cfg["rope_theta"])
            logits_p = torch.einsum("qhd,thd->hqt", q, pkr) / math.sqrt(128)
            logits_g = torch.einsum("qhd,thd->hqt", q, gkr) / math.sqrt(128)
            attn_p, attn_g = logits_p.softmax(-1), logits_g.softmax(-1)
            out_p = torch.einsum("hqt,thd->qhd", attn_p, pv.repeat_interleave(4, 1))
            out_g = torch.einsum("hqt,thd->qhd", attn_g, gv.repeat_interleave(4, 1))
            sample_rows.append({
                "target_layer": target_layer,
                "k_standardized_nmse": ((pks - gks).square().mean() / gks.square().mean().clamp_min(1e-8)).item(),
                "v_standardized_nmse": ((pvs - gvs).square().mean() / gvs.square().mean().clamp_min(1e-8)).item(),
                "k_cosine": F.cosine_similarity(pk.flatten(), gk.flatten(), 0).item(),
                "v_cosine": F.cosine_similarity(pv.flatten(), gv.flatten(), 0).item(),
                "route_kl": (attn_g * (attn_g.clamp_min(1e-12).log() - attn_p.clamp_min(1e-12).log())).sum(-1).mean().item(),
                "attention_output_cosine": F.cosine_similarity(out_p.flatten(), out_g.flatten(), 0).item(),
            })
        all_rows.append(sample_rows)
    result = []
    for slot, layer in enumerate(cfg["anchor_layers"]):
        row = {"target_layer": layer}
        for key in (
            "k_standardized_nmse", "v_standardized_nmse", "k_cosine", "v_cosine",
            "route_kl", "attention_output_cosine",
        ):
            row[key] = sum(sample[slot][key] for sample in all_rows) / len(all_rows)
        result.append(row)
    writer.train()
    return result


@torch.no_grad()
def writer_validation_nll(cfg, writer, rows, store, model, tok, controller):
    writer.eval(); model.eval()
    values = []
    for sample in rows["validation"]:
        memory = writer_memory(cfg, writer, store, "validation", sample)
        values.append(answer_loss(cfg, model, tok, controller, sample, memory).item())
    writer.train()
    return sum(values) / len(values)


@torch.no_grad()
def writer_generation_snapshot(cfg, writer, rows, store, model, tok, controller, update):
    writer.eval(); output = []
    limit = min(cfg["generation_eval_samples"], len(rows["validation"]))
    for sample in rows["validation"][:limit]:
        for kind in ("correct", "shuffled"):
            memory = writer_memory(cfg, writer, store, "validation", sample, kind)
            prediction = generate(cfg, model, tok, controller, sample, memory)
            output.append({
                "update": update, "id": sample["id"], "kind": kind,
                "answer": sample["answer"], "prediction": prediction,
                "token_f1": token_f1(prediction, sample["answer"]),
            })
    writer.train()
    return output


def train_writer(cfg, mode, variant):
    if variant not in ("f1", "f2"):
        raise ValueError(variant)
    seed_all(cfg["seed"] + (100 if variant == "f1" else 200))
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    checkpoint, reader_protocol = chosen_reader_checkpoint(cfg, mode)
    r1, model, tok, _ = load_reader(cfg, mode, checkpoint=checkpoint)
    controller = controller_for(model)
    initialization = "identity" if variant == "f1" else "a1"
    writer = make_writer(cfg, mode, initialization).train()
    optimizer = AdamW(writer.parameters(), lr=cfg["writer_lr"], weight_decay=cfg["weight_decay"])
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["writer_updates"]
    grad_acc = cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]
    interval = 1 if mode == "smoke" else cfg["eval_interval"]
    samples, cursor, epoch = list(rows["train"]), 0, 0
    history, evaluations, generations = [], [], []
    best, stale = float("inf"), 0
    out = Path(cfg["work_dir"]) / "artifacts" / mode / f"anchor_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        losses = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + (3000 if variant == "f1" else 5000) + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            memory = writer_memory(cfg, writer, store, "train", sample)
            loss = answer_loss(cfg, model, tok, controller, sample, memory)
            (loss / grad_acc).backward()
            losses.append(loss.detach().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]).item()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        history.append({"update": update, "answer_nll": sum(losses) / len(losses), "gradient_norm": grad_norm})
        if update % interval == 0 or update == maximum:
            nll = writer_validation_nll(cfg, writer, rows, store, model, tok, controller)
            rep = representation_metrics(cfg, writer, store, rows["validation"][:min(8, len(rows["validation"]))])
            selected = nll < best
            if selected:
                best, stale = nll, 0
                torch.save({
                    "writer": writer.state_dict(), "update": update,
                    "validation_answer_nll": nll, "initialization": initialization,
                }, out / "best.pt")
            else:
                stale += 1
            evaluations.append({
                "update": update, "validation_answer_nll": nll,
                "selected": selected, "per_anchor_representation": rep,
            })
            save_json(out / "history.json", history)
            save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}: Anchor-{variant.upper()} {update}/{maximum}")
        if update % cfg["generation_interval"] == 0 or update == maximum:
            generations.extend(writer_generation_snapshot(cfg, writer, rows, store, model, tok, controller, update))
            save_json(out / "generation_snapshots.json", generations)
        if update >= cfg["early_stop_after"] and stale >= cfg["early_stop_patience"]:
            progress(f"{mode}: Anchor-{variant.upper()} early stop at {update}")
            break
    save_json(out / "summary.json", {
        "completed": True, "variant": variant, "initialization": initialization,
        "reader_protocol": reader_protocol,
        "only_loss": "gold_answer_teacher_forcing_cross_entropy",
        "best_validation_answer_nll": best, "updates_completed": history[-1]["update"],
        "trainable_parameters": ["feature_k", "feature_v"],
    })


def load_trained_writer(cfg, mode, variant):
    initialization = "identity" if variant == "f1" else "a1"
    writer = make_writer(cfg, mode, initialization)
    state = torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / f"anchor_{variant}" / "best.pt",
        map_location="cpu", weights_only=False,
    )["writer"]
    writer.load_state_dict(state)
    return writer.eval()


def imported_references(cfg, mode):
    if mode == "smoke":
        return {}
    v2 = load_json(Path(cfg["v2_dir"]) / "artifacts" / "development" / "evaluation" / "summary.json")
    return {
        "q35_a2_full36": v2.get("q35_v2_a2"),
        "q35_b_full36": v2.get("q35_v2_b"),
        "previous_8b_f2": v2.get("previous_writer8"),
    }


def final_evaluate(cfg, mode):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    checkpoint, reader_protocol = chosen_reader_checkpoint(cfg, mode)
    r1, model, tok, _ = load_reader(cfg, mode, checkpoint=checkpoint)
    controller = controller_for(model)
    f0, f1, f2 = make_writer(cfg, mode, "a1").eval(), load_trained_writer(cfg, mode, "f1"), load_trained_writer(cfg, mode, "f2")
    conditions = [
        {"key": f"native4_anchor8_{reader_protocol}", "source": "native4", "anchor": True, "lora": True, "reader_protocol": reader_protocol},
        {"key": f"q35_anchor_f0_{reader_protocol}", "source": "q35", "writer": f0, "lora": True, "reader_protocol": reader_protocol},
        {"key": "q35_anchor_f1", "source": "q35", "writer": f1, "lora": True, "reader_protocol": reader_protocol},
        {"key": "q35_anchor_f2", "source": "q35", "writer": f2, "lora": True, "reader_protocol": reader_protocol},
        {"key": "q35_anchor_f2_shuffled", "source": "q35", "writer": f2, "kind": "shuffled", "lora": True, "reader_protocol": reader_protocol},
        {"key": "q35_anchor_f2_no_memory", "compact": False, "lora": True, "reader_protocol": reader_protocol},
    ]
    records = evaluate_conditions(cfg, rows, store, r1, model, tok, controller, conditions)
    upper_root = Path(cfg["work_dir"]) / "artifacts" / mode / "upper_bound"
    records = load_json(upper_root / "per_sample.json") + records
    summary = summarize(records)
    summary["imported_full36_references"] = imported_references(cfg, mode)
    correct = summary["q35_anchor_f2"]
    summary["anchor_f2_dependence"] = {
        "correct_minus_shuffled_em": correct["em"] - summary["q35_anchor_f2_shuffled"]["em"],
        "correct_minus_no_memory_em": correct["em"] - summary["q35_anchor_f2_no_memory"]["em"],
        "correct_minus_shuffled_f1": correct["token_f1"] - summary["q35_anchor_f2_shuffled"]["token_f1"],
        "correct_minus_no_memory_f1": correct["token_f1"] - summary["q35_anchor_f2_no_memory"]["token_f1"],
    }
    root = Path(cfg["work_dir"]) / "artifacts" / mode / "evaluation"
    write_records(root, records, summary)
    save_json(root / "completion.json", {
        "completed": True, "hard_gate": None,
        "reader_protocol_for_functional_training": reader_protocol,
        "non_anchor_external_slots": 0,
    })
    progress(f"{mode}: final evaluation completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("action", choices=("audit", "upper_bound", "reader", "f1", "f2", "evaluate"))
    args = parser.parse_args()
    cfg = load_json(args.config)
    actions = {
        "audit": audit, "upper_bound": upper_bound, "reader": train_anchor_reader,
        "f1": lambda c, m: train_writer(c, m, "f1"),
        "f2": lambda c, m: train_writer(c, m, "f2"), "evaluate": final_evaluate,
    }
    actions[args.action](cfg, args.mode)


if __name__ == "__main__":
    main()
