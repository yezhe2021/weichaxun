from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_model, log, run_root, save_json, save_tensor, seed_all
from pairs import load_pair, rows
from protocol import final_logits
from translator import NativeKVTranslator, direct_nearest, representation_loss


def checkpoint_path(cfg, arm, name): return run_root(cfg) / "checkpoints" / arm / f"{name}.pt"


def save_checkpoint(cfg, arm, name, module, step, representation):
    save_tensor(checkpoint_path(cfg, arm, name), {"signature": cfg["signature"], "arm": arm,
        "window": module.window, "step": step, "validation_representation": representation,
        "state": {key: value.detach().cpu() for key, value in module.state_dict().items()}})


def load_checkpoint(cfg, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]: raise RuntimeError("Checkpoint signature mismatch")
    module = NativeKVTranslator(payload["window"]).cuda(); module.load_state_dict(payload["state"])
    return module, payload


def batch(cfg, split, selected_rows):
    payloads = [load_pair(cfg, split, row) for row in selected_rows]
    names = ("source_k", "source_v", "target_k", "target_v")
    return tuple(torch.stack([payload[name] for payload in payloads]).cuda() for name in names)


@torch.no_grad()
def representation_metrics(cfg, module, split, selected_rows):
    module.eval(); totals = {key: 0.0 for key in ("loss", "k_nmse", "v_nmse", "k_cosine", "v_cosine")}
    for begin in range(0, len(selected_rows), cfg["batch_size"]):
        sk, sv, tk, tv = batch(cfg, split, selected_rows[begin:begin + cfg["batch_size"]])
        with torch.amp.autocast("cuda", dtype=torch.float16): pk, pv = module(sk, sv)
        loss, detail = representation_loss(pk, pv, tk, tv); count = sk.shape[0]
        totals["loss"] += loss.item() * count
        for key, value in detail.items(): totals[key] += value.item() * count
    return {key: value / len(selected_rows) for key, value in totals.items()}


def train_stage_a(cfg, arm, window):
    seed_all(cfg["seed"] + window); train_rows, validation_rows = rows(cfg, "train"), rows(cfg, "validation")
    module = NativeKVTranslator(window).cuda()
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["stage_a_learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    root = run_root(cfg) / "training" / arm / "stage_a"; root.mkdir(parents=True, exist_ok=True)
    stream = (root / "steps.jsonl").open("w", encoding="utf-8")
    candidates, step, clipped, started = [], 0, 0, time.monotonic()
    try:
        for epoch in range(1, cfg["stage_a_epochs"] + 1):
            order = list(range(len(train_rows))); random.Random(cfg["seed"] + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                indices = order[begin:begin + cfg["batch_size"]]
                selected = [train_rows[index] for index in indices]
                sk, sv, tk, tv = batch(cfg, "train", selected)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16): pk, pv = module(sk, sv)
                loss, detail = representation_loss(pk, pv, tk, tv)
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite Stage-A loss")
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if not torch.isfinite(norm): raise RuntimeError("Invalid Stage-A gradient")
                clipped += int(norm.item() > cfg["clip"]); scaler.step(optimizer); scaler.update(); step += 1
                record = {"epoch": epoch, "step": step, "loss": loss.item(),
                          **{key: value.item() for key, value in detail.items()},
                          "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
                stream.write(json.dumps(record) + "\n"); stream.flush()
                if step % 16 == 0: log(f"Stage A {arm} {step}/{cfg['stage_a_steps']} loss={loss.item():.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["stage_a_steps"]:
                    metrics = representation_metrics(cfg, module, "validation", validation_rows)
                    name = f"step_{step}"; save_checkpoint(cfg, arm, name, module, step, metrics)
                    candidates.append({"step": step, "path": str(checkpoint_path(cfg, arm, name)),
                                       "representation": metrics})
                    save_json(root / "candidates.json", candidates)
                    log(f"Stage A {arm} validation step={step} rep={metrics['loss']:.6f}")
                if step >= cfg["stage_a_steps"]: break
            if step >= cfg["stage_a_steps"]: break
    finally:
        stream.close()
    summary = {"arm": arm, "window": window, "steps": step, "clip_rate": clipped / step,
               "candidates": candidates, "seconds": time.monotonic() - started}
    save_json(root / "summary.json", summary)
    del module, optimizer, scaler; torch.cuda.empty_cache(); return summary


def teacher_path(cfg, split, row): return run_root(cfg) / "teacher_cache" / split / f"{row['id']}.pt"


@torch.no_grad()
def prepare_teachers(cfg, model):
    positions = torch.arange(33, device="cuda")
    for split in ("train", "validation", "test"):
        selected_rows = rows(cfg, split)
        for number, row in enumerate(selected_rows, 1):
            path = teacher_path(cfg, split, row)
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload.get("signature") == cfg["signature"]: continue
            pair = load_pair(cfg, split, row); metadata = pair["metadata"]
            logits = final_logits(model, metadata["qwen_suffix"], pair["target_k"].cuda(), pair["target_v"].cuda(),
                                  positions=positions, suffix_start=33)
            ids = torch.tensor(metadata["qwen_choice_ids"], device=logits.device)
            save_tensor(path, {"signature": cfg["signature"], "choice_logits": logits[ids].cpu(),
                               "full_logits": logits.cpu() if split != "train" else torch.empty(0)})
            if number % 32 == 0: log(f"Oracle teacher {split}: {number}/{len(selected_rows)}")


def translated(module, pair, direct=False):
    sk, sv = pair["source_k"].cuda()[None], pair["source_v"].cuda()[None]
    if direct: return tuple(x[0] for x in direct_nearest(sk, sv))
    with torch.amp.autocast("cuda", dtype=torch.float16): pk, pv = module(sk, sv)
    return pk[0], pv[0]


def choice_loss(student, teacher_choice, choice_ids, temperature=1.0):
    ids = torch.tensor(choice_ids, device=student.device)
    s = F.log_softmax(student[ids].float() / temperature, -1)
    t = F.log_softmax(teacher_choice.to(student.device).float() / temperature, -1)
    return F.kl_div(s, t.detach(), reduction="sum", log_target=True) * temperature**2


@torch.no_grad()
def functional_validation(cfg, model, module, selected_rows, split, direct=False):
    if module is not None: module.eval()
    totals = {key: 0.0 for key in ("choice_kl", "accuracy", "oracle_accuracy", "oracle_agreement")}
    for row in selected_rows:
        pair, teacher = load_pair(cfg, split, row), torch.load(teacher_path(cfg, split, row), map_location="cpu", weights_only=True)
        pk, pv = translated(module, pair, direct)
        metadata = pair["metadata"]
        student = final_logits(model, metadata["qwen_suffix"], pk, pv,
                               positions=torch.arange(33, device="cuda"), suffix_start=33)
        ids = metadata["qwen_choice_ids"]
        prediction = int(student[ids].argmax()); oracle_prediction = int(teacher["choice_logits"].argmax())
        totals["choice_kl"] += choice_loss(student, teacher["choice_logits"], ids, cfg["temperature"]).item()
        totals["accuracy"] += prediction == metadata["gold_index"]
        totals["oracle_accuracy"] += oracle_prediction == metadata["gold_index"]
        totals["oracle_agreement"] += prediction == oracle_prediction
    return {key: value / len(selected_rows) for key, value in totals.items()}


def select_stage_a(cfg, model, summary):
    validation_rows = rows(cfg, "validation"); evaluated = []
    for candidate in summary["candidates"]:
        module, _ = load_checkpoint(cfg, candidate["path"])
        metrics = functional_validation(cfg, model, module, validation_rows, "validation")
        evaluated.append({**candidate, "functional": metrics}); del module; torch.cuda.empty_cache()
    best = min(evaluated, key=lambda item: (item["functional"]["choice_kl"], item["step"]))
    module, payload = load_checkpoint(cfg, best["path"])
    save_checkpoint(cfg, summary["arm"], "best", module, payload["step"], payload["validation_representation"])
    root = run_root(cfg) / "training" / summary["arm"] / "stage_a"
    save_json(root / "functional_candidates.json", evaluated)
    save_json(root / "selection.json", {"criterion": "minimum validation choice KL; gold accuracy diagnostic only", "best": best})
    del module; torch.cuda.empty_cache(); return best


def train_stage_b(cfg, model):
    seed_all(cfg["seed"] + 100)
    module, source_checkpoint = load_checkpoint(cfg, checkpoint_path(cfg, "local5_linear", "best"))
    train_rows, validation_rows = rows(cfg, "train"), rows(cfg, "validation")
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["stage_b_learning_rate"], weight_decay=0)
    # Gradients traverse the frozen FP16 Qwen attention path. The default 65536
    # initial scale overflows on V100 even when the unscaled choice loss is finite.
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=1000000)
    root = run_root(cfg) / "training" / "local5_choice_kl" / "stage_b"; root.mkdir(parents=True, exist_ok=True)
    stream = (root / "steps.jsonl").open("w", encoding="utf-8")
    candidates, step, clipped, started = [], 0, 0, time.monotonic()
    try:
        for epoch in range(1, cfg["stage_b_epochs"] + 1):
            order = list(range(len(train_rows))); random.Random(cfg["seed"] + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                indices = order[begin:begin + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True); losses = []
                for index in indices:
                    row = train_rows[index]; pair = load_pair(cfg, "train", row)
                    teacher = torch.load(teacher_path(cfg, "train", row), map_location="cpu", weights_only=True)
                    pk, pv = translated(module, pair)
                    metadata = pair["metadata"]
                    student = final_logits(model, metadata["qwen_suffix"], pk, pv,
                                           positions=torch.arange(33, device="cuda"), suffix_start=33)
                    loss = choice_loss(student, teacher["choice_logits"], metadata["qwen_choice_ids"], cfg["temperature"])
                    if not torch.isfinite(loss): raise RuntimeError("Nonfinite Stage-B loss")
                    scaler.scale(loss / len(indices)).backward(); losses.append(loss.item())
                scaler.unscale_(optimizer); norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if not torch.isfinite(norm): raise RuntimeError("Invalid Stage-B gradient")
                clipped += int(norm.item() > cfg["clip"]); scaler.step(optimizer); scaler.update(); step += 1
                record = {"epoch": epoch, "step": step, "mean_choice_kl": sum(losses) / len(losses),
                          "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
                stream.write(json.dumps(record) + "\n"); stream.flush()
                if step % 16 == 0: log(f"Stage B local5 choice-KL {step}/{cfg['stage_b_steps']} loss={record['mean_choice_kl']:.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["stage_b_steps"]:
                    metrics = functional_validation(cfg, model, module, validation_rows, "validation")
                    name = f"step_{step}"; save_checkpoint(cfg, "local5_choice_kl", name, module, step, None)
                    candidates.append({"step": step, "path": str(checkpoint_path(cfg, "local5_choice_kl", name)),
                                       "functional": metrics})
                    save_json(root / "candidates.json", candidates)
                    log(f"Stage B validation step={step} choice_KL={metrics['choice_kl']:.6f} accuracy={metrics['accuracy']:.4f}")
                if step >= cfg["stage_b_steps"]: break
            if step >= cfg["stage_b_steps"]: break
    finally:
        stream.close()
    best = min(candidates, key=lambda item: (item["functional"]["choice_kl"], item["step"]))
    selected, payload = load_checkpoint(cfg, best["path"])
    save_checkpoint(cfg, "local5_choice_kl", "best", selected, payload["step"], None)
    summary = {"arm": "local5_choice_kl", "initialized_from": source_checkpoint["step"],
               "steps": step, "clip_rate": clipped / step, "selection_criterion": "validation_choice_kl",
               "best": best, "candidates": candidates, "seconds": time.monotonic() - started}
    save_json(root / "summary.json", summary)
    del module, selected, optimizer, scaler; torch.cuda.empty_cache(); return summary


def full_kl(student, teacher):
    return F.kl_div(F.log_softmax(student.float(), -1), F.log_softmax(teacher.float(), -1),
                    reduction="sum", log_target=True).item()


@torch.no_grad()
def test_condition(cfg, model, pair, predicted, condition):
    pk, pv = predicted; tk, tv = pair["target_k"].cuda(), pair["target_v"].cuda()
    if condition == "oracle": key, value = tk, tv
    elif condition == "native_token0_translated_anchors":
        key, value = pk.clone(), pv.clone(); key[:, 0], value[:, 0] = tk[:, 0], tv[:, 0]
    elif condition == "translated_token0_native_anchors":
        key, value = tk.clone(), tv.clone(); key[:, 0], value[:, 0] = pk[:, 0], pv[:, 0]
    elif condition == "translated_all": key, value = pk, pv
    else: raise ValueError(condition)
    metadata = pair["metadata"]
    return final_logits(model, metadata["qwen_suffix"], key, value,
                        positions=torch.arange(33, device="cuda"), suffix_start=33)


@torch.no_grad()
def final_evaluate_arm(cfg, model, arm, module=None, direct=False, diagnostics=False):
    test_rows = rows(cfg, "test"); conditions = (["translated_all"] if not diagnostics else
        ["oracle", "native_token0_translated_anchors", "translated_token0_native_anchors", "translated_all"])
    totals = {condition: {key: 0.0 for key in
              ("accuracy", "oracle_accuracy", "qwen_full_accuracy", "oracle_agreement", "choice_kl", "full_kl")}
              for condition in conditions}
    rep_totals = {key: 0.0 for key in ("loss", "k_nmse", "v_nmse", "k_cosine", "v_cosine")}
    records = []
    for number, row in enumerate(test_rows, 1):
        pair = load_pair(cfg, "test", row); predicted = translated(module, pair, direct)
        rep_loss, details = representation_loss(predicted[0], predicted[1], pair["target_k"].cuda(), pair["target_v"].cuda())
        rep_totals["loss"] += rep_loss.item()
        for key, value in details.items(): rep_totals[key] += value.item()
        teacher = torch.load(teacher_path(cfg, "test", row), map_location="cpu", weights_only=True)
        metadata, choice_ids = pair["metadata"], pair["metadata"]["qwen_choice_ids"]
        oracle_prediction = int(teacher["choice_logits"].argmax())
        qwen_full_prediction = int(pair["qwen_full_native_logits"][choice_ids].argmax())
        record = {"id": row["id"], "arm": arm, "gold_index": metadata["gold_index"], "conditions": {}}
        for condition in conditions:
            logits = test_condition(cfg, model, pair, predicted, condition)
            prediction = int(logits[choice_ids].argmax())
            values = {"accuracy": float(prediction == metadata["gold_index"]),
                      "oracle_accuracy": float(oracle_prediction == metadata["gold_index"]),
                      "qwen_full_accuracy": float(qwen_full_prediction == metadata["gold_index"]),
                      "oracle_agreement": float(prediction == oracle_prediction),
                      "choice_kl": choice_loss(logits, teacher["choice_logits"], choice_ids, cfg["temperature"]).item(),
                      "full_kl": full_kl(logits, teacher["full_logits"].cuda())}
            record["conditions"][condition] = {"prediction": prediction, **values}
            for key, value in values.items(): totals[condition][key] += value
        records.append(record)
        if number % 16 == 0: log(f"Final {arm}: {number}/{len(test_rows)}")
    metrics = {condition: {key: value / len(test_rows) for key, value in values.items()}
               for condition, values in totals.items()}
    for values in metrics.values():
        values["selection_retention"] = values["oracle_accuracy"] / values["qwen_full_accuracy"]
        values["translation_retention"] = values["accuracy"] / values["oracle_accuracy"] if values["oracle_accuracy"] else None
    result = {"functional": metrics,
              "representation": {key: value / len(test_rows) for key, value in rep_totals.items()},
              "sample_count": len(test_rows)}
    root = run_root(cfg) / "evaluation" / arm; root.mkdir(parents=True, exist_ok=True)
    save_json(root / "summary.json", result)
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as output:
        for record in records: output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result


def run_all(cfg):
    from pairs import prepare_pairs
    prepare_pairs(cfg)
    stage_a = {arm: train_stage_a(cfg, arm, window) for arm, window in cfg["stage_a_arms"].items()}
    model = load_model(cfg, "qwen")
    try:
        prepare_teachers(cfg, model)
        selected = {arm: select_stage_a(cfg, model, summary) for arm, summary in stage_a.items()}
        stage_b = train_stage_b(cfg, model)
        results = {"E0_oracle": {"accuracy": functional_validation(cfg, model, None, rows(cfg, "test"), "test", direct=True)["oracle_accuracy"]}}
        results["E1_direct_nearest"] = final_evaluate_arm(cfg, model, "direct_nearest", direct=True)
        for arm in ("nearest_linear", "local5_linear", "local5_choice_kl"):
            module, _ = load_checkpoint(cfg, checkpoint_path(cfg, arm, "best")); module.eval()
            results[arm] = final_evaluate_arm(cfg, model, arm, module=module,
                                              diagnostics=arm in ("local5_linear", "local5_choice_kl"))
            del module; torch.cuda.empty_cache()
    finally:
        del model; torch.cuda.empty_cache()
    save_json(run_root(cfg) / "results" / "summary.json", {
        "experiment": "rawtext_anchor_nativekv_translator_llama_to_qwen",
        "pairing": "Llama-driven raw-text anchors; target never uses Qwen-self selection",
        "schema": "33 pre-RoPE entries; Qwen canonical positions 0..32; suffix starts 33",
        "checkpoint_selection": "minimum validation choice KL; gold accuracy diagnostic only",
        "stage_a": stage_a, "stage_a_selected": selected, "stage_b": stage_b, "results": results})
    log("ALL RAW-TEXT ANCHOR NATIVE-KV TRANSLATOR EXPERIMENTS COMPLETED")
