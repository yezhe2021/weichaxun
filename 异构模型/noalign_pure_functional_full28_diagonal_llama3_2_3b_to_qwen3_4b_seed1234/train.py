from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_qwen, log, read_json, run_root, save_json, save_tensor, seed_all
from data import load_noalign, protocol_audit, rows
from protocol import final_logits
from translator import NativeKVTranslator


ARM = "noalign_randominit_full28_diagonal_choice_kl"


def checkpoint_path(cfg, name):
    return run_root(cfg) / "checkpoints" / ARM / f"{name}.pt"


def save_checkpoint(cfg, name, module, step):
    save_tensor(checkpoint_path(cfg, name), {
        "signature": cfg["signature"], "arm": ARM, "architecture": module.architecture,
        "step": step, "state": {key: value.detach().cpu() for key, value in module.state_dict().items()},
    })


def load_checkpoint(cfg, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]:
        raise RuntimeError("Checkpoint signature mismatch")
    module = NativeKVTranslator(payload["architecture"]).cuda()
    module.load_state_dict(payload["state"])
    return module, payload


def translate(module, item):
    sk = item["source_k"].cuda()[None]
    sv = item["source_v"].cuda()[None]
    with torch.amp.autocast("cuda", dtype=torch.float16):
        key, value = module(sk, sv)
    return key[0], value[0]


def memory(item, functional):
    key, value = functional
    return (torch.cat((item["qwen_native_token0_k"].cuda(), key), 1),
            torch.cat((item["qwen_native_token0_v"].cuda(), value), 1))


def choice_kl(student, teacher, choice_ids, temperature):
    ids = torch.tensor(choice_ids, device=student.device)
    log_student = F.log_softmax(student[ids].float() / temperature, -1)
    log_teacher = F.log_softmax(teacher.to(student.device)[ids].float() / temperature, -1)
    return F.kl_div(log_student, log_teacher.detach(), reduction="sum", log_target=True) * temperature ** 2


def full_kl(student, teacher):
    return F.kl_div(F.log_softmax(student.float(), -1),
                    F.log_softmax(teacher.to(student.device).float(), -1),
                    reduction="sum", log_target=True)


@torch.no_grad()
def evaluate(cfg, model, module, split, write_records=False):
    module.eval(); selected = rows(cfg, split)
    totals = {key: 0.0 for key in ("accuracy", "teacher_accuracy", "teacher_agreement", "choice_kl", "full_kl")}
    records = []
    for number, row in enumerate(selected, 1):
        item = load_noalign(cfg, split, row)
        key, value = memory(item, translate(module, item))
        logits = final_logits(model, item["qwen_suffix"], key, value,
                              positions=torch.arange(33, device="cuda"), suffix_start=33)
        ids = item["qwen_choice_ids"]
        prediction = int(logits[ids].argmax())
        teacher_prediction = int(item["teacher_full_logits"][ids].argmax())
        values = {
            "accuracy": float(prediction == item["gold_index"]),
            "teacher_accuracy": float(teacher_prediction == item["gold_index"]),
            "teacher_agreement": float(prediction == teacher_prediction),
            "choice_kl": choice_kl(logits, item["teacher_full_logits"], ids, cfg["temperature"]).item(),
            "full_kl": full_kl(logits, item["teacher_full_logits"]).item(),
        }
        for name, value_item in values.items(): totals[name] += value_item
        if write_records:
            records.append({"id": item["id"], "gold_index": item["gold_index"],
                            "prediction": prediction, "teacher_prediction": teacher_prediction, **values})
        if write_records and number % 16 == 0:
            log(f"NoAlign test: {number}/{len(selected)}")
    metrics = {name: value / len(selected) for name, value in totals.items()}
    metrics["teacher_retention"] = metrics["accuracy"] / metrics["teacher_accuracy"] if metrics["teacher_accuracy"] else None
    return metrics, records


def train(cfg, model):
    seed_all(cfg["seed"])
    module = NativeKVTranslator(cfg["architecture"]).cuda()
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=1000000)
    train_rows, validation_rows = rows(cfg, "train"), rows(cfg, "validation")
    root = run_root(cfg) / "training" / ARM
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / "steps.jsonl").open("w", encoding="utf-8")
    candidates, step, clipped, started = [], 0, 0, time.monotonic()
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            order = list(range(len(train_rows)))
            random.Random(cfg["seed"] + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                indices = order[begin:begin + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True); losses = []
                for index in indices:
                    item = load_noalign(cfg, "train", train_rows[index])
                    key, value = memory(item, translate(module, item))
                    student = final_logits(model, item["qwen_suffix"], key, value,
                                           positions=torch.arange(33, device="cuda"), suffix_start=33)
                    loss = choice_kl(student, item["teacher_full_logits"],
                                     item["qwen_choice_ids"], cfg["temperature"])
                    if not torch.isfinite(loss): raise RuntimeError("Nonfinite NoAlign loss")
                    scaler.scale(loss / len(indices)).backward(); losses.append(loss.item())
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if not torch.isfinite(norm): raise RuntimeError("Invalid NoAlign gradient")
                clipped += int(norm.item() > cfg["clip"])
                scaler.step(optimizer); scaler.update(); step += 1
                record = {"epoch": epoch, "step": step, "mean_choice_kl": sum(losses) / len(losses),
                          "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
                stream.write(json.dumps(record) + "\n"); stream.flush()
                if step % 16 == 0:
                    log(f"NoAlign Stage B {step}/{cfg['steps']} choice_KL={record['mean_choice_kl']:.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["steps"]:
                    metrics, _ = evaluate(cfg, model, module, "validation")
                    name = f"step_{step}"; save_checkpoint(cfg, name, module, step)
                    candidates.append({"step": step, "path": str(checkpoint_path(cfg, name)), "functional": metrics})
                    save_json(root / "candidates.json", candidates)
                    log(f"NoAlign validation step={step} acc={metrics['accuracy']:.4f} choice_KL={metrics['choice_kl']:.6f}")
                if step >= cfg["steps"]: break
            if step >= cfg["steps"]: break
    finally:
        stream.close()
    best_choice = min(candidates, key=lambda x: (x["functional"]["choice_kl"], x["step"]))
    best_accuracy = max(candidates, key=lambda x: (x["functional"]["accuracy"], -x["functional"]["choice_kl"], -x["step"]))
    for alias, candidate in (("best_choice_kl", best_choice), ("best_accuracy", best_accuracy)):
        selected_module, payload = load_checkpoint(cfg, candidate["path"])
        save_checkpoint(cfg, alias, selected_module, payload["step"])
        del selected_module; torch.cuda.empty_cache()
    summary = {"arm": ARM, "random_initialization": True, "stage_a": False,
               "objective": "pure full-Qwen-teacher choice KL", "steps": step,
               "epochs": cfg["epochs"], "clip_rate": clipped / max(step, 1),
               "best_choice_kl": best_choice, "best_accuracy": best_accuracy,
               "candidates": candidates, "seconds": time.monotonic() - started}
    save_json(root / "summary.json", summary)
    del module, optimizer, scaler; torch.cuda.empty_cache()
    return summary


def test_selected(cfg, model, training):
    result = {}
    for alias in ("best_accuracy", "best_choice_kl"):
        module, payload = load_checkpoint(cfg, checkpoint_path(cfg, alias))
        metrics, records = evaluate(cfg, model, module, "test", True)
        result[alias] = {"checkpoint_step": payload["step"], "functional": metrics}
        root = run_root(cfg) / "evaluation" / alias; root.mkdir(parents=True, exist_ok=True)
        save_json(root / "summary.json", result[alias])
        with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as output:
            for record in records: output.write(json.dumps(record, ensure_ascii=False) + "\n")
        del module; torch.cuda.empty_cache()
    baseline_path = Path(cfg["aligned_baseline_root"]) / "results" / "stage_b_full28_diagonal_summary.json"
    baseline = read_json(baseline_path) if baseline_path.exists() else {"unavailable": str(baseline_path)}
    save_json(run_root(cfg) / "results" / "summary.json", {
        "experiment": "NoAlign RandomInit Full28 Diagonal Pure Functional",
        "protocol": "Qwen Native token0 + 32 ordered Llama-derived functional memories + Qwen query",
        "training": training, "test": result,
        "aligned_full28_diagonal_reference": baseline,
    })


def run_all(cfg):
    protocol_audit(cfg)
    model = load_qwen(cfg)
    try:
        summary = train(cfg, model)
        test_selected(cfg, model, summary)
    finally:
        del model; torch.cuda.empty_cache()
    log("NOALIGN RANDOMINIT FULL28 DIAGONAL EXPERIMENT COMPLETED")
