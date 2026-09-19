from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_model, log, read_json, run_root, save_json, save_tensor, seed_all
from pairs import load_pair, rows
from protocol import final_logits
from translator import NativeKVTranslator, representation_loss


def checkpoint_path(cfg, arm, name):
    return run_root(cfg) / "checkpoints" / arm / f"{name}.pt"


def save_checkpoint(cfg, arm, name, module, step, representation=None):
    save_tensor(checkpoint_path(cfg, arm, name), {
        "signature": cfg["signature"], "arm": arm, "architecture": module.architecture,
        "step": step, "validation_representation": representation,
        "state": {key: value.detach().cpu() for key, value in module.state_dict().items()},
    })


def load_checkpoint(cfg, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]:
        raise RuntimeError("Checkpoint signature mismatch")
    module = NativeKVTranslator(payload["architecture"]).cuda()
    module.load_state_dict(payload["state"])
    return module, payload


def content_batch(cfg, split, selected_rows):
    payloads = [load_pair(cfg, split, row) for row in selected_rows]
    # Index 0 is explicitly excluded from both input and target training tensors.
    names = ("source_k", "source_v", "target_k", "target_v")
    return tuple(torch.stack([payload[name][:, 1:] for payload in payloads]).cuda() for name in names)


@torch.no_grad()
def representation_metrics(cfg, module, split, selected_rows):
    module.eval()
    totals = {key: 0.0 for key in ("loss", "k_nmse", "v_nmse", "k_cosine", "v_cosine")}
    for begin in range(0, len(selected_rows), cfg["batch_size"]):
        sk, sv, tk, tv = content_batch(cfg, split, selected_rows[begin:begin + cfg["batch_size"]])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pk, pv = module(sk, sv)
        loss, detail = representation_loss(pk, pv, tk, tv)
        count = sk.shape[0]
        totals["loss"] += loss.item() * count
        for key, value in detail.items():
            totals[key] += value.item() * count
    return {key: value / len(selected_rows) for key, value in totals.items()}


def train_stage_a(cfg, architecture):
    offset = cfg["architectures"].index(architecture)
    seed_all(cfg["seed"] + offset)
    train_rows, validation_rows = rows(cfg, "train"), rows(cfg, "validation")
    module = NativeKVTranslator(architecture).cuda()
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["stage_a_learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    root = run_root(cfg) / "training" / architecture / "stage_a"
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / "steps.jsonl").open("w", encoding="utf-8")
    candidates, step, clipped, started = [], 0, 0, time.monotonic()
    try:
        for epoch in range(1, cfg["stage_a_epochs"] + 1):
            order = list(range(len(train_rows)))
            random.Random(cfg["seed"] + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                selected = [train_rows[index] for index in order[begin:begin + cfg["batch_size"]]]
                sk, sv, tk, tv = content_batch(cfg, "train", selected)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pk, pv = module(sk, sv)
                loss, detail = representation_loss(pk, pv, tk, tv)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite Stage-A loss: {architecture}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if not torch.isfinite(norm):
                    raise RuntimeError(f"Invalid Stage-A gradient: {architecture}")
                clipped += int(norm.item() > cfg["clip"])
                scaler.step(optimizer)
                scaler.update()
                step += 1
                record = {"epoch": epoch, "step": step, "loss": loss.item(),
                          **{key: value.item() for key, value in detail.items()},
                          "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                if step % 16 == 0:
                    log(f"Stage A {architecture} {step}/{cfg['stage_a_steps']} loss={loss.item():.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["stage_a_steps"]:
                    metrics = representation_metrics(cfg, module, "validation", validation_rows)
                    name = f"step_{step}"
                    save_checkpoint(cfg, architecture, name, module, step, metrics)
                    candidates.append({"step": step, "path": str(checkpoint_path(cfg, architecture, name)),
                                       "representation": metrics})
                    save_json(root / "candidates.json", candidates)
                    log(f"Stage A {architecture} validation step={step} rep={metrics['loss']:.6f}")
                if step >= cfg["stage_a_steps"]:
                    break
            if step >= cfg["stage_a_steps"]:
                break
    finally:
        stream.close()
    summary = {"architecture": architecture, "steps": step, "clip_rate": clipped / max(step, 1),
               "parameters": sum(p.numel() for p in module.parameters()),
               "candidates": candidates, "seconds": time.monotonic() - started}
    save_json(root / "summary.json", summary)
    del module, optimizer, scaler
    torch.cuda.empty_cache()
    return summary


def teacher_path(cfg, split, row):
    return run_root(cfg) / "teacher_cache" / split / f"{row['id']}.pt"


def load_teacher(cfg, split, row):
    payload = torch.load(teacher_path(cfg, split, row), map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]: raise RuntimeError("Sync teacher cache signature mismatch")
    return payload


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
            if number % 32 == 0: log(f"Sync Oracle teacher {split}: {number}/{len(selected_rows)}")


def translated_content(module, pair):
    sk = pair["source_k"][:, 1:].cuda()[None]
    sv = pair["source_v"][:, 1:].cuda()[None]
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pk, pv = module(sk, sv)
    return pk[0], pv[0]


def receiver_memory(pair, translated):
    pk, pv = translated
    # True Qwen-native token0 is fixed; Writer only supplies the 32 content anchors.
    return (torch.cat((pair["target_k"][:, :1].cuda(), pk), dim=1),
            torch.cat((pair["target_v"][:, :1].cuda(), pv), dim=1))


def choice_loss(student, teacher_choice, choice_ids, temperature=1.0):
    ids = torch.tensor(choice_ids, device=student.device)
    s = F.log_softmax(student[ids].float() / temperature, -1)
    t = F.log_softmax(teacher_choice.to(student.device).float() / temperature, -1)
    return F.kl_div(s, t.detach(), reduction="sum", log_target=True) * temperature ** 2


def full_kl(student, teacher):
    return F.kl_div(F.log_softmax(student.float(), -1),
                    F.log_softmax(teacher.to(student.device).float(), -1),
                    reduction="sum", log_target=True)


@torch.no_grad()
def functional_metrics(cfg, model, module, selected_rows, split, records=False):
    module.eval()
    totals = {key: 0.0 for key in ("accuracy", "oracle_accuracy", "qwen_full_accuracy",
                                    "oracle_agreement", "choice_kl", "full_kl")}
    per_sample = []
    for number, row in enumerate(selected_rows, 1):
        pair, teacher = load_pair(cfg, split, row), load_teacher(cfg, split, row)
        key, value = receiver_memory(pair, translated_content(module, pair))
        metadata = pair["metadata"]
        logits = final_logits(model, metadata["qwen_suffix"], key, value,
                              positions=torch.arange(33, device="cuda"), suffix_start=33)
        ids = metadata["qwen_choice_ids"]
        prediction = int(logits[ids].argmax())
        oracle_prediction = int(teacher["choice_logits"].argmax())
        qwen_full_prediction = int(pair["qwen_full_native_logits"][ids].argmax())
        values = {
            "accuracy": float(prediction == metadata["gold_index"]),
            "oracle_accuracy": float(oracle_prediction == metadata["gold_index"]),
            "qwen_full_accuracy": float(qwen_full_prediction == metadata["gold_index"]),
            "oracle_agreement": float(prediction == oracle_prediction),
            "choice_kl": choice_loss(logits, teacher["choice_logits"], ids, cfg["temperature"]).item(),
            "full_kl": full_kl(logits, teacher["full_logits"]).item(),
        }
        for key_name, value_item in values.items():
            totals[key_name] += value_item
        if records:
            per_sample.append({"id": row["id"], "gold_index": metadata["gold_index"],
                               "prediction": prediction, "oracle_prediction": oracle_prediction, **values})
        if records and number % 16 == 0:
            log(f"Test {module.architecture}: {number}/{len(selected_rows)}")
    metrics = {key: value / len(selected_rows) for key, value in totals.items()}
    metrics["accuracy_retention"] = (metrics["accuracy"] / metrics["oracle_accuracy"]
                                      if metrics["oracle_accuracy"] else None)
    return metrics, per_sample


def copy_selected(cfg, architecture, candidate, alias):
    module, payload = load_checkpoint(cfg, candidate["path"])
    save_checkpoint(cfg, architecture, alias, module, payload["step"], payload["validation_representation"])
    del module
    torch.cuda.empty_cache()


def select_stage_a(cfg, model, summary):
    architecture = summary["architecture"]
    evaluated = []
    for candidate in summary["candidates"]:
        module, _ = load_checkpoint(cfg, candidate["path"])
        metrics, _ = functional_metrics(cfg, model, module, rows(cfg, "validation"), "validation")
        evaluated.append({**candidate, "functional": metrics})
        log(f"Functional {architecture} step={candidate['step']} acc={metrics['accuracy']:.4f} choice_KL={metrics['choice_kl']:.6f}")
        del module
        torch.cuda.empty_cache()
    best_choice = min(evaluated, key=lambda item: (item["functional"]["choice_kl"], item["step"]))
    best_accuracy = max(evaluated, key=lambda item: (item["functional"]["accuracy"],
                                                     -item["functional"]["choice_kl"], -item["step"]))
    copy_selected(cfg, architecture, best_choice, "best_choice_kl")
    copy_selected(cfg, architecture, best_accuracy, "best_accuracy")
    result = {"architecture": architecture,
              "selection_rule": "validation only; best_accuracy ties broken by lower choice KL then earlier step",
              "best_choice_kl": best_choice, "best_accuracy": best_accuracy,
              "candidates": evaluated}
    root = run_root(cfg) / "training" / architecture / "stage_a"
    save_json(root / "functional_candidates.json", evaluated)
    save_json(root / "selection.json", result)
    return result


@torch.no_grad()
def oracle_metrics(cfg, model, split):
    totals = {key: 0.0 for key in ("accuracy", "qwen_full_accuracy")}
    selected_rows = rows(cfg, split)
    for row in selected_rows:
        pair, teacher = load_pair(cfg, split, row), load_teacher(cfg, split, row)
        totals["accuracy"] += int(teacher["choice_logits"].argmax()) == pair["metadata"]["gold_index"]
        ids = pair["metadata"]["qwen_choice_ids"]
        totals["qwen_full_accuracy"] += int(pair["qwen_full_native_logits"][ids].argmax()) == pair["metadata"]["gold_index"]
    return {key: value / len(selected_rows) for key, value in totals.items()}


def evaluate_selected(cfg, model, selections):
    results = {"oracle": oracle_metrics(cfg, model, "test")}
    for architecture in cfg["architectures"]:
        results[architecture] = {}
        for alias in ("best_accuracy", "best_choice_kl"):
            module, payload = load_checkpoint(cfg, checkpoint_path(cfg, architecture, alias))
            metrics, records = functional_metrics(cfg, model, module, rows(cfg, "test"), "test", records=True)
            rep = representation_metrics(cfg, module, "test", rows(cfg, "test"))
            result = {"checkpoint_step": payload["step"], "functional": metrics, "representation": rep}
            results[architecture][alias] = result
            root = run_root(cfg) / "evaluation" / architecture / alias
            save_json(root / "summary.json", result)
            root.mkdir(parents=True, exist_ok=True)
            with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as output:
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            del module
            torch.cuda.empty_cache()
    save_json(run_root(cfg) / "results" / "phase1_summary.json", {
        "experiment": "rawtext_anchor_nativekv_full28_head8_ablation_llama_to_qwen",
        "primary_metric": "test accuracy; checkpoint selection uses validation only",
        "protocol": "Qwen Native token0 + 32 translated content anchors; suffix starts at 33",
        "selections": selections, "results": results,
    })
    return results


def weight_audit(cfg, arm, alias="best_accuracy"):
    module, payload = load_checkpoint(cfg, checkpoint_path(cfg, arm, alias))
    architecture = module.architecture
    audit = {"arm": arm, "architecture": architecture, "checkpoint": alias, "step": payload["step"]}
    if architecture != "local5_samehead":
        for component, maps in (("k", module.k_depth), ("v", module.v_depth)):
            matrix = []
            for mapping in maps:
                blocks = mapping.weight.detach().float().cpu().reshape(128, 28, 128)
                scores = blocks.square().sum((0, 2)).sqrt()
                matrix.append((scores / scores.sum().clamp_min(1e-12)).tolist())
            audit[f"{component}_layer_norm_36x28"] = matrix
    if architecture == "full28_head8":
        for component, maps in (("k", module.k_head), ("v", module.v_head)):
            matrices = []
            for layer_maps in maps:
                layer = []
                for mapping in layer_maps:
                    blocks = mapping.weight.detach().float().cpu().reshape(128, 8, 128)
                    scores = blocks.square().sum((0, 2)).sqrt()
                    layer.append((scores / scores.sum().clamp_min(1e-12)).tolist())
                matrices.append(layer)
            audit[f"{component}_head_norm_36x8x8"] = matrices
    save_json(run_root(cfg) / "audit" / f"weights_{arm}_{alias}.json", audit)
    del module
    torch.cuda.empty_cache()


def train_stage_b(cfg, model, architecture="full28_head8"):
    """Functional fine-tuning from Phase-1 validation-best accuracy checkpoint."""
    seed_all(cfg["seed"] + 100)
    module, source = load_checkpoint(cfg, checkpoint_path(cfg, architecture, "best_accuracy"))
    train_rows, validation_rows = rows(cfg, "train"), rows(cfg, "validation")
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["stage_b_learning_rate"], weight_decay=0)
    # The frozen FP16 receiver backward path overflows with a large initial scale on V100.
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=1000000)
    arm = f"{architecture}_choice_kl"
    root = run_root(cfg) / "training" / arm / "stage_b"
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / "steps.jsonl").open("w", encoding="utf-8")
    candidates, step, clipped, started = [], 0, 0, time.monotonic()
    try:
        for epoch in range(1, cfg["stage_b_epochs"] + 1):
            order = list(range(len(train_rows)))
            random.Random(cfg["seed"] + 100 + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                indices = order[begin:begin + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index in indices:
                    row = train_rows[index]
                    pair, teacher = load_pair(cfg, "train", row), load_teacher(cfg, "train", row)
                    key, value = receiver_memory(pair, translated_content(module, pair))
                    metadata = pair["metadata"]
                    student = final_logits(model, metadata["qwen_suffix"], key, value,
                                           positions=torch.arange(33, device="cuda"), suffix_start=33)
                    loss = choice_loss(student, teacher["choice_logits"],
                                       metadata["qwen_choice_ids"], cfg["temperature"])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite Stage-B choice loss")
                    scaler.scale(loss / len(indices)).backward()
                    losses.append(loss.item())
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if not torch.isfinite(norm):
                    raise RuntimeError("Invalid Stage-B gradient")
                clipped += int(norm.item() > cfg["clip"])
                scaler.step(optimizer)
                scaler.update()
                step += 1
                record = {"epoch": epoch, "step": step,
                          "mean_choice_kl": sum(losses) / len(losses),
                          "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                if step % 16 == 0:
                    log(f"Stage B {arm} {step}/{cfg['stage_b_steps']} choice_KL={record['mean_choice_kl']:.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["stage_b_steps"]:
                    metrics, _ = functional_metrics(cfg, model, module, validation_rows, "validation")
                    name = f"step_{step}"
                    save_checkpoint(cfg, arm, name, module, step)
                    candidate = {"step": step, "path": str(checkpoint_path(cfg, arm, name)),
                                 "functional": metrics}
                    candidates.append(candidate)
                    save_json(root / "candidates.json", candidates)
                    log(f"Stage B validation step={step} acc={metrics['accuracy']:.4f} choice_KL={metrics['choice_kl']:.6f}")
                if step >= cfg["stage_b_steps"]:
                    break
            if step >= cfg["stage_b_steps"]:
                break
    finally:
        stream.close()
    best_choice = min(candidates, key=lambda item: (item["functional"]["choice_kl"], item["step"]))
    best_accuracy = max(candidates, key=lambda item: (item["functional"]["accuracy"],
                                                      -item["functional"]["choice_kl"], -item["step"]))
    copy_selected(cfg, arm, best_choice, "best_choice_kl")
    copy_selected(cfg, arm, best_accuracy, "best_accuracy")
    summary = {"arm": arm, "architecture": architecture,
               "initialized_from": {"alias": "best_accuracy", "step": source["step"]},
               "objective": "ONLY final-position A-J choice KL; no gold labels; Qwen Native token0 fixed",
               "steps": step, "epochs": cfg["stage_b_epochs"], "clip_rate": clipped / max(step, 1),
               "best_choice_kl": best_choice, "best_accuracy": best_accuracy,
               "candidates": candidates, "seconds": time.monotonic() - started}
    save_json(root / "summary.json", summary)
    del module, optimizer, scaler
    torch.cuda.empty_cache()
    return summary


def evaluate_stage_b(cfg, model, summary):
    arm = summary["arm"]
    result = {}
    for alias in ("best_accuracy", "best_choice_kl"):
        module, payload = load_checkpoint(cfg, checkpoint_path(cfg, arm, alias))
        metrics, records = functional_metrics(cfg, model, module, rows(cfg, "test"), "test", records=True)
        rep = representation_metrics(cfg, module, "test", rows(cfg, "test"))
        value = {"checkpoint_step": payload["step"], "functional": metrics, "representation": rep}
        result[alias] = value
        root = run_root(cfg) / "evaluation" / arm / alias
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / "summary.json", value)
        with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        del module
        torch.cuda.empty_cache()
    save_json(run_root(cfg) / "results" / f"stage_b_{summary['architecture']}_summary.json", {
        "experiment": f"{summary['architecture']}_pure_choice_kl_stage_b",
        "training": summary, "test": result,
    })
    return result


def run_stage_b(cfg):
    architecture = "full28_diagonal"
    source = checkpoint_path(cfg, architecture, "best_accuracy")
    if not source.exists():
        raise FileNotFoundError(f"Missing Phase-1 initialization: {source}")
    model = load_model(cfg, "qwen")
    try:
        summary = train_stage_b(cfg, model, architecture)
        test = evaluate_stage_b(cfg, model, summary)
    finally:
        del model
        torch.cuda.empty_cache()
    weight_audit(cfg, f"{architecture}_choice_kl")
    oracle = read_json(Path(cfg["oracle_root"]) / "results" / "summary.json")
    rawanchor = read_json(Path(cfg["rawanchor_baseline_root"]) / "results" / "stage_b_full28_diagonal_summary.json")
    save_json(run_root(cfg) / "results" / "final_comparison.json", {
        "primary_metric": "test accuracy",
        "qwen_full_accuracy": oracle["accuracy"]["qwen_full"],
        "rawanchor_oracle_accuracy": oracle["accuracy"]["current_rawanchor"],
        "sync_max_unique_oracle_accuracy": oracle["accuracy"]["sync_max_unique"],
        "rawanchor_translated_reference": rawanchor["test"],
        "sync_translated": test,
    })
    log("STAGE B SYNC FULL28 DIAGONAL CHOICE-KL COMPLETED")


def run_stage_b_all(cfg):
    """Run the three missing arms and combine them with the completed Head8 arm."""
    architectures = list(cfg["architectures"])
    model = load_model(cfg, "qwen")
    combined = {}
    try:
        for architecture in architectures:
            result_path = run_root(cfg) / "results" / f"stage_b_{architecture}_summary.json"
            legacy_head8 = run_root(cfg) / "results" / "stage_b_summary.json"
            if architecture == "full28_head8" and not result_path.exists() and legacy_head8.exists():
                legacy = json.loads(legacy_head8.read_text(encoding="utf-8"))
                legacy["experiment"] = "full28_head8_pure_choice_kl_stage_b"
                save_json(result_path, legacy)
                combined[architecture] = legacy
                log("Reusing already completed Full28 Head8 Stage B")
                continue
            if result_path.exists():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if existing.get("training", {}).get("steps") == cfg["stage_b_steps"]:
                    combined[architecture] = existing
                    log(f"Reusing completed Stage B arm: {architecture}")
                    continue
            source = checkpoint_path(cfg, architecture, "best_accuracy")
            if not source.exists():
                raise FileNotFoundError(f"Missing Phase-1 initialization: {source}")
            summary = train_stage_b(cfg, model, architecture)
            test = evaluate_stage_b(cfg, model, summary)
            weight_audit(cfg, summary["arm"])
            combined[architecture] = {"experiment": f"{architecture}_pure_choice_kl_stage_b",
                                      "training": summary, "test": test}
            log(f"Stage B architecture completed: {architecture}")
    finally:
        del model
        torch.cuda.empty_cache()
    comparison = {
        "protocol": {
            "native_token0": True, "raw_text_anchors": 32, "train_samples": cfg["train_samples"],
            "steps": cfg["stage_b_steps"], "learning_rate": cfg["stage_b_learning_rate"],
            "objective": "pure final-position choice KL", "checkpoint_interval": cfg["validation_interval_steps"],
            "initialization": "each architecture's Phase-1 validation best_accuracy checkpoint",
            "selection": "validation only; best_accuracy and best_choice_kl both reported on test",
        },
        "architectures": combined,
        "test_accuracy": {
            architecture: {
                alias: payload["test"][alias]["functional"]["accuracy"]
                for alias in ("best_accuracy", "best_choice_kl")
            }
            for architecture, payload in combined.items()
        },
    }
    save_json(run_root(cfg) / "results" / "stage_b_architecture_comparison.json", comparison)
    log("ALL FOUR STAGE B ARCHITECTURE ARMS COMPLETED")


def run_phase1(cfg):
    from pairs import audit_pairs
    audit_pairs(cfg)
    stage_a = {architecture: train_stage_a(cfg, architecture) for architecture in cfg["architectures"]}
    model = load_model(cfg, "qwen")
    try:
        prepare_teachers(cfg, model)
        selections = {architecture: select_stage_a(cfg, model, summary)
                      for architecture, summary in stage_a.items()}
        evaluate_selected(cfg, model, selections)
    finally:
        del model
        torch.cuda.empty_cache()
    for architecture in cfg["architectures"]:
        weight_audit(cfg, architecture)
    save_json(run_root(cfg) / "results" / "phase1_training.json", stage_a)
    log("PHASE 1 ARCHITECTURE AUDIT COMPLETED")
