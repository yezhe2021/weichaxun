from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_model, log, run_root, save_json, save_tensor, seed_all
from data import load_pair, load_source, manifest_rows
from protocol import final_logits
from translator import NativeKVTranslator, representation_loss


def checkpoint_path(cfg, stage, step):
    return run_root(cfg) / stage / "checkpoints" / f"step_{step}.pt"


def save_checkpoint(cfg, stage, module, step, validation):
    path = checkpoint_path(cfg, stage, step)
    save_tensor(path, {"signature": cfg["signature"], "stage": stage, "architecture": module.architecture,
                       "step": step, "validation": validation,
                       "state": {name: value.detach().cpu() for name, value in module.state_dict().items()}})
    return path


def load_checkpoint(cfg, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"] or payload["architecture"] != cfg["architecture"]:
        raise RuntimeError("Checkpoint mismatch")
    module = NativeKVTranslator(payload["architecture"]).cuda()
    module.load_state_dict(payload["state"], strict=True)
    return module, payload


def choice_kl(student, teacher_choice, choice_ids, temperature):
    ids = torch.tensor(choice_ids, device=student.device)
    s = F.log_softmax(student[ids].float() / temperature, -1)
    t = F.log_softmax(teacher_choice.to(student.device).float() / temperature, -1)
    return F.kl_div(s, t.detach(), reduction="sum", log_target=True) * temperature ** 2


def source_target_batch(cfg, split, selected_rows):
    sources = [load_source(cfg, split, row) for row in selected_rows]
    pairs = [load_pair(cfg, split, row) for row in selected_rows]
    source_k = torch.stack([item["source_k"][:, 1:] for item in sources]).cuda()
    source_v = torch.stack([item["source_v"][:, 1:] for item in sources]).cuda()
    target_k = torch.stack([item["target_k"][:, 1:] for item in pairs]).cuda()
    target_v = torch.stack([item["target_v"][:, 1:] for item in pairs]).cuda()
    return source_k, source_v, target_k, target_v


def translated_content(module, source):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        key, value = module(source["source_k"][:, 1:].cuda()[None], source["source_v"][:, 1:].cuda()[None])
    return key[0], value[0]


def receiver_memory(pair, content):
    key, value = content
    return (torch.cat((pair["question_k"].cuda(), key), 1),
            torch.cat((pair["question_v"].cuda(), value), 1))


def receiver_logits(model, row, key, value):
    return final_logits(model, row["encoded"]["qwen"]["receiver_answer"], key, value,
                        positions=torch.arange(key.shape[1], device="cuda"), suffix_start=key.shape[1])


def student_logits(model, row, pair, content):
    return receiver_logits(model, row, *receiver_memory(pair, content))


def teacher_path(cfg, split, row):
    return run_root(cfg) / "oracle_teacher" / split / f"{row['id']}.pt"


def load_teacher(cfg, split, row):
    payload = torch.load(teacher_path(cfg, split, row), map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]: raise RuntimeError("Teacher mismatch")
    return payload


@torch.no_grad()
def prepare_oracle_teachers(cfg, model):
    for split in ("train", "validation", "test"):
        selected = manifest_rows(cfg, split)
        for number, row in enumerate(selected, 1):
            path = teacher_path(cfg, split, row)
            if path.exists():
                try:
                    load_teacher(cfg, split, row); continue
                except RuntimeError: pass
            pair = load_pair(cfg, split, row)
            logits = receiver_logits(model, row, *receiver_memory(
                pair, (pair["target_k"][:, 1:].cuda(), pair["target_v"][:, 1:].cuda())))
            ids = row["encoded"]["qwen"]["choice_ids"]
            choice = logits[ids].cpu()
            save_tensor(path, {"signature": cfg["signature"], "choice_logits": choice,
                               "prediction": int(choice.argmax())})
            if number % 32 == 0 or number == len(selected):
                log(f"Posterior-Question Oracle teacher {split}: {number}/{len(selected)}")


@torch.no_grad()
def representation_metrics(cfg, module, split):
    module.eval(); selected = manifest_rows(cfg, split)
    totals = {key: 0.0 for key in ("loss", "k_nmse", "v_nmse", "k_cosine", "v_cosine")}
    for begin in range(0, len(selected), cfg["batch_size"]):
        batch = selected[begin:begin + cfg["batch_size"]]
        sk, sv, tk, tv = source_target_batch(cfg, split, batch)
        with torch.amp.autocast("cuda", dtype=torch.float16): pk, pv = module(sk, sv)
        loss, details = representation_loss(pk, pv, tk, tv)
        totals["loss"] += loss.item() * len(batch)
        for key, value in details.items(): totals[key] += value.item() * len(batch)
    return {key: value / len(selected) for key, value in totals.items()}


@torch.no_grad()
def functional_metrics(cfg, model, module, split):
    module.eval(); selected = manifest_rows(cfg, split)
    totals = {key: 0.0 for key in ("accuracy", "oracle_accuracy", "oracle_agreement", "oracle_choice_kl")}
    for row in selected:
        source, pair, teacher = load_source(cfg, split, row), load_pair(cfg, split, row), load_teacher(cfg, split, row)
        logits = student_logits(model, row, pair, translated_content(module, source))
        ids, gold = row["encoded"]["qwen"]["choice_ids"], row["gold_index"]
        prediction = int(logits[ids].argmax()); oracle_prediction = teacher["prediction"]
        totals["accuracy"] += prediction == gold
        totals["oracle_accuracy"] += oracle_prediction == gold
        totals["oracle_agreement"] += prediction == oracle_prediction
        totals["oracle_choice_kl"] += choice_kl(logits, teacher["choice_logits"], ids, cfg["temperature"]).item()
    return {key: value / len(selected) for key, value in totals.items()}


def train_stage_a(cfg):
    seed_all(cfg["seed"]); module = NativeKVTranslator(cfg["architecture"]).cuda()
    train_rows, validation_rows = manifest_rows(cfg, "train"), manifest_rows(cfg, "validation")
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["stage_a_learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    root = run_root(cfg) / "stage_a"; root.mkdir(parents=True, exist_ok=True)
    stream = (root / "training_steps.jsonl").open("w", encoding="utf-8")
    candidates, step, clipped, started = [], 0, 0, time.monotonic()
    try:
        for epoch in range(1, cfg["stage_a_epochs"] + 1):
            order = list(range(len(train_rows))); random.Random(cfg["seed"] + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                chosen = [train_rows[index] for index in order[begin:begin + cfg["batch_size"]]]
                sk, sv, tk, tv = source_target_batch(cfg, "train", chosen)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16): pk, pv = module(sk, sv)
                loss, details = representation_loss(pk, pv, tk, tv)
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite Stage-A loss")
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if not torch.isfinite(norm): raise RuntimeError("Invalid Stage-A gradient")
                clipped += int(norm.item() > cfg["clip"]); scaler.step(optimizer); scaler.update(); step += 1
                record = {"epoch": epoch, "step": step, "loss": loss.item(),
                          **{key: value.item() for key, value in details.items()},
                          "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
                stream.write(json.dumps(record) + "\n"); stream.flush()
                if step % 16 == 0: log(f"Stage A {step}/{cfg['stage_a_steps']} loss={loss.item():.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["stage_a_steps"]:
                    validation = representation_metrics(cfg, module, "validation")
                    path = save_checkpoint(cfg, "stage_a", module, step, {"representation": validation})
                    candidates.append({"step": step, "path": str(path), "representation": validation})
                    save_json(root / "candidates.json", candidates)
                    log(f"Stage A validation step={step} loss={validation['loss']:.6f}")
                if step >= cfg["stage_a_steps"]: break
            if step >= cfg["stage_a_steps"]: break
    finally: stream.close()
    summary = {"stage": "A", "architecture": cfg["architecture"], "objective": "KV reconstruction only",
               "steps": step, "clip_rate": clipped / max(step, 1), "seconds": time.monotonic() - started,
               "candidates": candidates}
    save_json(root / "training_summary.json", summary)
    del module, optimizer, scaler; torch.cuda.empty_cache()
    return summary


def select_stage_a(cfg, model):
    root = run_root(cfg) / "stage_a"
    summary = json.loads((root / "training_summary.json").read_text(encoding="utf-8"))
    evaluated = []
    for candidate in summary["candidates"]:
        module, _ = load_checkpoint(cfg, candidate["path"])
        functional = functional_metrics(cfg, model, module, "validation")
        evaluated.append({**candidate, "functional": functional})
        log(f"Stage A functional step={candidate['step']} acc={functional['accuracy']:.4f} KL={functional['oracle_choice_kl']:.6f}")
        del module; torch.cuda.empty_cache()
    best_accuracy = max(evaluated, key=lambda item: (item["functional"]["accuracy"],
                                                      -item["functional"]["oracle_choice_kl"], -item["step"]))
    best_kl = min(evaluated, key=lambda item: (item["functional"]["oracle_choice_kl"], item["step"]))
    selection = {"selection": "validation only", "best_accuracy": best_accuracy,
                 "best_choice_kl": best_kl, "candidates": evaluated}
    save_json(root / "selection.json", selection)
    return selection


def train_stage_b(cfg, model):
    selection = json.loads((run_root(cfg) / "stage_a" / "selection.json").read_text(encoding="utf-8"))
    module, source = load_checkpoint(cfg, selection["best_accuracy"]["path"])
    seed_all(cfg["seed"] + 100)
    train_rows = manifest_rows(cfg, "train")
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["stage_b_learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=1000000)
    root = run_root(cfg) / "stage_b"; root.mkdir(parents=True, exist_ok=True)
    stream = (root / "training_steps.jsonl").open("w", encoding="utf-8")
    candidates, step, clipped, started = [], 0, 0, time.monotonic()
    try:
        for epoch in range(1, cfg["stage_b_epochs"] + 1):
            order = list(range(len(train_rows))); random.Random(cfg["seed"] + 100 + epoch).shuffle(order)
            for begin in range(0, len(order), cfg["batch_size"]):
                indices = order[begin:begin + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True); losses = []
                for index in indices:
                    row = train_rows[index]
                    source_item, pair, teacher = load_source(cfg, "train", row), load_pair(cfg, "train", row), load_teacher(cfg, "train", row)
                    logits = student_logits(model, row, pair, translated_content(module, source_item))
                    loss = choice_kl(logits, teacher["choice_logits"], row["encoded"]["qwen"]["choice_ids"], cfg["temperature"])
                    if not torch.isfinite(loss): raise RuntimeError("Nonfinite Stage-B loss")
                    scaler.scale(loss / len(indices)).backward(); losses.append(loss.item())
                scaler.unscale_(optimizer); norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if not torch.isfinite(norm): raise RuntimeError("Invalid Stage-B gradient")
                clipped += int(norm.item() > cfg["clip"]); scaler.step(optimizer); scaler.update(); step += 1
                record = {"epoch": epoch, "step": step, "mean_choice_kl": sum(losses) / len(losses),
                          "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
                stream.write(json.dumps(record) + "\n"); stream.flush()
                if step % 16 == 0: log(f"Stage B {step}/{cfg['stage_b_steps']} KL={record['mean_choice_kl']:.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["stage_b_steps"]:
                    functional = functional_metrics(cfg, model, module, "validation")
                    representation = representation_metrics(cfg, module, "validation")
                    validation = {"functional": functional, "representation": representation}
                    path = save_checkpoint(cfg, "stage_b", module, step, validation)
                    candidates.append({"step": step, "path": str(path), **validation})
                    save_json(root / "candidates.json", candidates)
                    log(f"Stage B validation step={step} acc={functional['accuracy']:.4f} KL={functional['oracle_choice_kl']:.6f}")
                if step >= cfg["stage_b_steps"]: break
            if step >= cfg["stage_b_steps"]: break
    finally: stream.close()
    best_accuracy = max(candidates, key=lambda item: (item["functional"]["accuracy"],
                                                       -item["functional"]["oracle_choice_kl"], -item["step"]))
    best_kl = min(candidates, key=lambda item: (item["functional"]["oracle_choice_kl"], item["step"]))
    summary = {"stage": "B", "objective": "32-memory Oracle final-position A-J choice KL only",
               "initialized_from_stage_a_step": source["step"], "steps": step,
               "clip_rate": clipped / max(step, 1), "seconds": time.monotonic() - started,
               "selection": "validation only", "best_accuracy": best_accuracy,
               "best_choice_kl": best_kl, "candidates": candidates}
    save_json(root / "training_summary.json", summary)
    del module, optimizer, scaler; torch.cuda.empty_cache()
    return summary


@torch.no_grad()
def evaluate_native_baselines(cfg, model):
    selected = manifest_rows(cfg, "test")
    names = ("qwen_full_native", "llama_full_native", "qwen_self_selected32_native",
             "llama_selected_qwen_native_oracle", "zero", "no_memory")
    totals = {name: {"accuracy": 0.0, "oracle_agreement": 0.0, "oracle_choice_kl": 0.0}
              for name in names}
    records = []
    for number, row in enumerate(selected, 1):
        pair, teacher = load_pair(cfg, "test", row), load_teacher(cfg, "test", row)
        ids, gold = row["encoded"]["qwen"]["choice_ids"], row["gold_index"]
        zero_content = (torch.zeros_like(pair["target_k"][:, 1:].cuda()),
                        torch.zeros_like(pair["target_v"][:, 1:].cuda()))
        logits_by_name = {
            "qwen_full_native": pair["qwen_full_choice_logits"].cuda(),
            "llama_full_native": pair["llama_full_choice_logits"].cuda(),
            "qwen_self_selected32_native": receiver_logits(model, row, *receiver_memory(
                pair, (pair["qwen_self_k"].cuda(), pair["qwen_self_v"].cuda())))[ids],
            "llama_selected_qwen_native_oracle": teacher["choice_logits"].cuda(),
            "zero": receiver_logits(model, row, *receiver_memory(pair, zero_content))[ids],
            "no_memory": receiver_logits(model, row, pair["question_k"].cuda(), pair["question_v"].cuda())[ids],
        }
        oracle_prediction = teacher["prediction"]
        record = {"id": row["id"], "gold_index": gold, "conditions": {}}
        for name, choice_logits in logits_by_name.items():
            prediction = int(choice_logits.argmax())
            log_probs = F.log_softmax(choice_logits.float() / cfg["temperature"], -1)
            teacher_log_probs = F.log_softmax(teacher["choice_logits"].cuda().float() / cfg["temperature"], -1)
            kl = F.kl_div(log_probs, teacher_log_probs, reduction="sum", log_target=True).item()
            values = {"accuracy": float(prediction == gold),
                      "oracle_agreement": float(prediction == oracle_prediction), "oracle_choice_kl": kl}
            record["conditions"][name] = {"prediction": prediction, "choice_logits": choice_logits.cpu().tolist(), **values}
            for key, value in values.items(): totals[name][key] += value
        records.append(record)
        if number % 16 == 0 or number == len(selected): log(f"Native baselines: {number}/{len(selected)}")
    result = {name: {key: value / len(selected) for key, value in metrics.items()}
              for name, metrics in totals.items()}
    result["mean_random_chance"] = sum(1 / len(row["labels"]) for row in selected) / len(selected)
    root = run_root(cfg) / "evaluation" / "baselines"; root.mkdir(parents=True, exist_ok=True)
    save_json(root / "summary.json", result)
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as output:
        for record in records: output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result


@torch.no_grad()
def evaluate_translated(cfg, model, module, alias):
    module.eval(); selected = manifest_rows(cfg, "test")
    totals = {name: {key: 0.0 for key in ("accuracy", "oracle_agreement", "oracle_choice_kl")}
              for name in ("translated", "shuffle")}
    records = []
    for number, row in enumerate(selected, 1):
        shuffled_row = selected[(number) % len(selected)]
        source, shuffled_source = load_source(cfg, "test", row), load_source(cfg, "test", shuffled_row)
        pair, teacher = load_pair(cfg, "test", row), load_teacher(cfg, "test", row)
        ids, gold = row["encoded"]["qwen"]["choice_ids"], row["gold_index"]
        outputs = {
            "translated": student_logits(model, row, pair, translated_content(module, source)),
            "shuffle": student_logits(model, row, pair, translated_content(module, shuffled_source)),
        }
        oracle_prediction = teacher["prediction"]
        record = {"id": row["id"], "shuffle_source_id": shuffled_row["id"],
                  "gold_index": gold, "conditions": {}}
        for name, logits in outputs.items():
            choice_logits = logits[ids]; prediction = int(choice_logits.argmax())
            kl = choice_kl(logits, teacher["choice_logits"], ids, cfg["temperature"]).item()
            values = {"accuracy": float(prediction == gold),
                      "oracle_agreement": float(prediction == oracle_prediction), "oracle_choice_kl": kl}
            record["conditions"][name] = {"prediction": prediction,
                                           "choice_logits": choice_logits.cpu().tolist(), **values}
            for key, value in values.items(): totals[name][key] += value
        records.append(record)
        if number % 16 == 0 or number == len(selected): log(f"Translated {alias}: {number}/{len(selected)}")
    functional = {name: {key: value / len(selected) for key, value in metrics.items()}
                  for name, metrics in totals.items()}
    result = {"checkpoint": alias, "functional": functional,
              "representation": representation_metrics(cfg, module, "test")}
    root = run_root(cfg) / "evaluation" / "translated" / alias; root.mkdir(parents=True, exist_ok=True)
    save_json(root / "summary.json", result)
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as output:
        for record in records: output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result


def safe_retention(translated, oracle, control):
    denominator = oracle - control
    return (translated - control) / denominator if abs(denominator) > 1e-12 else None


def append_results_document(cfg, comparison):
    if cfg["mode"] != "study": return
    document = Path(cfg["shared_results_document"]); document.parent.mkdir(parents=True, exist_ok=True)
    tag = "mmlupro_sender_qoq_posterior_question_rawanchor_full28_diagonal_llama3_2_3b_to_qwen3_4b_seed1234"
    begin, end = f"<!-- BEGIN {tag} -->", f"<!-- END {tag} -->"
    rows_text = []
    for name, metrics in comparison["main_table"].items():
        rows_text.append(f"| {name} | {100 * metrics['accuracy']:.2f}% | "
                         f"{100 * metrics.get('oracle_agreement', 0):.2f}% | {metrics.get('oracle_choice_kl', 0):.6f} |")
    gaps = comparison["gaps"]
    section = "\n".join([begin, f"## {tag}", "",
        "Machine-generated after successful completion. Sender routing uses Question -> Options -> repeated Question; Receiver remains native Question -> translated Options KV -> native Answer:.", "",
        "| Protocol | Accuracy | Oracle agreement | Oracle choice-KL |", "|---|---:|---:|---:|",
        *rows_text, "",
        f"Selection gap={100*gaps['selection_gap']:.2f} points; translation gap={100*gaps['translation_gap']:.2f} points; "
        f"information gain vs zero={100*gaps['information_gain_vs_zero']:.2f} points; "
        f"information gain vs no-memory={100*gaps['information_gain_vs_no_memory']:.2f} points.", end, ""])
    current = document.read_text(encoding="utf-8") if document.exists() else "# Experiment Results\n\n"
    if begin in current and end in current:
        left, tail = current.split(begin, 1); _, right = tail.split(end, 1)
        current = left.rstrip() + "\n\n" + section + right.lstrip("\n")
    else: current = current.rstrip() + "\n\n" + section
    document.write_text(current, encoding="utf-8")


def finalize(cfg):
    model = load_model(cfg, "qwen")
    try:
        baselines_path = run_root(cfg) / "evaluation" / "baselines" / "summary.json"
        baselines = json.loads(baselines_path.read_text(encoding="utf-8")) if baselines_path.exists() else evaluate_native_baselines(cfg, model)
        training = json.loads((run_root(cfg) / "stage_b" / "training_summary.json").read_text(encoding="utf-8"))
        translated = {}
        for alias in ("best_accuracy", "best_choice_kl"):
            candidate = training[alias]
            module, payload = load_checkpoint(cfg, candidate["path"])
            translated[alias] = {"checkpoint_step": payload["step"], **evaluate_translated(cfg, model, module, alias)}
            del module; torch.cuda.empty_cache()
    finally:
        del model; torch.cuda.empty_cache()
    primary = translated["best_accuracy"]["functional"]
    main_table = {
        "Qwen Full Native": baselines["qwen_full_native"],
        "Llama Full Native": baselines["llama_full_native"],
        "Qwen Self-Selected32 Native": baselines["qwen_self_selected32_native"],
        "Llama-selected -> Qwen Native Oracle": baselines["llama_selected_qwen_native_oracle"],
        "Translated Full28-Diagonal": primary["translated"],
        "Shuffle": primary["shuffle"], "Zero": baselines["zero"], "No-memory": baselines["no_memory"]}
    qwen_full = main_table["Qwen Full Native"]["accuracy"]
    oracle = main_table["Llama-selected -> Qwen Native Oracle"]["accuracy"]
    translated_accuracy = main_table["Translated Full28-Diagonal"]["accuracy"]
    zero, no_memory = main_table["Zero"]["accuracy"], main_table["No-memory"]["accuracy"]
    comparison = {"experiment": "mmlupro_sender_qoq_posterior_question_rawanchor_full28_diagonal_llama3_2_3b_to_qwen3_4b_seed1234",
        "status": "completed", "prompt": "Question -> Options -> Answer:",
        "routing_query": "last token of repeated posterior Sender Question",
        "sender_prompt": "Question -> Options -> repeated Question",
        "selection": "32 option-only raw-text regions",
        "receiver_protocol": "native Question + external selected Options KV + native Answer:",
        "checkpoint_selection": "validation only; primary=best validation accuracy",
        "main_table": main_table, "translated_checkpoints": translated,
        "mean_random_chance": baselines["mean_random_chance"],
        "gaps": {"selection_gap": qwen_full - oracle, "translation_gap": oracle - translated_accuracy,
                 "information_gain_vs_zero": translated_accuracy - zero,
                 "information_gain_vs_no_memory": translated_accuracy - no_memory,
                 "oracle_retention_vs_zero": safe_retention(translated_accuracy, oracle, zero),
                 "oracle_retention_vs_no_memory": safe_retention(translated_accuracy, oracle, no_memory)}}
    save_json(run_root(cfg) / "results" / "comparison.json", comparison)
    append_results_document(cfg, comparison)
    log("POSTERIOR-QUESTION SENDER ROUTING EXPERIMENT COMPLETED")
    return comparison
