from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import load_model, log, read_json, run_root, save_json, save_tensor, seed_all
from protocol import final_logits
from translator import NativeKVTranslator


EXPERIMENT = "mmlupro_memoryfirst_receiver_oracle_audit_llama3_2_3b_to_qwen3_4b_seed1234"


def source_run(cfg):
    return Path(cfg["source_experiment_root"]) / "runs" / "study"


def source_configuration(cfg):
    payload = read_json(source_run(cfg) / "run_config.json")
    if payload.get("protocol") != "receiver_native_question_options_only_v1":
        raise RuntimeError("Unexpected source protocol")
    if payload.get("architecture") != cfg["architecture"]:
        raise RuntimeError("Source architecture mismatch")
    return payload


def rows(cfg, split):
    source_cfg = source_configuration(cfg)
    payload = read_json(source_run(cfg) / "manifests" / f"{split}.json")
    if payload.get("signature") != source_cfg["signature"]:
        raise RuntimeError("Source manifest signature mismatch")
    return payload["rows"][: cfg[f"{split}_samples"]]


def source_payload(cfg, split, row, kind):
    directory = {"source": "source_selected_cache", "pair": "pair_cache"}[kind]
    payload = torch.load(source_run(cfg) / directory / split / f"{row['id']}.pt",
                         map_location="cpu", weights_only=True)
    if payload.get("signature") != source_configuration(cfg)["signature"]:
        raise RuntimeError(f"Source {kind} cache signature mismatch")
    return payload


def choice_kl(student_choice, teacher_choice, temperature):
    s = F.log_softmax(student_choice.float() / temperature, -1)
    t = F.log_softmax(teacher_choice.to(student_choice.device).float() / temperature, -1)
    return F.kl_div(s, t.detach(), reduction="sum", log_target=True) * temperature ** 2


def receiver_suffix(row):
    fields = row["encoded"]["qwen"]
    question = fields["question_prefix"]
    if not question or question[0] != fields["body"][0] or fields["full"][0] != fields["body"][0]:
        raise RuntimeError(f"Receiver token0 protocol mismatch: {row['id']}")
    # token0 is supplied by the native memory anchor. The remaining Question + Options header
    # and the Receiver-native Answer: are processed normally after the latent memory.
    return question[1:] + fields["receiver_answer"]


def memory_first_logits(model, row, key, value):
    if key.shape[1] != 33 or value.shape[1] != 33:
        raise RuntimeError(f"Expected token0 + 32 memory KVs, got {key.shape[1]}")
    positions = torch.arange(33, device="cuda")
    return final_logits(model, receiver_suffix(row), key, value,
                        positions=positions, suffix_start=33)


def question_first_logits(model, row, pair):
    key = torch.cat((pair["question_k"].cuda(), pair["target_k"][:, 1:].cuda()), 1)
    value = torch.cat((pair["question_v"].cuda(), pair["target_v"][:, 1:].cuda()), 1)
    fields = row["encoded"]["qwen"]
    return final_logits(model, fields["receiver_answer"], key, value,
                        positions=torch.arange(key.shape[1], device="cuda"), suffix_start=key.shape[1])


def oracle_memory(pair):
    return pair["target_k"].cuda(), pair["target_v"].cuda()


def translated_memory(module, source, pair):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        key, value = module(source["source_k"][:, 1:].cuda()[None],
                            source["source_v"][:, 1:].cuda()[None])
    return (torch.cat((pair["target_k"][:, :1].cuda(), key[0]), 1),
            torch.cat((pair["target_v"][:, :1].cuda(), value[0]), 1))


@torch.no_grad()
def oracle_audit(cfg):
    model = load_model(cfg, "qwen")
    totals = {name: 0 for name in ("question_first_correct", "memory_first_correct", "agreement")}
    records = []
    try:
        selected = rows(cfg, "test")
        for number, row in enumerate(selected, 1):
            pair = source_payload(cfg, "test", row, "pair")
            ids = row["encoded"]["qwen"]["choice_ids"]
            q_choice = question_first_logits(model, row, pair)[ids].float()
            m_choice = memory_first_logits(model, row, *oracle_memory(pair))[ids].float()
            q_pred, m_pred, gold = int(q_choice.argmax()), int(m_choice.argmax()), int(row["gold_index"])
            totals["question_first_correct"] += q_pred == gold
            totals["memory_first_correct"] += m_pred == gold
            totals["agreement"] += q_pred == m_pred
            records.append({"id": row["id"], "gold_index": gold,
                            "question_first_prediction": q_pred, "memory_first_prediction": m_pred,
                            "question_first_choice_logits": q_choice.cpu().tolist(),
                            "memory_first_choice_logits": m_choice.cpu().tolist(),
                            "memory_vs_question_first_choice_kl": choice_kl(
                                m_choice, q_choice, cfg["temperature"]).item()})
            if number % 16 == 0 or number == len(selected):
                log(f"Memory-first Oracle audit: {number}/{len(selected)}")
    finally:
        del model; torch.cuda.empty_cache()
    count = len(records)
    gain_correct = totals["memory_first_correct"] - totals["question_first_correct"]
    passed = gain_correct >= cfg["oracle_min_gain_correct"]
    summary = {"experiment": EXPERIMENT, "status": "completed", "sample_count": count,
               "fixed_router": "Answer-query + 32 option-region Top1",
               "fixed_alignment": "RawAnchor", "external_memory": "Qwen-native token0 + 32 selected option KVs",
               "question_first": {"correct": totals["question_first_correct"],
                                  "accuracy": totals["question_first_correct"] / count},
               "memory_first": {"correct": totals["memory_first_correct"],
                                "accuracy": totals["memory_first_correct"] / count},
               "layout_agreement": totals["agreement"] / count,
               "gain_correct": gain_correct, "gain_accuracy": gain_correct / count,
               "gate": {"minimum_gain_correct": cfg["oracle_min_gain_correct"], "passed": passed}}
    root = run_root(cfg) / "oracle_audit"; root.mkdir(parents=True, exist_ok=True)
    save_json(root / "summary.json", summary)
    with (root / "per_sample.jsonl").open("w", encoding="utf-8") as output:
        for record in records: output.write(json.dumps(record, ensure_ascii=False) + "\n")
    log(f"MEMORY-FIRST ORACLE AUDIT COMPLETED: {summary}")
    return summary


def teacher_path(cfg, split, row):
    return run_root(cfg) / "memory_first_teacher" / split / f"{row['id']}.pt"


def load_teacher(cfg, split, row):
    payload = torch.load(teacher_path(cfg, split, row), map_location="cpu", weights_only=True)
    if payload.get("signature") != cfg["signature"]:
        raise RuntimeError("Memory-first teacher signature mismatch")
    return payload


@torch.no_grad()
def prepare_teachers(cfg):
    model = load_model(cfg, "qwen")
    try:
        for split in ("train", "validation", "test"):
            selected = rows(cfg, split)
            for number, row in enumerate(selected, 1):
                path = teacher_path(cfg, split, row)
                if path.exists():
                    try:
                        load_teacher(cfg, split, row); continue
                    except RuntimeError:
                        pass
                pair = source_payload(cfg, split, row, "pair")
                ids = row["encoded"]["qwen"]["choice_ids"]
                choice = memory_first_logits(model, row, *oracle_memory(pair))[ids].cpu()
                save_tensor(path, {"signature": cfg["signature"], "choice_logits": choice,
                                   "prediction": int(choice.argmax())})
                if number % 32 == 0 or number == len(selected):
                    log(f"Memory-first teacher {split}: {number}/{len(selected)}")
    finally:
        del model; torch.cuda.empty_cache()


def load_stage_a(cfg):
    selection = read_json(source_run(cfg) / "stage_a" / "selection.json")
    candidate = selection["best_accuracy"]
    payload = torch.load(candidate["path"], map_location="cpu", weights_only=True)
    source_cfg = source_configuration(cfg)
    if payload.get("signature") != source_cfg["signature"] or payload.get("architecture") != cfg["architecture"]:
        raise RuntimeError("Source Stage-A checkpoint mismatch")
    module = NativeKVTranslator(cfg["architecture"]).cuda()
    module.load_state_dict(payload["state"], strict=True)
    return module, {"step": payload["step"], "path": candidate["path"]}


def save_checkpoint(cfg, module, step, validation):
    path = run_root(cfg) / "stage_b" / "checkpoints" / f"step_{step}.pt"
    save_tensor(path, {"signature": cfg["signature"], "architecture": cfg["architecture"],
                       "step": step, "validation": validation,
                       "state": {name: value.detach().cpu() for name, value in module.state_dict().items()}})
    return path


def load_checkpoint(cfg, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("signature") != cfg["signature"]:
        raise RuntimeError("Stage-B checkpoint mismatch")
    module = NativeKVTranslator(cfg["architecture"]).cuda()
    module.load_state_dict(payload["state"], strict=True)
    return module, payload


@torch.no_grad()
def functional_metrics(cfg, model, module, split, save_records=False):
    module.eval(); selected = rows(cfg, split)
    totals = {key: 0.0 for key in ("accuracy", "oracle_accuracy", "oracle_agreement", "oracle_choice_kl")}
    records = []
    for number, row in enumerate(selected, 1):
        source = source_payload(cfg, split, row, "source")
        pair = source_payload(cfg, split, row, "pair")
        teacher = load_teacher(cfg, split, row)
        ids, gold = row["encoded"]["qwen"]["choice_ids"], int(row["gold_index"])
        choice = memory_first_logits(model, row, *translated_memory(module, source, pair))[ids]
        prediction, oracle_prediction = int(choice.argmax()), int(teacher["prediction"])
        values = {"accuracy": float(prediction == gold),
                  "oracle_accuracy": float(oracle_prediction == gold),
                  "oracle_agreement": float(prediction == oracle_prediction),
                  "oracle_choice_kl": choice_kl(choice, teacher["choice_logits"], cfg["temperature"]).item()}
        for key, value in values.items(): totals[key] += value
        if save_records:
            records.append({"id": row["id"], "gold_index": gold, "prediction": prediction,
                            "oracle_prediction": oracle_prediction, "choice_logits": choice.cpu().tolist(), **values})
        if save_records and (number % 16 == 0 or number == len(selected)):
            log(f"Memory-first translated evaluation: {number}/{len(selected)}")
    return ({key: value / len(selected) for key, value in totals.items()}, records)


def train_stage_b(cfg):
    seed_all(cfg["seed"] + 100); module, initialization = load_stage_a(cfg)
    model = load_model(cfg, "qwen"); train_rows = rows(cfg, "train")
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
                    source = source_payload(cfg, "train", row, "source")
                    pair = source_payload(cfg, "train", row, "pair")
                    teacher = load_teacher(cfg, "train", row)
                    ids = row["encoded"]["qwen"]["choice_ids"]
                    choice = memory_first_logits(model, row, *translated_memory(module, source, pair))[ids]
                    loss = choice_kl(choice, teacher["choice_logits"], cfg["temperature"])
                    if not torch.isfinite(loss): raise RuntimeError("Nonfinite Stage-B loss")
                    scaler.scale(loss / len(indices)).backward(); losses.append(loss.item())
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
                if not torch.isfinite(norm): raise RuntimeError("Invalid Stage-B gradient")
                clipped += int(norm.item() > cfg["clip"]); scaler.step(optimizer); scaler.update(); step += 1
                record = {"epoch": epoch, "step": step, "mean_choice_kl": sum(losses) / len(losses),
                          "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
                stream.write(json.dumps(record) + "\n"); stream.flush()
                if step % 16 == 0: log(f"Memory-first Stage B {step}/{cfg['stage_b_steps']} KL={record['mean_choice_kl']:.6f}")
                if step % cfg["validation_interval_steps"] == 0 or step == cfg["stage_b_steps"]:
                    validation, _ = functional_metrics(cfg, model, module, "validation")
                    path = save_checkpoint(cfg, module, step, validation)
                    candidates.append({"step": step, "path": str(path), "functional": validation})
                    save_json(root / "candidates.json", candidates)
                    log(f"Memory-first Stage B validation step={step} acc={validation['accuracy']:.4f} KL={validation['oracle_choice_kl']:.6f}")
                if step >= cfg["stage_b_steps"]: break
            if step >= cfg["stage_b_steps"]: break
    finally:
        stream.close()
    best_accuracy = max(candidates, key=lambda x: (x["functional"]["accuracy"], -x["functional"]["oracle_choice_kl"], -x["step"]))
    best_kl = min(candidates, key=lambda x: (x["functional"]["oracle_choice_kl"], x["step"]))
    summary = {"stage": "B", "objective": "memory-first Oracle A-J choice KL only",
               "initialized_from_source_stage_a": initialization, "steps": step,
               "clip_rate": clipped / max(step, 1), "seconds": time.monotonic() - started,
               "selection": "validation only", "best_accuracy": best_accuracy,
               "best_choice_kl": best_kl, "candidates": candidates}
    save_json(root / "training_summary.json", summary)
    del module, model, optimizer, scaler; torch.cuda.empty_cache()
    return summary


def append_results(cfg, result):
    if cfg["mode"] != "study": return
    path = Path(cfg["shared_results_document"]); path.parent.mkdir(parents=True, exist_ok=True)
    begin, end = f"<!-- BEGIN {EXPERIMENT} -->", f"<!-- END {EXPERIMENT} -->"
    audit = result["oracle_audit"]
    lines = [begin, f"## {EXPERIMENT}", "", "Machine-generated after successful completion.", "",
             "| Layout | Correct | Accuracy |", "|---|---:|---:|",
             f"| Question-first Native Oracle | {audit['question_first']['correct']}/128 | {100*audit['question_first']['accuracy']:.2f}% |",
             f"| Memory-first Native Oracle | {audit['memory_first']['correct']}/128 | {100*audit['memory_first']['accuracy']:.2f}% |",
             "", f"Oracle gain={audit['gain_correct']} correct ({100*audit['gain_accuracy']:.2f} points); "
             f"Stage-B gate={'passed' if audit['gate']['passed'] else 'failed'}."]
    for name, item in result.get("translated", {}).items():
        metrics = item["functional"]
        lines.append(f"- {name} (step {item['step']}): accuracy={100*metrics['accuracy']:.2f}%, "
                     f"agreement={100*metrics['oracle_agreement']:.2f}%, choice-KL={metrics['oracle_choice_kl']:.6f}")
    lines += [end, ""]
    section = "\n".join(lines)
    current = path.read_text(encoding="utf-8") if path.exists() else "# Experiment Results\n\n"
    if begin in current and end in current:
        left, tail = current.split(begin, 1); _, right = tail.split(end, 1)
        current = left.rstrip() + "\n\n" + section + right.lstrip("\n")
    else: current = current.rstrip() + "\n\n" + section
    path.write_text(current, encoding="utf-8")


def finalize(cfg):
    audit = read_json(run_root(cfg) / "oracle_audit" / "summary.json")
    training_path = run_root(cfg) / "stage_b" / "training_summary.json"
    result = {"experiment": EXPERIMENT, "status": "completed", "oracle_audit": audit,
              "stage_b_executed": training_path.exists(),
              "stage_b_trigger": "forced_by_user" if training_path.exists() and not audit["gate"]["passed"]
                                 else "oracle_gate" if training_path.exists() else "not_run"}
    if training_path.exists():
        training = read_json(training_path); model = load_model(cfg, "qwen"); translated = {}
        try:
            for alias in ("best_accuracy", "best_choice_kl"):
                candidate = training[alias]; module, payload = load_checkpoint(cfg, candidate["path"])
                functional, records = functional_metrics(cfg, model, module, "test", save_records=True)
                translated[alias] = {"step": payload["step"], "functional": functional}
                root = run_root(cfg) / "evaluation" / alias; root.mkdir(parents=True, exist_ok=True)
                save_json(root / "summary.json", translated[alias])
                with (root / "per_sample.jsonl").open("w", encoding="utf-8") as output:
                    for record in records: output.write(json.dumps(record, ensure_ascii=False) + "\n")
                del module; torch.cuda.empty_cache()
        finally:
            del model; torch.cuda.empty_cache()
        result["translated"] = translated
    save_json(run_root(cfg) / "results" / "comparison.json", result)
    append_results(cfg, result)
    log("MEMORY-FIRST RECEIVER EXPERIMENT COMPLETED")
    return result
