from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


ROOT = Path(__file__).resolve().parent


def cfg_load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def load_base(cfg):
    path = Path(cfg["base_experiment_dir"]) / "experiment.py"
    spec = importlib.util.spec_from_file_location("native4_writer8_base", path)
    module = importlib.util.module_from_spec(spec)
    old = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    return module


def writer_class(cfg):
    path = Path(cfg["base_experiment_dir"]) / "writer.py"
    spec = importlib.util.spec_from_file_location("native4_writer8_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Native4ChannelWriter


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dev():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def stats(cfg, mode):
    return torch.load(
        Path(cfg["base_experiment_dir"])
        / "artifacts"
        / mode
        / "protocol"
        / "fixed_scales.pt",
        map_location="cpu",
        weights_only=False,
    )


def make_writer(cfg, mode, initialization):
    model = writer_class(cfg)(cfg, stats(cfg, mode), "linear").to(dev())
    if initialization == "stage_a":
        checkpoint = torch.load(
            Path(cfg["base_experiment_dir"])
            / "artifacts"
            / mode
            / "linear"
            / "stage_a"
            / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["writer"])
    if model.zero_check() != 0.0:
        raise RuntimeError("Writer(0) != 0")
    return model


def answer_logits(cfg, r1, reader, tok, sample, key=None, value=None, mask=None, compact=False):
    question = sample["question_token_ids"]
    target = r1.answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
    ids = torch.tensor([question + target[:-1]], dtype=torch.long, device=dev())
    qpos = list(range(len(question))) if compact else sample["question_position_ids"]
    apos = list(range(qpos[-1] + 1, qpos[-1] + len(target)))
    kwargs = {}
    if key is not None:
        attention = torch.cat([mask.to(dev()), torch.ones_like(ids)], 1)
        kwargs = {
            "past_key_values": r1.dynamic_cache(
                reader, key, value, sample["selected_position_ids"]
            ),
            "cache_position": torch.arange(
                key.shape[1], key.shape[1] + ids.shape[1], device=dev()
            ),
        }
    else:
        attention = torch.ones_like(ids)
    logits = reader(
        ids,
        attention_mask=attention,
        position_ids=torch.tensor([qpos + apos], device=dev()),
        use_cache=False,
        **kwargs,
    ).logits
    selected = logits[:, len(question) - 1 : len(question) - 1 + len(target)].float()[0]
    gold = torch.tensor(target, dtype=torch.long, device=dev())
    return selected, gold


def protocol_audit(cfg, mode):
    base = load_base(cfg)
    rows = base.rows_for(cfg, mode)
    checkpoint = (
        Path(cfg["r1_dir"])
        / "artifacts"
        / base.source_mode(mode)
        / "sparse_reader"
        / "best.pt"
    )
    previous = (
        Path(cfg["base_experiment_dir"])
        / "artifacts"
        / mode
        / "linear"
        / "stage_a"
        / "best.pt"
    )
    if not checkpoint.exists() or not previous.exists():
        raise RuntimeError("required frozen Reader or Stage-A initialization is missing")
    report = {
        "passed": True,
        "train": len(rows["train"]),
        "validation": len(rows["validation"]),
        "test": len(rows["test"]),
        "reader_checkpoint": str(checkpoint),
        "reader_receiver_lora_frozen": True,
        "writer_input": "8B evidence KV only; no question",
        "channel_shape": [36, "T<=128", 8, 128],
        "f0": "scale-only, no training",
        "f1": "scale-only initialization, functional-only training",
        "f2": "previous linear Stage-A initialization, functional-only training",
        "optimized_losses": ["answer_nll", "native_output_distillation", "dependence_margin"],
        "diagnostic_only": ["kv", "cosine", "route", "attention_output"],
        "hard_gates_enforced": False,
    }
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "protocol_audit.json", report)
    progress(f"{mode}: functional-only protocol audit passed")


@torch.no_grad()
def teacher_cache(cfg, mode):
    seed_all(cfg["seed"])
    base = load_base(cfg)
    rows = base.rows_for(cfg, mode)
    store = base.Stores(cfg, mode, rows)
    r1, reader, tok = base.load_reader(cfg, mode)
    temperature = cfg["distill_temperature"]
    topk = cfg["distill_topk"]
    root = Path(cfg["work_dir"]) / "cache" / mode / "native_teacher"
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        records = {}
        for index, sample in enumerate(rows[split], 1):
            key, value, mask = store.memory(split, "4b", sample, "correct")
            with torch.autocast("cuda", dtype=torch.float16):
                logits, gold = answer_logits(
                    cfg,
                    r1,
                    reader,
                    tok,
                    sample,
                    key.to(dev()),
                    value.to(dev()),
                    mask,
                )
            values, indices = logits.topk(topk, -1)
            scaled = logits / temperature
            records[sample["id"]] = {
                "top_indices": indices.cpu().to(torch.int32),
                "top_logits": values.cpu().half(),
                "scaled_logsumexp": scaled.logsumexp(-1).cpu(),
                "gold_probability": logits.softmax(-1).gather(1, gold[:, None]).squeeze(1).cpu(),
                "answer_tokens": gold.cpu(),
            }
            if index % 32 == 0 or index == len(rows[split]):
                progress(f"{mode}: Native teacher {split} {index}/{len(rows[split])}")
        torch.save(records, root / f"{split}.pt")
    del reader
    torch.cuda.empty_cache()


class TeacherStore:
    def __init__(self, cfg, mode):
        root = Path(cfg["work_dir"]) / "cache" / mode / "native_teacher"
        self.records = {
            split: torch.load(root / f"{split}.pt", map_location="cpu", weights_only=False)
            for split in ("train", "validation")
        }

    def get(self, split, sample_id):
        return self.records[split][sample_id]


def sparse_distillation(student_logits, teacher, temperature):
    indices = teacher["top_indices"].long().to(student_logits.device)
    teacher_top_logits = teacher["top_logits"].float().to(student_logits.device)
    teacher_lse = teacher["scaled_logsumexp"].float().to(student_logits.device)
    student_scaled = student_logits.float() / temperature
    student_lse = student_scaled.logsumexp(-1)
    teacher_logp = teacher_top_logits / temperature - teacher_lse[:, None]
    student_logp = student_scaled.gather(1, indices) - student_lse[:, None]
    teacher_p = teacher_logp.exp()
    student_p = student_logp.exp()
    teacher_rest = (1 - teacher_p.sum(-1)).clamp_min(1e-8)
    student_rest = (1 - student_p.sum(-1)).clamp_min(1e-8)
    kl_top = (teacher_p * (teacher_logp - student_logp)).sum(-1)
    kl_rest = teacher_rest * (teacher_rest.log() - student_rest.log())
    return (kl_top + kl_rest).mean() * (temperature**2)


def writer_memory(cfg, base, writer, store, split, sample, kind):
    key, value, mask = store.memory(split, "8b", sample, kind)
    return (*writer(key.to(dev()), value.to(dev())), mask)


@torch.no_grad()
def validation_metrics(cfg, mode, base, r1, writer, store, teacher, reader, tok, samples, generation):
    writer.eval()
    answer_values, shuffled_values, distill_values = [], [], []
    predictions = []
    for sample in samples:
        key, value, mask = writer_memory(cfg, base, writer, store, "validation", sample, "correct")
        logits, gold = answer_logits(cfg, r1, reader, tok, sample, key.half(), value.half(), mask)
        answer_values.append(F.cross_entropy(logits, gold).item())
        distill_values.append(
            sparse_distillation(
                logits, teacher.get("validation", sample["id"]), cfg["distill_temperature"]
            ).item()
        )
        wrong_k, wrong_v, wrong_mask = writer_memory(
            cfg, base, writer, store, "validation", sample, "shuffled"
        )
        wrong_logits, _ = answer_logits(
            cfg, r1, reader, tok, sample, wrong_k.half(), wrong_v.half(), wrong_mask
        )
        shuffled_values.append(F.cross_entropy(wrong_logits, gold).item())
    if generation:
        limit = len(samples) if mode == "smoke" else min(cfg["generation_probe_size"], len(samples))
        for sample in samples[:limit]:
            key, value, mask = writer_memory(
                cfg, base, writer, store, "validation", sample, "correct"
            )
            prediction = r1.greedy_generate(
                cfg, reader, tok, sample, key.half(), value.half(), mask
            )
            predictions.append(
                {
                    "em": float(
                        r1.normalize_answer(prediction)
                        == r1.normalize_answer(sample["answer"])
                    ),
                    "f1": r1.token_f1(prediction, sample["answer"]),
                }
            )
    return {
        "answer_nll": sum(answer_values) / len(answer_values),
        "shuffled_nll": sum(shuffled_values) / len(shuffled_values),
        "correct_shuffled_nll_gap": (
            sum(shuffled_values) / len(shuffled_values)
            - sum(answer_values) / len(answer_values)
        ),
        "distill_kl": sum(distill_values) / len(distill_values),
        "generation_em": (
            sum(x["em"] for x in predictions) / len(predictions) if predictions else None
        ),
        "generation_f1": (
            sum(x["f1"] for x in predictions) / len(predictions) if predictions else None
        ),
        "generation_probe_count": len(predictions),
    }


def save_f0(cfg, mode):
    writer = make_writer(cfg, mode, "scale")
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "f0_scale_only"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"writer": writer.state_dict(), "update": 0, "trained": False}, out / "best.pt")
    save_json(out / "summary.json", {"completed": True, "trained": False, "initialization": "scale-only"})


def train_functional(cfg, mode, group):
    seed_all(cfg["seed"])
    base = load_base(cfg)
    rows = base.rows_for(cfg, mode)
    store = base.Stores(cfg, mode, rows)
    teacher = TeacherStore(cfg, mode)
    r1, reader, tok = base.load_reader(cfg, mode)
    initialization = "scale" if group == "f1" else "stage_a"
    writer = make_writer(cfg, mode, initialization)
    optimizer = AdamW(
        writer.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["functional_updates"]
    warmup = min(maximum, cfg["warmup_updates"])
    scheduler = LambdaLR(
        optimizer, lambda step: min((step + 1) / max(warmup, 1), 1.0)
    )
    interval = 1 if mode == "smoke" else cfg["eval_interval"]
    generation_interval = 1 if mode == "smoke" else cfg["generation_eval_interval"]
    out = Path(cfg["work_dir"]) / "artifacts" / mode / group
    out.mkdir(parents=True, exist_ok=True)
    initial = validation_metrics(
        cfg, mode, base, r1, writer, store, teacher, reader, tok, rows["validation"], True
    )
    best_score = (
        initial["answer_nll"],
        -initial["correct_shuffled_nll_gap"],
        -(initial["generation_em"] or 0),
        -(initial["generation_f1"] or 0),
    )
    torch.save(
        {"writer": writer.state_dict(), "update": 0, "validation": initial},
        out / "best.pt",
    )
    history, evaluations = [], [{"update": 0, **initial, "selected": True}]
    samples, cursor, epoch = list(rows["train"]), 0, 0
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        micro = []
        use_dependence = update % cfg["dependence_every"] == 0
        for _ in range(cfg["gradient_accumulation"]):
            if cursor == 0:
                random.Random(cfg["seed"] + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            writer.train()
            key, value, mask = writer_memory(
                cfg, base, writer, store, "train", sample, "correct"
            )
            logits, gold = answer_logits(
                cfg, r1, reader, tok, sample, key.half(), value.half(), mask
            )
            answer = F.cross_entropy(logits, gold)
            distill = sparse_distillation(
                logits, teacher.get("train", sample["id"]), cfg["distill_temperature"]
            )
            dependence = torch.zeros((), device=dev())
            shuffled_nll = torch.full((), float("nan"), device=dev())
            if use_dependence:
                wrong_k, wrong_v, wrong_mask = writer_memory(
                    cfg, base, writer, store, "train", sample, "shuffled"
                )
                wrong_logits, _ = answer_logits(
                    cfg, r1, reader, tok, sample, wrong_k.half(), wrong_v.half(), wrong_mask
                )
                shuffled_nll = F.cross_entropy(wrong_logits, gold)
                dependence = F.relu(
                    cfg["dependence_margin"] + answer - shuffled_nll
                )
            loss = (
                answer
                + cfg["distill_weight"] * distill
                + cfg["dependence_weight"] * dependence
            )
            (loss / cfg["gradient_accumulation"]).backward()
            micro.append(
                {
                    "loss": loss.detach().item(),
                    "answer_nll": answer.detach().item(),
                    "distill_kl": distill.detach().item(),
                    "dependence": dependence.detach().item(),
                    "shuffled_nll": shuffled_nll.detach().item(),
                }
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            writer.parameters(), cfg["gradient_clip"]
        ).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(
            {
                "update": update,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
                "dependence_computed": use_dependence,
                **{
                    key: sum(x[key] for x in micro if not torch.isnan(torch.tensor(x[key]))) /
                    max(1, sum(not torch.isnan(torch.tensor(x[key])) for x in micro))
                    for key in micro[0]
                },
            }
        )
        if update % interval == 0 or update == maximum:
            generation = update % generation_interval == 0 or update == maximum
            validation = validation_metrics(
                cfg,
                mode,
                base,
                r1,
                writer,
                store,
                teacher,
                reader,
                tok,
                rows["validation"],
                generation,
            )
            score = (
                validation["answer_nll"],
                -validation["correct_shuffled_nll_gap"],
                -(validation["generation_em"] or 0),
                -(validation["generation_f1"] or 0),
            )
            selected = score < best_score
            if selected:
                best_score = score
                torch.save(
                    {"writer": writer.state_dict(), "update": update, "validation": validation},
                    out / "best.pt",
                )
            evaluations.append({"update": update, **validation, "selected": selected})
            save_json(out / "history.json", history)
            save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}/{group}: functional-only {update}/{maximum}")
    save_json(
        out / "summary.json",
        {
            "completed": True,
            "updates": maximum,
            "initialization": initialization,
            "optimized_losses": ["answer_nll", "distill_kl", "dependence"],
            "representation_losses_used_for_backprop": False,
            "hard_gates_enforced": False,
        },
    )
    del reader
    torch.cuda.empty_cache()


def load_group_writer(cfg, mode, group):
    if group == "kv_stage_a":
        writer = make_writer(cfg, mode, "stage_a")
        return writer.eval()
    writer = make_writer(cfg, mode, "scale")
    checkpoint = torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / group / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    writer.load_state_dict(checkpoint["writer"])
    return writer.eval()


def aggregate(rows):
    report = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        report[condition] = {
            key: sum(float(row[key]) for row in selected) / len(selected)
            for key in ("em", "token_f1", "nll")
        }
        report[condition]["count"] = len(selected)
    return report


@torch.no_grad()
def evaluate(cfg, mode):
    seed_all(cfg["seed"])
    base = load_base(cfg)
    rows = base.rows_for(cfg, mode)
    store = base.Stores(cfg, mode, rows)
    r1, reader, tok = base.load_reader(cfg, mode)
    groups = {
        name: load_group_writer(cfg, mode, name)
        for name in ("f0_scale_only", "f1", "f2", "kv_stage_a")
    }
    output_rows, diagnostics = [], {}
    for name, writer in groups.items():
        diagnostic_rows = []
        for kind in ("correct", "shuffled", "zero"):
            condition = f"{name}_{kind}"
            progress(f"{mode}: evaluate {condition}")
            for sample in rows["test"]:
                key, value, mask = writer_memory(
                    cfg, base, writer, store, "test", sample, kind
                )
                prediction = r1.greedy_generate(
                    cfg, reader, tok, sample, key.half(), value.half(), mask
                )
                logits, gold = answer_logits(
                    cfg, r1, reader, tok, sample, key.half(), value.half(), mask
                )
                output_rows.append(
                    {
                        "sample_id": sample["id"],
                        "type": sample.get("type", "unknown"),
                        "condition": condition,
                        "answer": sample["answer"],
                        "prediction": prediction,
                        "em": float(
                            r1.normalize_answer(prediction)
                            == r1.normalize_answer(sample["answer"])
                        ),
                        "token_f1": r1.token_f1(prediction, sample["answer"]),
                        "nll": F.cross_entropy(logits, gold).item(),
                        "manual_c_p_w": "",
                    }
                )
                if kind == "correct":
                    k4, v4, _ = store.memory("test", "4b", sample, "correct")
                    values = base.representation_losses(
                        cfg,
                        store.query4("test", sample["id"]),
                        (key, value),
                        (k4.to(dev()), v4.to(dev())),
                        sample["selected_position_ids"],
                    )
                    diagnostic_rows.append(
                        {
                            metric: tensor.item()
                            for metric, tensor in values.items()
                            if metric not in {"stage_a", "stage_b_rep"}
                        }
                    )
        diagnostics[name] = {
            metric: sum(row[metric] for row in diagnostic_rows) / len(diagnostic_rows)
            for metric in diagnostic_rows[0]
        }
    summary = aggregate(output_rows)
    previous = json.loads(
        (
            Path(cfg["base_experiment_dir"])
            / "artifacts"
            / mode
            / "evaluation"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    summary["reference_conditions"] = {
        key: previous[key]
        for key in (
            "question_only",
            "native4_correct",
            "native4_shuffled",
            "raw8_correct",
            "raw8_shuffled",
            "reader_off",
        )
    }
    summary["continuous_representation_diagnostics"] = diagnostics
    for name in groups:
        correct = summary[f"{name}_correct"]
        shuffled = summary[f"{name}_shuffled"]
        summary[f"{name}_comparison"] = {
            "correct_shuffled_em_gap": correct["em"] - shuffled["em"],
            "correct_shuffled_nll_gap": shuffled["nll"] - correct["nll"],
            "hard_gate": False,
        }
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "evaluation"
    save_json(out / "summary.json", summary)
    save_json(out / "per_sample.json", output_rows)
    save_json(
        out / "completion.json",
        {
            "completed": True,
            "hard_gates_enforced": False,
            "single_question_per_context_limitation": True,
            "multi_query_reusability_not_claimed": True,
        },
    )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manual_c_p_w.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument(
        "action",
        choices=("audit", "teacher_cache", "f0", "train", "evaluate"),
    )
    parser.add_argument("--group", choices=("f1", "f2"))
    args = parser.parse_args()
    cfg = cfg_load(args.config)
    if args.action == "audit":
        protocol_audit(cfg, args.mode)
    elif args.action == "teacher_cache":
        teacher_cache(cfg, args.mode)
    elif args.action == "f0":
        save_f0(cfg, args.mode)
    elif args.action == "train":
        if not args.group:
            parser.error("--group is required")
        train_functional(cfg, args.mode, args.group)
    else:
        evaluate(cfg, args.mode)


if __name__ == "__main__":
    main()
