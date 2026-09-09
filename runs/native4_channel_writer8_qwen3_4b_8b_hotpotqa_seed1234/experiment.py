from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parent
from writer import Native4ChannelWriter


def load_cfg(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def source_mode(mode):
    return "smoke" if mode == "smoke" else "formal"


def query_mode(mode):
    return "smoke" if mode == "smoke" else "development"


def r1_module(cfg):
    path = str(Path(cfg["r1_dir"]))
    if path not in sys.path:
        sys.path.insert(0, path)
    import r1_common

    return r1_common


def rows_for(cfg, mode):
    source = source_mode(mode)
    rows = json.loads(
        (Path(cfg["r1_dir"]) / "artifacts" / source / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    sizes = cfg["smoke_sizes"] if mode == "smoke" else cfg["sizes"]
    selected = {split: list(rows[split][: sizes[split]]) for split in sizes}
    limit = cfg["max_evidence_tokens"]
    for split, samples in selected.items():
        for sample in samples:
            sample["selected_position_ids"] = sample["selected_position_ids"][:limit]
            sample["selected_token_count"] = len(sample["selected_position_ids"])
        for index, sample in enumerate(samples):
            candidates = [
                other
                for other in samples
                if other["id"] != sample["id"]
                and other["answer"].strip().lower() != sample["answer"].strip().lower()
            ]
            sample["shuffle_id"] = candidates[(index * 17 + 7) % len(candidates)]["id"]
    return selected


class Stores:
    def __init__(self, cfg, mode, rows):
        self.cfg, self.mode, self.rows = cfg, mode, rows
        self.positions = {
            split: {sample["id"]: index for index, sample in enumerate(samples)}
            for split, samples in rows.items()
        }
        self.cache = {}

    def _load(self, key, path):
        if key not in self.cache:
            if len(self.cache) > 12:
                self.cache.clear()
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.cache[key]

    def native(self, split, sender, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["shard_size"]
        path = (
            Path(self.cfg["r1_dir"])
            / "cache"
            / source_mode(self.mode)
            / split
            / sender
            / f"shard_{shard:05d}.pt"
        )
        record = self._load(("native", split, sender, shard), path)[
            index % self.cfg["shard_size"]
        ]
        if record["id"] != sample_id:
            raise RuntimeError(f"native shard mismatch: {sample_id}")
        return record

    def query4(self, split, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["query_shard_size"]
        path = (
            Path(self.cfg["a0_dir"])
            / "cache"
            / query_mode(self.mode)
            / split
            / "query_4b"
            / f"shard_{shard:05d}.pt"
        )
        record = self._load(("query", split, shard), path)[
            index % self.cfg["query_shard_size"]
        ]
        if record["id"] != sample_id:
            raise RuntimeError(f"query shard mismatch: {sample_id}")
        return record

    def memory(self, split, sender, sample, kind="correct"):
        source_id = sample["shuffle_id"] if kind == "shuffled" else sample["id"]
        record = self.native(split, sender, source_id)
        target = sample["selected_token_count"]
        valid = min(target, record["native_k"].shape[1])
        key = record["native_k"][:, :target].clone()
        value = record["native_v"][:, :target].clone()
        if valid < target:
            key = F.pad(key, (0, 0, 0, 0, 0, target - valid))
            value = F.pad(value, (0, 0, 0, 0, 0, target - valid))
        mask = torch.zeros((1, target), dtype=torch.long)
        mask[:, :valid] = 1
        if kind == "zero":
            key.zero_()
            value.zero_()
        return key, value, mask


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(cfg, mode):
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    r1 = r1_module(cfg)
    tok4 = r1.tokenizer(cfg["model_4b"])
    tok8 = r1.tokenizer(cfg["model_8b"])
    if tok4.get_vocab() != tok8.get_vocab():
        raise RuntimeError("4B/8B tokenizer vocabularies differ")
    if tok4.special_tokens_map != tok8.special_tokens_map:
        raise RuntimeError("4B/8B special token maps differ")
    checked = 0
    truncated = 0
    for split, samples in rows.items():
        for sample in samples:
            four = store.native(split, "4b", sample["id"])
            eight = store.native(split, "8b", sample["id"])
            query = store.query4(split, sample["id"])
            if four["native_k"].shape != eight["native_k"].shape:
                raise RuntimeError(f"4B/8B shape mismatch: {sample['id']}")
            shape = four["native_k"].shape
            if (shape[0], shape[2], shape[3]) != (36, 8, 128):
                raise RuntimeError(f"KV layout mismatch: {sample['id']}")
            if query["query"].shape != torch.Size([36, 2, 32, 128]):
                raise RuntimeError(f"Receiver query mismatch: {sample['id']}")
            if sample["selected_token_count"] == cfg["max_evidence_tokens"]:
                truncated += int(four["native_k"].shape[1] > cfg["max_evidence_tokens"])
            text = tok4.decode(sample["full_sequence_token_ids"])
            if tok4.encode(text, add_special_tokens=False) != tok8.encode(
                text, add_special_tokens=False
            ):
                raise RuntimeError(f"4B/8B tokenization mismatch: {sample['id']}")
            checked += 1
    checkpoint = (
        Path(cfg["r1_dir"])
        / "artifacts"
        / source_mode(mode)
        / "sparse_reader"
        / "best.pt"
    )
    report = {
        "passed": True,
        "samples_checked": checked,
        "truncated_samples": truncated,
        "sender_input": "evidence/context only; question absent",
        "receiver_input": "question only plus external memory",
        "channel": "Qwen3-4B native sparse pre-RoPE K and native V",
        "reader_checkpoint": str(checkpoint),
        "reader_checkpoint_sha256": sha256(checkpoint),
        "reader_lora": "q_proj and o_proj on all 36 layers, frozen",
        "position_protocol": "R1 selected_position_ids",
        "token_mask_preserved": True,
        "depth_interaction": False,
    }
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "protocol_audit.json", report)
    progress(f"{mode}: protocol audit passed ({checked} samples)")


def statistics(cfg, mode):
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    shape = (36, 8, 128)
    sums = {
        name: torch.zeros(shape, dtype=torch.float64)
        for name in ("scale4_k", "scale4_v", "scale8_k", "scale8_v")
    }
    count = 0
    for index, sample in enumerate(rows["train"], 1):
        for sender in ("4b", "8b"):
            record = store.native("train", sender, sample["id"])
            tokens = min(record["native_k"].shape[1], cfg["max_evidence_tokens"])
            sums[f"scale{sender[0]}_k"] += record["native_k"][:, :tokens].double().square().sum(1)
            sums[f"scale{sender[0]}_v"] += record["native_v"][:, :tokens].double().square().sum(1)
        count += min(
            store.native("train", "4b", sample["id"])["native_k"].shape[1],
            cfg["max_evidence_tokens"],
        )
        if index % 64 == 0 or index == len(rows["train"]):
            progress(f"{mode}: train-only fixed scales {index}/{len(rows['train'])}")
    stats = {key: (value / count + 1e-8).sqrt().float() for key, value in sums.items()}
    stats.update({"train_samples": len(rows["train"]), "train_tokens": count})
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "protocol"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out / "fixed_scales.pt")
    save_json(
        out / "fixed_scales.json",
        {
            "train_only": True,
            "padding_excluded": True,
            "shape": [36, 8, 128],
            "train_samples": len(rows["train"]),
            "train_tokens": count,
            "sha256": sha256(out / "fixed_scales.pt"),
        },
    )


def load_stats(cfg, mode):
    return torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "protocol" / "fixed_scales.pt",
        map_location="cpu",
        weights_only=False,
    )


def apply_rope(x, positions, theta):
    half = x.shape[-1] // 2
    inverse = 1.0 / (
        float(theta)
        ** (torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1])
    )
    freq = torch.outer(torch.tensor(positions, device=x.device).float(), inverse)
    emb = torch.cat((freq, freq), -1)[None, :, None]
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x.float() * emb.cos() + rotated.float() * emb.sin()


def representation_losses(cfg, query, prediction, target, positions):
    pred_k, pred_v = prediction
    gold_k, gold_v = target

    def nmse(pred, gold):
        error = (pred.float() - gold.float()).square().mean(1)
        energy = gold.float().square().mean(1).clamp_min(1e-8)
        return (error / energy).mean()

    def cosine_loss(pred, gold):
        p = pred.float().permute(0, 2, 1, 3).reshape(36, 8, -1)
        g = gold.float().permute(0, 2, 1, 3).reshape(36, 8, -1)
        return (1 - F.cosine_similarity(p, g, -1)).mean()

    groups = cfg["num_query_heads"] // cfg["num_kv_heads"]
    q = apply_rope(query["query"].to(pred_k.device), query["query_position_ids"], cfg["rope_theta"])
    pk = apply_rope(pred_k, positions, cfg["rope_theta"]).repeat_interleave(groups, 2)
    gk = apply_rope(gold_k, positions, cfg["rope_theta"]).repeat_interleave(groups, 2)
    pv = pred_v.float().repeat_interleave(groups, 2)
    gv = gold_v.float().repeat_interleave(groups, 2)
    logits_p = torch.einsum("lqhd,lthd->lhqt", q, pk) / math.sqrt(128)
    logits_g = torch.einsum("lqhd,lthd->lhqt", q, gk) / math.sqrt(128)
    attn_p, attn_g = logits_p.softmax(-1), logits_g.softmax(-1)
    route = (
        attn_g * (attn_g.clamp_min(1e-12).log() - attn_p.clamp_min(1e-12).log())
    ).sum(-1).mean()
    out_p = torch.einsum("lhqt,lthd->lqhd", attn_p, pv)
    out_g = torch.einsum("lhqt,lthd->lqhd", attn_g, gv)
    out_nmse = (out_p - out_g).square().mean() / out_g.square().mean().clamp_min(1e-8)
    out_cosine = F.cosine_similarity(out_p.flatten(), out_g.flatten(), 0)
    values = {
        "kv": nmse(pred_k, gold_k) + nmse(pred_v, gold_v),
        "cos": cosine_loss(pred_k, gold_k) + cosine_loss(pred_v, gold_v),
        "route": route,
        "out": out_nmse + 0.5 * (1 - out_cosine),
        "k_cosine": 1 - cosine_loss(pred_k, gold_k),
        "v_cosine": 1 - cosine_loss(pred_v, gold_v),
        "output_cosine": out_cosine,
        "output_nmse": out_nmse,
    }
    values["stage_a"] = 0.5 * values["kv"] + 0.25 * values["cos"] + 0.5 * route + values["out"]
    values["stage_b_rep"] = 0.25 * values["kv"] + 0.25 * route + 0.5 * values["out"]
    return values


def sample_values(cfg, writer, store, split, sample, kind="correct"):
    k8, v8, mask = store.memory(split, "8b", sample, kind)
    k4, v4, _ = store.memory(split, "4b", sample, "correct")
    k8, v8 = k8.to(device()), v8.to(device())
    k4, v4 = k4.to(device()), v4.to(device())
    predicted = writer(k8, v8)
    values = representation_losses(
        cfg,
        store.query4(split, sample["id"]),
        predicted,
        (k4, v4),
        sample["selected_position_ids"],
    )
    return predicted, mask, values


@torch.no_grad()
def validate_representation(cfg, writer, store, samples):
    writer.eval()
    rows = []
    for sample in samples:
        _, _, values = sample_values(cfg, writer, store, "validation", sample)
        rows.append({key: value.item() for key, value in values.items()})
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def schedule(optimizer, cfg, maximum):
    warmup = min(cfg["warmup_updates"], maximum)
    return LambdaLR(
        optimizer,
        lambda step: min((step + 1) / max(warmup, 1), 1.0),
    )


def stage_a(cfg, mode, variant):
    seed_all(cfg["seed"])
    rows, store = rows_for(cfg, mode), None
    store = Stores(cfg, mode, rows)
    writer = Native4ChannelWriter(cfg, load_stats(cfg, mode), variant).to(device())
    if writer.zero_check() != 0.0:
        raise RuntimeError("Writer(0) != 0")
    optimizer = AdamW(writer.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["stage_a_updates"]
    scheduler = schedule(optimizer, cfg, maximum)
    interval = 1 if mode == "smoke" else cfg["eval_interval"]
    out = Path(cfg["work_dir"]) / "artifacts" / mode / variant / "stage_a"
    out.mkdir(parents=True, exist_ok=True)
    best, history, evaluations = -float("inf"), [], []
    samples, cursor, epoch = list(rows["train"]), 0, 0
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        micro = []
        for _ in range(cfg["gradient_accumulation"]):
            if cursor == 0:
                random.Random(cfg["seed"] + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            writer.train()
            _, _, values = sample_values(cfg, writer, store, "train", sample)
            (values["stage_a"] / cfg["gradient_accumulation"]).backward()
            micro.append({key: value.detach().item() for key, value in values.items()})
        grad_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "update": update,
            "grad_norm": grad_norm,
            "lr": scheduler.get_last_lr()[0],
            **{key: sum(x[key] for x in micro) / len(micro) for key in micro[0]},
        }
        history.append(row)
        if update % interval == 0 or update == maximum:
            validation = validate_representation(cfg, writer, store, rows["validation"])
            selected = validation["output_cosine"] > best
            if selected:
                best = validation["output_cosine"]
                torch.save(
                    {"writer": writer.state_dict(), "update": update, "validation": validation},
                    out / "best.pt",
                )
            evaluations.append({"update": update, **validation, "selected": selected})
            save_json(out / "history.json", history)
            save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}/{variant}: stage A {update}/{maximum}")
    save_json(out / "summary.json", {"completed": True, "best_output_cosine": best, "updates": maximum})


def load_reader(cfg, mode):
    r1 = r1_module(cfg)
    model = r1.inject_lora(r1.load_model(cfg["model_4b"]), cfg)
    checkpoint = Path(cfg["r1_dir"]) / "artifacts" / source_mode(mode) / "sparse_reader" / "best.pt"
    r1.load_lora(model, checkpoint)
    r1.set_lora(model, True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return r1, model, r1.tokenizer(cfg["model_4b"])


@torch.no_grad()
def validate_answer(cfg, writer, store, reader, tok, r1, samples):
    writer.eval()
    values = []
    for sample in samples[: min(16, len(samples))]:
        predicted, mask, _ = sample_values(cfg, writer, store, "validation", sample)
        values.append(r1.answer_loss(cfg, reader, tok, sample, predicted[0].half(), predicted[1].half(), mask).item())
    return sum(values) / len(values)


def stage_b(cfg, mode, variant):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    writer = Native4ChannelWriter(cfg, load_stats(cfg, mode), variant).to(device())
    stage_a_path = Path(cfg["work_dir"]) / "artifacts" / mode / variant / "stage_a" / "best.pt"
    writer.load_state_dict(torch.load(stage_a_path, map_location="cpu", weights_only=False)["writer"])
    r1, reader, tok = load_reader(cfg, mode)
    optimizer = AdamW(writer.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["stage_b_updates"]
    scheduler = schedule(optimizer, cfg, maximum)
    interval = 1 if mode == "smoke" else cfg["eval_interval"]
    out = Path(cfg["work_dir"]) / "artifacts" / mode / variant / "stage_b"
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history, evaluations = [], []
    samples, cursor, epoch = list(rows["train"]), 0, 0
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        micro = []
        for _ in range(cfg["gradient_accumulation"]):
            if cursor == 0:
                random.Random(cfg["seed"] + 10000 + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            writer.train()
            correct, mask, rep = sample_values(cfg, writer, store, "train", sample)
            wrong_k, wrong_v, wrong_mask = store.memory("train", "8b", sample, "shuffled")
            shuffled = writer(wrong_k.to(device()), wrong_v.to(device()))
            correct_nll = r1.answer_loss(
                cfg, reader, tok, sample, correct[0].half(), correct[1].half(), mask
            )
            shuffled_nll = r1.answer_loss(
                cfg, reader, tok, sample, shuffled[0].half(), shuffled[1].half(), wrong_mask
            )
            dependence = F.relu(cfg["dependence_margin"] + correct_nll - shuffled_nll)
            loss = rep["stage_b_rep"] + correct_nll + 0.2 * dependence
            (loss / cfg["gradient_accumulation"]).backward()
            micro.append(
                {
                    "loss": loss.detach().item(),
                    "answer_nll": correct_nll.detach().item(),
                    "shuffled_nll": shuffled_nll.detach().item(),
                    "dependence": dependence.detach().item(),
                    "output_cosine": rep["output_cosine"].detach().item(),
                }
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(
            {
                "update": update,
                "grad_norm": grad_norm,
                "lr": scheduler.get_last_lr()[0],
                **{key: sum(x[key] for x in micro) / len(micro) for key in micro[0]},
            }
        )
        if update % interval == 0 or update == maximum:
            validation_nll = validate_answer(cfg, writer, store, reader, tok, r1, rows["validation"])
            representation = validate_representation(cfg, writer, store, rows["validation"])
            selected = validation_nll < best
            if selected:
                best = validation_nll
                torch.save(
                    {
                        "writer": writer.state_dict(),
                        "update": update,
                        "validation_nll": validation_nll,
                        "validation_representation": representation,
                    },
                    out / "best.pt",
                )
            evaluations.append(
                {"update": update, "validation_nll": validation_nll, **representation, "selected": selected}
            )
            save_json(out / "history.json", history)
            save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}/{variant}: stage B {update}/{maximum}")
    save_json(out / "summary.json", {"completed": True, "best_validation_nll": best, "updates": maximum})
    del reader
    torch.cuda.empty_cache()


def aggregate(rows):
    report = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        record = {}
        for key in ("em", "token_f1", "nll"):
            record[key] = sum(float(row[key]) for row in selected) / len(selected)
        for kind in ("bridge", "comparison"):
            subset = [row for row in selected if row["type"] == kind]
            record[f"{kind}_f1"] = (
                sum(float(row["token_f1"]) for row in subset) / len(subset) if subset else None
            )
        record["count"] = len(selected)
        report[condition] = record
    return report


def evaluate(cfg, mode):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    r1, reader, tok = load_reader(cfg, mode)
    writers = {}
    for variant in cfg["variants"]:
        model = Native4ChannelWriter(cfg, load_stats(cfg, mode), variant).to(device()).eval()
        checkpoint = torch.load(
            Path(cfg["work_dir"]) / "artifacts" / mode / variant / "stage_b" / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["writer"])
        writers[variant] = model
    conditions = [
        ("question_only", "none", "correct", True),
        ("supporting_text", "text", "correct", False),
        ("native4_correct", "4b", "correct", True),
        ("native4_shuffled", "4b", "shuffled", True),
        ("native4_zero", "4b", "zero", True),
        ("raw8_correct", "8b", "correct", True),
        ("raw8_shuffled", "8b", "shuffled", True),
        ("reader_off", "off", "correct", False),
    ]
    for variant in cfg["variants"]:
        for kind in ("correct", "shuffled", "zero"):
            conditions.append((f"writer_{variant}_{kind}", variant, kind, True))
    output_rows = []
    for condition, family, kind, lora_enabled in conditions:
        r1.set_lora(reader, lora_enabled)
        progress(f"{mode}: evaluate {condition}")
        for sample in rows["test"]:
            key = value = mask = None
            supporting = family == "text"
            compact = family in {"none", "off"}
            if family in {"4b", "8b"}:
                key, value, mask = store.memory("test", family, sample, kind)
                key, value = key.to(device()), value.to(device())
            elif family in writers:
                k8, v8, mask = store.memory("test", "8b", sample, kind)
                with torch.no_grad():
                    key, value = writers[family](k8.to(device()), v8.to(device()))
            with torch.no_grad():
                prediction = r1.greedy_generate(
                    cfg,
                    reader,
                    tok,
                    sample,
                    None if key is None else key.half(),
                    None if value is None else value.half(),
                    mask,
                    supporting_text=supporting,
                    compact_positions=compact,
                )
                if supporting:
                    target = r1.answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
                    prompt = sample["full_sequence_token_ids"]
                    sequence = torch.tensor([prompt + target[:-1]], device=device())
                    logits = reader(
                        sequence,
                        attention_mask=torch.ones_like(sequence),
                        position_ids=torch.arange(sequence.shape[1], device=device()).unsqueeze(0),
                        use_cache=False,
                    ).logits
                    selected = logits[:, len(prompt) - 1 : len(prompt) - 1 + len(target)].float()
                    nll = F.cross_entropy(
                        selected.reshape(-1, selected.shape[-1]),
                        torch.tensor(target, device=device()),
                    ).item()
                else:
                    nll = r1.answer_loss(
                        cfg,
                        reader,
                        tok,
                        sample,
                        None if key is None else key.half(),
                        None if value is None else value.half(),
                        mask,
                        compact_question_positions=compact,
                    ).item()
            output_rows.append(
                {
                    "sample_id": sample["id"],
                    "type": sample.get("type", "unknown"),
                    "condition": condition,
                    "answer": sample["answer"],
                    "prediction": prediction,
                    "em": float(r1.normalize_answer(prediction) == r1.normalize_answer(sample["answer"])),
                    "token_f1": r1.token_f1(prediction, sample["answer"]),
                    "nll": nll,
                    "manual_c_p_w": "",
                }
            )
    summary = aggregate(output_rows)
    aq = summary["question_only"]["em"]
    an = summary["native4_correct"]["em"]
    for variant in cfg["variants"]:
        aw = summary[f"writer_{variant}_correct"]["em"]
        shuffled = summary[f"writer_{variant}_shuffled"]["em"]
        summary[f"writer_{variant}_diagnostics"] = {
            "retention": (aw - aq) / (an - aq) if an != aq else None,
            "correct_shuffled_em_gap": aw - shuffled,
            "correct_shuffled_nll_gap": summary[f"writer_{variant}_shuffled"]["nll"]
            - summary[f"writer_{variant}_correct"]["nll"],
            "criteria_diagnostic_only": True,
        }
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "evaluation"
    save_json(out / "summary.json", summary)
    save_json(out / "per_sample.json", output_rows)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manual_c_p_w.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    save_json(
        out / "completion.json",
        {
            "completed": True,
            "hard_gates_enforced": cfg["hard_gates_enforced"],
            "question_never_enters_writer": True,
            "reader_frozen": True,
            "receiver_frozen": True,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument(
        "action",
        choices=("audit", "statistics", "stage_a", "stage_b", "evaluate"),
    )
    parser.add_argument("--variant", choices=("linear", "mlp", "full"))
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    if args.action in {"stage_a", "stage_b"} and not args.variant:
        parser.error("--variant is required")
    if args.action == "audit":
        audit(cfg, args.mode)
    elif args.action == "statistics":
        statistics(cfg, args.mode)
    elif args.action == "stage_a":
        stage_a(cfg, args.mode, args.variant)
    elif args.action == "stage_b":
        stage_b(cfg, args.mode, args.variant)
    else:
        evaluate(cfg, args.mode)


if __name__ == "__main__":
    main()
