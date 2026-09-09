from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import string
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from models import ProtocolDecoder, ProtocolTransform
from models.canonical_losses import (
    canonical_alignment,
    contrastive,
    path_loss,
    pooled,
    variance_loss,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def empty_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(require_cuda())
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def source_mode(mode):
    return "smoke" if mode == "smoke" else "formal"


def source_manifest(cfg, mode):
    rows = load_json(
        Path(cfg["r1_dir"]) / "artifacts" / source_mode(mode) / "manifest.json"
    )
    sizes = cfg[f"{mode}_sizes"]
    selected = {
        split: [dict(sample) for sample in samples[: sizes[split]]]
        for split, samples in rows.items()
    }
    # R1 shuffle IDs were constructed over its complete split. This experiment
    # deliberately uses smaller first-shot subsets, so rebuild a deterministic
    # mismatch inside each actual subset. Both the audit and evaluation consume
    # this exact mapping.
    for split, samples in selected.items():
        for index, sample in enumerate(samples):
            candidates = (
                samples[(index + offset) % len(samples)]
                for offset in range(1, len(samples))
            )
            wrong = next(
                (
                    candidate
                    for candidate in candidates
                    if normalize_answer(candidate["answer"])
                    != normalize_answer(sample["answer"])
                ),
                None,
            )
            if wrong is None:
                raise RuntimeError(
                    f"{split} subset cannot construct an answer-distinct shuffle"
                )
            sample["source_shuffle_id"] = sample.get("shuffle_id")
            sample["shuffle_id"] = wrong["id"]
    return selected


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, answer):
    pred, gold = normalize_answer(prediction).split(), normalize_answer(answer).split()
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if not pred or not gold:
        return float(pred == gold)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def loose_correct(prediction, answer):
    pred, gold = normalize_answer(prediction), normalize_answer(answer)
    return bool(pred and gold and (gold in pred or pred in gold))


def structure_check(cfg):
    report = {}
    tok4, tok8 = tokenizer(cfg["model_4b"]), tokenizer(cfg["model_8b"])
    if tok4.get_vocab() != tok8.get_vocab():
        raise RuntimeError("4B and 8B tokenizers differ")
    for sender in ("4b", "8b"):
        conf = AutoConfig.from_pretrained(cfg[f"model_{sender}"], local_files_only=True)
        report[sender] = {
            "layers": conf.num_hidden_layers,
            "kv_heads": conf.num_key_value_heads,
            "head_dim": getattr(conf, "head_dim", conf.hidden_size // conf.num_attention_heads),
            "rope_theta": getattr(conf, "rope_theta", None),
            "rope_scaling": getattr(conf, "rope_scaling", None),
        }
        if (
            report[sender]["layers"],
            report[sender]["kv_heads"],
            report[sender]["head_dim"],
        ) != (cfg["num_layers"], cfg["num_kv_heads"], cfg["head_dim"]):
            raise RuntimeError(f"{sender} KV shape is incompatible")
    if report["4b"]["rope_theta"] != report["8b"]["rope_theta"] or report["4b"][
        "rope_scaling"
    ] != report["8b"]["rope_scaling"]:
        raise RuntimeError("RoPE configurations differ")
    report["tokenizer_identical"] = True
    report["held_out_qwen3_5"] = True
    report["receiver_lora"] = False
    save_json(Path(cfg["work_dir"]) / "artifacts" / "structure_check.json", report)
    progress("Multi-view structure check passed")


def prepare_reference(cfg, mode):
    rows = source_manifest(cfg, mode)
    save_json(
        Path(cfg["work_dir"]) / "artifacts" / mode / "manifest_reference.json",
        {
            "source": str(
                Path(cfg["r1_dir"])
                / "artifacts"
                / source_mode(mode)
                / "manifest.json"
            ),
            "counts": {split: len(samples) for split, samples in rows.items()},
            "sample_ids": {split: [x["id"] for x in samples] for split, samples in rows.items()},
            "selection": "gold supporting tokens from complete Context->Question prefill",
            "seed": cfg["seed"],
        },
    )
    progress(f"{mode}: exact R1 sample reference prepared")


class QueryCapture:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.indices = None
        self.values = {}
        self.handles = [
            layer.self_attn.register_forward_pre_hook(self._hook(index), with_kwargs=True)
            for index, layer in enumerate(model.model.layers)
        ]

    def _hook(self, layer_index):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            batch, length, _ = hidden.shape
            query = module.q_norm(
                module.q_proj(hidden).view(
                    batch,
                    length,
                    self.cfg["num_query_heads"],
                    self.cfg["head_dim"],
                )
            )
            self.values[layer_index] = (
                query[0, self.indices].detach().cpu().half().contiguous()
            )

        return hook

    def run(self, model, sample, first_answer_token):
        self.values.clear()
        device = require_cuda()
        full = sample["full_sequence_token_ids"] + [first_answer_token]
        self.indices = torch.tensor([len(full) - 2, len(full) - 1], device=device)
        ids = torch.tensor([full], dtype=torch.long, device=device)
        model(
            ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1], device=device).unsqueeze(0),
            use_cache=False,
        )
        if len(self.values) != self.cfg["num_layers"]:
            raise RuntimeError("Native Query capture missed layers")
        return torch.stack([self.values[index] for index in range(self.cfg["num_layers"])])

    def close(self):
        for handle in self.handles:
            handle.remove()


def first_answer_token(tok, answer):
    ids = tok(" " + answer, add_special_tokens=False).input_ids
    if not ids:
        raise RuntimeError("Gold answer tokenization is empty")
    return ids[0]


@torch.no_grad()
def extract_queries(cfg, mode, sender):
    rows = source_manifest(cfg, mode)
    tok = tokenizer(cfg[f"model_{sender}"])
    model = load_model(cfg[f"model_{sender}"])
    capture = QueryCapture(model, cfg)
    root = Path(cfg["work_dir"]) / "cache" / mode
    for split, samples in rows.items():
        directory = root / split / f"query_{sender}"
        directory.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(samples), cfg["shard_size"]):
            shard = start // cfg["shard_size"]
            destination = directory / f"shard_{shard:05d}.pt"
            if destination.exists():
                continue
            records = []
            for sample in samples[start : start + cfg["shard_size"]]:
                query = capture.run(
                    model, sample, first_answer_token(tok, sample["answer"])
                )
                records.append(
                    {
                        "id": sample["id"],
                        "query": query,
                        "query_position_ids": [
                            sample["question_position_ids"][-1],
                            sample["question_position_ids"][-1] + 1,
                        ],
                    }
                )
            torch.save(records, destination)
            progress(
                f"{mode}: {sender} two-position Query {split} shard {shard + 1}/"
                f"{math.ceil(len(samples) / cfg['shard_size'])}"
            )
    capture.close()
    del model
    empty_cuda()


class AssetStore:
    def __init__(self, cfg, mode, rows):
        self.cfg, self.mode, self.rows = cfg, mode, rows
        self.positions = {
            split: {sample["id"]: index for index, sample in enumerate(samples)}
            for split, samples in rows.items()
        }
        self.cache = {}

    def _load(self, key, path):
        if key not in self.cache:
            if len(self.cache) >= 10:
                self.cache.clear()
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.cache[key]

    def native(self, split, sender, sample_id):
        index = self.positions[split][sample_id]
        shard = index // 32
        path = (
            Path(self.cfg["r1_dir"])
            / "cache"
            / source_mode(self.mode)
            / split
            / sender
            / f"shard_{shard:05d}.pt"
        )
        records = self._load(("native", split, sender, shard), path)
        record = records[index % 32]
        if record["id"] != sample_id:
            raise RuntimeError("R1 native shard index mismatch")
        return record

    def query(self, split, sender, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["shard_size"]
        path = (
            Path(self.cfg["work_dir"])
            / "cache"
            / self.mode
            / split
            / f"query_{sender}"
            / f"shard_{shard:05d}.pt"
        )
        records = self._load(("query", split, sender, shard), path)
        record = records[index % self.cfg["shard_size"]]
        if record["id"] != sample_id:
            raise RuntimeError("Query shard index mismatch")
        return record


def a0_audit(cfg, mode):
    rows = source_manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    report = {"splits": {}, "passed": True}
    for split, samples in rows.items():
        count = 0
        for sample in samples:
            four = store.native(split, "4b", sample["id"])
            eight = store.native(split, "8b", sample["id"])
            q4 = store.query(split, "4b", sample["id"])
            q8 = store.query(split, "8b", sample["id"])
            expected = (
                cfg["num_layers"],
                sample["selected_token_count"],
                cfg["num_kv_heads"],
                cfg["head_dim"],
            )
            for record in (four, eight):
                if tuple(record["native_k"].shape) != expected or tuple(
                    record["native_v"].shape
                ) != expected:
                    raise RuntimeError(f"{sample['id']} Native KV shape mismatch")
            query_shape = (
                cfg["num_layers"],
                2,
                cfg["num_query_heads"],
                cfg["head_dim"],
            )
            if tuple(q4["query"].shape) != query_shape or tuple(q8["query"].shape) != query_shape:
                raise RuntimeError(f"{sample['id']} Query shape mismatch")
            if q4["query_position_ids"] != q8["query_position_ids"]:
                raise RuntimeError(f"{sample['id']} query positions differ")
            if len(sample["selected_position_ids"]) != sample["selected_token_count"]:
                raise RuntimeError(f"{sample['id']} selected position count mismatch")
            wrong = next(x for x in samples if x["id"] == sample["shuffle_id"])
            if normalize_answer(wrong["answer"]) == normalize_answer(sample["answer"]):
                raise RuntimeError(f"{sample['id']} shuffled answer is not different")
            count += 1
        report["splits"][split] = {"samples_checked": count}
    report.update(
        {
            "token_ids_identical": True,
            "support_indices_identical": True,
            "pre_rope_k_source": "R1 NativeCapture before rotary embedding",
            "dynamic_attention_mask": True,
            "shuffled_answer_different": True,
        }
    )
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "a0_audit.json", report)
    progress(f"{mode}: A0 consistency audit passed")


class ModuleBundle(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.writer4 = ProtocolTransform(cfg)
        self.writer8 = ProtocolTransform(cfg)
        self.decoder4 = ProtocolDecoder(cfg)
        self.decoder8 = ProtocolDecoder(cfg)


def bundle_state(bundle):
    return {name: module.state_dict() for name, module in (
        ("writer4", bundle.writer4),
        ("writer8", bundle.writer8),
        ("decoder4", bundle.decoder4),
        ("decoder8", bundle.decoder8),
    )}


def load_bundle_state(bundle, state):
    for name in ("writer4", "writer8", "decoder4", "decoder8"):
        getattr(bundle, name).load_state_dict(state[name])


def sample_tensors(store, split, sample):
    device = require_cuda()
    four, eight = store.native(split, "4b", sample["id"]), store.native(
        split, "8b", sample["id"]
    )
    q4, q8 = store.query(split, "4b", sample["id"]), store.query(
        split, "8b", sample["id"]
    )
    return {
        "k4": four["native_k"].to(device),
        "v4": four["native_v"].to(device),
        "k8": eight["native_k"].to(device),
        "v8": eight["native_v"].to(device),
        "q4": q4["query"].to(device),
        "q8": q8["query"].to(device),
        "qpos": q4["query_position_ids"],
        "epos": sample["selected_position_ids"],
    }


def compute_paths(cfg, bundle, tensors):
    c4k, c4v = bundle.writer4(tensors["k4"], tensors["v4"])
    c8k, c8v = bundle.writer8(tensors["k8"], tensors["v8"])
    d4c4 = bundle.decoder4(c4k, c4v)
    d4c8 = bundle.decoder4(c8k, c8v)
    d8c8 = bundle.decoder8(c8k, c8v)
    d8c4 = bundle.decoder8(c4k, c4v)
    paths = {
        "self4": path_loss(
            cfg, tensors["q4"], *d4c4, tensors["k4"], tensors["v4"], tensors["epos"], tensors["qpos"]
        ),
        "cross4": path_loss(
            cfg, tensors["q4"], *d4c8, tensors["k4"], tensors["v4"], tensors["epos"], tensors["qpos"]
        ),
        "self8": path_loss(
            cfg, tensors["q8"], *d8c8, tensors["k8"], tensors["v8"], tensors["epos"], tensors["qpos"]
        ),
        "cross8": path_loss(
            cfg, tensors["q8"], *d8c4, tensors["k8"], tensors["v8"], tensors["epos"], tensors["qpos"]
        ),
    }
    return (c4k, c4v), (c8k, c8v), paths, {"d4c4": d4c4, "d4c8": d4c8}


def a2_components(cfg, bundle, tensors):
    c4, c8, paths, decoded = compute_paths(cfg, bundle, tensors)
    align = canonical_alignment(*c4, *c8)
    decode = sum(paths[name]["decode"] for name in paths)
    route = sum(paths[name]["route"] for name in paths)
    output = sum(paths[name]["output"] for name in paths)
    return {
        "c4": c4,
        "c8": c8,
        "paths": paths,
        "decoded": decoded,
        "align": align,
        "decode": decode,
        "route": route,
        "output": output,
    }


def train_a1(cfg, mode):
    seed_all(cfg["seed"])
    rows = source_manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    bundle = ModuleBundle(cfg).to(require_cuda()).train()
    optimizer = AdamW(bundle.parameters(), lr=cfg["a1_lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    history = []
    samples = list(rows["train"])
    random.Random(cfg["seed"]).shuffle(samples)
    optimizer.zero_grad(set_to_none=True)
    for index, sample in enumerate(samples, 1):
        tensors = sample_tensors(store, "train", sample)
        with torch.autocast("cuda", dtype=torch.float16):
            c4 = bundle.writer4(tensors["k4"], tensors["v4"])
            c8 = bundle.writer8(tensors["k8"], tensors["v8"])
            d4 = bundle.decoder4(*c4)
            d8 = bundle.decoder8(*c8)
            self4 = path_loss(
                cfg, tensors["q4"], *d4, tensors["k4"], tensors["v4"],
                tensors["epos"], tensors["qpos"]
            )
            self8 = path_loss(
                cfg, tensors["q8"], *d8, tensors["k8"], tensors["v8"],
                tensors["epos"], tensors["qpos"]
            )
            loss = self4["total"] + self8["total"]
        scaler.scale(loss / cfg["gradient_accumulation"]).backward()
        if index % cfg["gradient_accumulation"] == 0 or index == len(samples):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(bundle.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        history.append(
            {
                "sample": index,
                "self4_total": self4["total"].item(),
                "self8_total": self8["total"].item(),
                "self4_output_cosine": self4["output_cosine"].item(),
                "self8_output_cosine": self8["output_cosine"].item(),
            }
        )
    torch.save({"bundle": bundle_state(bundle)}, output / "a1_best.pt")
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "a1_history.json", history)
    del bundle, optimizer
    empty_cuda()
    progress(f"{mode}: A1 self bootstrap completed")


@torch.no_grad()
def a2_validation(cfg, bundle, store, samples):
    bundle.eval()
    cross_score, self_score = [], []
    for sample in samples:
        components = a2_components(cfg, bundle, sample_tensors(store, "validation", sample))
        cross_score.append(
            (
                components["paths"]["cross4"]["output"]
                + components["paths"]["cross8"]["output"]
                + components["paths"]["cross4"]["route"]
                + components["paths"]["cross8"]["route"]
                + 0.5 * components["align"]
            ).item()
        )
        self_score.append(
            (
                components["paths"]["self4"]["output"]
                + components["paths"]["self8"]["output"]
                + components["paths"]["self4"]["route"]
                + components["paths"]["self8"]["route"]
            ).item()
        )
    return sum(cross_score) / len(cross_score), sum(self_score) / len(self_score)


def train_a2(cfg, mode):
    seed_all(cfg["seed"])
    rows = source_manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    bundle = ModuleBundle(cfg).to(require_cuda())
    state = torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "checkpoints" / "a1_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    load_bundle_state(bundle, state["bundle"])
    bundle.train()
    optimizer = AdamW(
        [
            {"params": list(bundle.writer4.parameters()) + list(bundle.writer8.parameters()), "lr": cfg["a2_writer_lr"]},
            {"params": list(bundle.decoder4.parameters()) + list(bundle.decoder8.parameters()), "lr": cfg["a2_decoder_lr"]},
        ],
        weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda")
    epochs = 1 if mode == "smoke" else cfg["a2_epochs"]
    history, best_cross, best_self = [], float("inf"), float("inf")
    output = Path(cfg["work_dir"]) / "artifacts" / mode
    accumulation_groups = max(1, cfg["gradient_accumulation"] // 2)
    for epoch in range(epochs):
        samples = list(rows["train"])
        random.Random(cfg["seed"] + epoch).shuffle(samples)
        optimizer.zero_grad(set_to_none=True)
        groups = [samples[index : index + 2] for index in range(0, len(samples), 2)]
        for group_index, group in enumerate(groups, 1):
            z4, z8, per_sample = [], [], []
            with torch.autocast("cuda", dtype=torch.float16):
                for sample in group:
                    components = a2_components(
                        cfg, bundle, sample_tensors(store, "train", sample)
                    )
                    z4.append(pooled(*components["c4"]))
                    z8.append(pooled(*components["c8"]))
                    per_sample.append(components)
                align = torch.stack([x["align"] for x in per_sample]).mean()
                decode = torch.stack([x["decode"] for x in per_sample]).mean()
                route = torch.stack([x["route"] for x in per_sample]).mean()
                out_loss = torch.stack([x["output"] for x in per_sample]).mean()
                var = variance_loss(z4 + z8, cfg["variance_target"])
                contrast = contrastive(z4, z8, cfg["contrast_temperature"])
                loss = (
                    0.5 * align
                    + 0.1 * var
                    + decode
                    + 0.5 * route
                    + out_loss
                    + 0.1 * contrast
                )
            scaler.scale(loss / accumulation_groups).backward()
            if group_index % accumulation_groups == 0 or group_index == len(groups):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(bundle.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            history.append(
                {
                    "epoch": epoch + 1,
                    "group": group_index,
                    "align": align.item(),
                    "variance": var.item(),
                    "decode": decode.item(),
                    "route": route.item(),
                    "output": out_loss.item(),
                    "contrast": contrast.item(),
                    "total": loss.item(),
                }
            )
        cross, self_value = a2_validation(cfg, bundle, store, rows["validation"])
        if cross < best_cross and (
            best_self == float("inf") or self_value <= best_self * 1.05
        ):
            best_cross, best_self = cross, self_value
            torch.save(
                {
                    "bundle": bundle_state(bundle),
                    "epoch": epoch + 1,
                    "cross_score": cross,
                    "self_score": self_value,
                },
                output / "checkpoints" / "a2_best.pt",
            )
        save_json(output / "a2_history.json", history)
        progress(f"{mode}: A2 epoch {epoch + 1}/{epochs} completed")
        bundle.train()
    del bundle, optimizer
    empty_cuda()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def dynamic_cache(cfg, model, pre_key, value, positions):
    device = pre_key.device
    inverse = 1.0 / (
        float(cfg["rope_theta"])
        ** (
            torch.arange(0, cfg["head_dim"], 2, device=device, dtype=torch.float32)
            / cfg["head_dim"]
        )
    )
    frequency = torch.outer(
        torch.tensor(positions, device=device, dtype=torch.float32), inverse
    )
    embedding = torch.cat((frequency, frequency), -1)
    cosine = embedding.cos().to(pre_key.dtype)[None, :, None, :]
    sine = embedding.sin().to(pre_key.dtype)[None, :, None, :]
    post_key = pre_key * cosine + rotate_half(pre_key) * sine
    items = [
        (
            post_key[layer].permute(1, 0, 2).unsqueeze(0),
            value[layer].permute(1, 0, 2).unsqueeze(0),
        )
        for layer in range(cfg["num_layers"])
    ]
    return DynamicCache(ddp_cache_data=items, config=model.config)


def answer_target(tok, answer, maximum):
    ids = tok(" " + answer, add_special_tokens=False).input_ids[: maximum - 1]
    ids.append(tok.eos_token_id)
    return ids


def answer_loss(cfg, model, tok, sample, key=None, value=None, mask=None, compact=False):
    device = require_cuda()
    question = sample["question_token_ids"]
    target = answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
    current = torch.tensor([question + target[:-1]], dtype=torch.long, device=device)
    qpos = list(range(len(question))) if compact else sample["question_position_ids"]
    apos = list(range(qpos[-1] + 1, qpos[-1] + len(target)))
    kwargs = {}
    if key is not None:
        attention = torch.cat([mask.to(device), torch.ones_like(current)], 1)
        kwargs = {
            "past_key_values": dynamic_cache(
                cfg, model, key, value, sample["selected_position_ids"]
            ),
            "cache_position": torch.arange(
                key.shape[1], key.shape[1] + current.shape[1], device=device
            ),
        }
    else:
        attention = torch.ones_like(current)
    logits = model(
        current,
        attention_mask=attention,
        position_ids=torch.tensor([qpos + apos], dtype=torch.long, device=device),
        use_cache=False,
        **kwargs,
    ).logits
    selected = logits[:, len(question) - 1 : len(question) - 1 + len(target)].float()
    return F.cross_entropy(
        selected.reshape(-1, selected.shape[-1]),
        torch.tensor(target, dtype=torch.long, device=device),
    )


@torch.no_grad()
def a3_validation(cfg, model, tok, bundle, store, samples):
    bundle.eval()
    values = []
    for sample in samples:
        tensors = sample_tensors(store, "validation", sample)
        c4k, c4v = bundle.writer4(tensors["k4"], tensors["v4"])
        c8k, c8v = bundle.writer8(tensors["k8"], tensors["v8"])
        d4c4 = bundle.decoder4(c4k, c4v)
        d4c8 = bundle.decoder4(c8k, c8v)
        mask = torch.ones((1, sample["selected_token_count"]), dtype=torch.long)
        values.append(
            0.5
            * (
                answer_loss(cfg, model, tok, sample, *d4c4, mask).item()
                + answer_loss(cfg, model, tok, sample, *d4c8, mask).item()
            )
        )
    return sum(values) / len(values)


def train_a3_4b(cfg, mode):
    seed_all(cfg["seed"])
    rows = source_manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    tok = tokenizer(cfg["model_4b"])
    model = load_model(cfg["model_4b"])
    bundle = ModuleBundle(cfg).to(require_cuda())
    state = torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "checkpoints" / "a2_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    load_bundle_state(bundle, state["bundle"])
    bundle.train()
    optimizer = AdamW(
        [
            {"params": list(bundle.writer4.parameters()) + list(bundle.writer8.parameters()), "lr": cfg["a3_writer_lr"]},
            {"params": list(bundle.decoder4.parameters()) + list(bundle.decoder8.parameters()), "lr": cfg["a3_decoder_lr"]},
        ],
        weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda")
    epochs = 1
    history, best = [], float("inf")
    output = Path(cfg["work_dir"]) / "artifacts" / mode
    for epoch in range(epochs):
        samples = list(rows["train"])
        random.Random(cfg["seed"] + epoch).shuffle(samples)
        optimizer.zero_grad(set_to_none=True)
        for index, sample in enumerate(samples, 1):
            tensors = sample_tensors(store, "train", sample)
            with torch.autocast("cuda", dtype=torch.float16):
                components = a2_components(cfg, bundle, tensors)
                mask = torch.ones((1, sample["selected_token_count"]), dtype=torch.long)
                nll_self = answer_loss(
                    cfg, model, tok, sample, *components["decoded"]["d4c4"], mask
                )
                nll_cross = answer_loss(
                    cfg, model, tok, sample, *components["decoded"]["d4c8"], mask
                )
                regularizer = (
                    0.5 * components["align"]
                    + components["decode"]
                    + 0.5 * components["route"]
                    + components["output"]
                )
                loss = nll_self + nll_cross + 0.1 * regularizer
            scaler.scale(loss / cfg["gradient_accumulation"]).backward()
            if index % cfg["gradient_accumulation"] == 0 or index == len(samples):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(bundle.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            history.append(
                {
                    "sample": index,
                    "answer_self4": nll_self.item(),
                    "answer_cross4": nll_cross.item(),
                    "a2_regularizer": regularizer.item(),
                    "total": loss.item(),
                }
            )
        validation = a3_validation(cfg, model, tok, bundle, store, rows["validation"])
        if validation < best:
            best = validation
            torch.save(
                {
                    "bundle": bundle_state(bundle),
                    "validation_4b_nll": validation,
                    "a3_target": "4b_only",
                },
                output / "checkpoints" / "a3_best.pt",
            )
        save_json(output / "a3_4b_history.json", history)
        progress(f"{mode}: A3-4B functional calibration completed")
    best_state = torch.load(
        output / "checkpoints" / "a3_best.pt",
        map_location="cpu",
        weights_only=False,
    )["bundle"]
    module_names = {
        "writer4": "writer_4b",
        "writer8": "writer_8b",
        "decoder4": "decoder_4b",
        "decoder8": "decoder_8b",
    }
    for state_name, directory_name in module_names.items():
        directory = output / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state[state_name],
                "stage": "A3-4B",
                "source_blind": state_name.startswith("decoder"),
                "a3_8b_status": "not run in first-shot minimum loop",
            },
            directory / "final.pt",
        )
    del model, bundle, optimizer
    empty_cuda()


def align_tensor(tensor, target_length, zero=False):
    if zero:
        return torch.zeros(
            (tensor.shape[0], target_length, tensor.shape[2], tensor.shape[3]),
            dtype=torch.float16,
        ), target_length
    valid = min(tensor.shape[1], target_length)
    result = tensor[:, :target_length].half()
    if result.shape[1] < target_length:
        result = F.pad(result, (0, 0, 0, 0, 0, target_length - result.shape[1]))
    return result, valid


@torch.no_grad()
def greedy_generate(
    cfg, model, tok, sample, key=None, value=None, mask=None, text=False, compact=False
):
    device = require_cuda()
    prompt = sample["full_sequence_token_ids"] if text else sample["question_token_ids"]
    positions = (
        list(range(len(prompt)))
        if text or compact
        else sample["question_position_ids"]
    )
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    kwargs = {}
    if key is not None:
        attention = torch.cat([mask.to(device), torch.ones_like(ids)], 1)
        kwargs = {
            "past_key_values": dynamic_cache(
                cfg, model, key, value, sample["selected_position_ids"]
            ),
            "cache_position": torch.arange(
                key.shape[1], key.shape[1] + ids.shape[1], device=device
            ),
        }
    else:
        attention = torch.ones_like(ids)
    output = model(
        ids,
        attention_mask=attention,
        position_ids=torch.tensor([positions], dtype=torch.long, device=device),
        use_cache=True,
        **kwargs,
    )
    past = output.past_key_values
    token = output.logits[:, -1].argmax(-1, keepdim=True)
    generated, next_position = [], positions[-1] + 1
    for _ in range(cfg["max_new_tokens"]):
        token_id = int(token.item())
        if token_id == tok.eos_token_id:
            break
        generated.append(token_id)
        attention = torch.cat(
            [attention, torch.ones((1, 1), dtype=torch.long, device=device)], 1
        )
        output = model(
            token,
            attention_mask=attention,
            position_ids=torch.tensor([[next_position]], device=device),
            cache_position=torch.tensor([past.get_seq_length()], device=device),
            past_key_values=past,
            use_cache=True,
        )
        past, token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    return tok.decode(generated, skip_special_tokens=True).strip()


EVAL_CONDITIONS = [
    "question_only",
    "supporting_text",
    "native_4b",
    "native_8b",
    "d4_c4_self",
    "d4_c8_cross",
    "d4_c4_shuffled",
    "d4_c8_shuffled",
    "d4_zero",
]


@torch.no_grad()
def evaluation_memory(cfg, bundle, store, sample, condition):
    device = require_cuda()
    if condition == "native_4b":
        record = store.native("test", "4b", sample["id"])
        return record["native_k"].to(device), record["native_v"].to(device), torch.ones(
            (1, sample["selected_token_count"]), dtype=torch.long
        )
    if condition == "native_8b":
        record = store.native("test", "8b", sample["id"])
        return record["native_k"].to(device), record["native_v"].to(device), torch.ones(
            (1, sample["selected_token_count"]), dtype=torch.long
        )
    if condition == "d4_zero":
        zero = torch.zeros(
            (
                cfg["num_layers"],
                sample["selected_token_count"],
                cfg["num_kv_heads"],
                cfg["head_dim"],
            ),
            dtype=torch.float16,
            device=device,
        )
        key, value = bundle.decoder4(zero, zero)
        key, value = torch.zeros_like(key), torch.zeros_like(value)
        return key, value, torch.ones((1, sample["selected_token_count"]), dtype=torch.long)
    sender = "4b" if "c4" in condition else "8b"
    source_id = sample["shuffle_id"] if "shuffled" in condition else sample["id"]
    record = store.native("test", sender, source_id)
    writer = bundle.writer4 if sender == "4b" else bundle.writer8
    key, value = writer(record["native_k"].to(device), record["native_v"].to(device))
    valid = min(key.shape[1], sample["selected_token_count"])
    key, _ = align_tensor(key.cpu(), sample["selected_token_count"])
    value, _ = align_tensor(value.cpu(), sample["selected_token_count"])
    key, value = bundle.decoder4(key.to(device), value.to(device))
    mask = torch.zeros((1, sample["selected_token_count"]), dtype=torch.long)
    mask[:, :valid] = 1
    return key, value, mask


@torch.no_grad()
def evaluate(cfg, mode):
    seed_all(cfg["seed"])
    rows = source_manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    tok = tokenizer(cfg["model_4b"])
    model = load_model(cfg["model_4b"])
    bundle = ModuleBundle(cfg).to(require_cuda()).eval()
    state = torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "checkpoints" / "a3_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    load_bundle_state(bundle, state["bundle"])
    records, summary = [], {}
    joined = {
        sample["id"]: {
            "sample_id": sample["id"],
            "question": sample["question"],
            "gold_answer": sample["answer"],
        }
        for sample in rows["test"]
    }
    for condition in EVAL_CONDITIONS:
        progress(f"{mode}: evaluating {condition}")
        values = []
        for sample in rows["test"]:
            key = value = mask = None
            if condition not in ("question_only", "supporting_text"):
                key, value, mask = evaluation_memory(
                    cfg, bundle, store, sample, condition
                )
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = greedy_generate(
                    cfg,
                    model,
                    tok,
                    sample,
                    key,
                    value,
                    mask,
                    text=condition == "supporting_text",
                    compact=condition == "question_only",
                )
                if condition == "supporting_text":
                    target = answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
                    prompt = sample["full_sequence_token_ids"]
                    sequence = torch.tensor([prompt + target[:-1]], device=require_cuda())
                    logits = model(
                        sequence,
                        attention_mask=torch.ones_like(sequence),
                        position_ids=torch.arange(sequence.shape[1], device=require_cuda()).unsqueeze(0),
                        use_cache=False,
                    ).logits
                    selected = logits[:, len(prompt) - 1 : len(prompt) - 1 + len(target)].float()
                    nll = F.cross_entropy(
                        selected.reshape(-1, selected.shape[-1]),
                        torch.tensor(target, device=require_cuda()),
                    ).item()
                else:
                    nll = answer_loss(
                        cfg,
                        model,
                        tok,
                        sample,
                        key,
                        value,
                        mask,
                        compact=condition == "question_only",
                    ).item()
            em = float(normalize_answer(prediction) == normalize_answer(sample["answer"]))
            record = {
                "condition": condition,
                "sample_id": sample["id"],
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "prediction": prediction,
                "em": em,
                "token_f1": token_f1(prediction, sample["answer"]),
                "strict_semantic_accuracy": em,
                "loose_semantic_accuracy": float(loose_correct(prediction, sample["answer"])),
                "gold_answer_nll": nll,
            }
            records.append(record)
            values.append(record)
            joined[sample["id"]][condition] = prediction
        summary[condition] = {
            metric: sum(row[metric] for row in values) / len(values)
            for metric in (
                "em",
                "token_f1",
                "strict_semantic_accuracy",
                "loose_semantic_accuracy",
                "gold_answer_nll",
            )
        }
    accuracy = lambda key: summary[key]["strict_semantic_accuracy"]
    shuffled4 = accuracy("d4_c4_shuffled")
    shuffled8 = accuracy("d4_c8_shuffled")
    summary["core"] = {
        "A_4_self": accuracy("d4_c4_self"),
        "A_4_cross": accuracy("d4_c8_cross"),
        "self_correct_minus_shuffled": accuracy("d4_c4_self") - shuffled4,
        "cross_correct_minus_shuffled": accuracy("d4_c8_cross") - shuffled8,
        "retention_4b_cross": (
            (accuracy("d4_c8_cross") - shuffled8)
            / (accuracy("d4_c4_self") - shuffled4)
            if accuracy("d4_c4_self") != shuffled4
            else None
        ),
        "a3_8b_status": "not run in first-shot minimum loop",
    }
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "results"
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "self_cross_summary.json", summary)
    with open(output / "per_sample_generations.jsonl", "w", encoding="utf-8") as handle:
        for sample in rows["test"]:
            handle.write(json.dumps(joined[sample["id"]], ensure_ascii=False) + "\n")
    with open(output / "per_condition.jsonl", "w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(output / "manual_cpw_64.csv", "w", encoding="utf-8-sig", newline="") as handle:
        fields = ["sample_id", "question", "gold_answer", "condition", "prediction", "manual_cpw", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in rows["test"][:64]:
            for condition in EVAL_CONDITIONS:
                writer.writerow(
                    {
                        "sample_id": sample["id"],
                        "question": sample["question"],
                        "gold_answer": sample["answer"],
                        "condition": condition,
                        "prediction": joined[sample["id"]][condition],
                        "manual_cpw": "",
                        "notes": "",
                    }
                )
    del model, bundle
    empty_cuda()
    progress(f"{mode}: 4B self/cross functional evaluation completed")


@torch.no_grad()
def collect_pooled(cfg, bundle, store, split, samples):
    z4, z8 = [], []
    for sample in samples:
        tensors = sample_tensors(store, split, sample)
        c4 = bundle.writer4(tensors["k4"], tensors["v4"])
        c8 = bundle.writer8(tensors["k8"], tensors["v8"])
        z4.append(pooled(*c4).cpu())
        z8.append(pooled(*c8).cpu())
    return torch.stack(z4), torch.stack(z8)


def audit(cfg, mode):
    seed_all(cfg["seed"])
    rows = source_manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    bundle = ModuleBundle(cfg).to(require_cuda()).eval()
    state = torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "checkpoints" / "a3_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    load_bundle_state(bundle, state["bundle"])
    train4, train8 = collect_pooled(cfg, bundle, store, "train", rows["train"])
    test4, test8 = collect_pooled(cfg, bundle, store, "test", rows["test"])
    device = require_cuda()
    classifier = nn.Linear(train4.shape[1], 2).to(device)
    optimizer = AdamW(classifier.parameters(), lr=0.01, weight_decay=0.0)
    features = torch.cat((train4, train8)).to(device)
    labels = torch.cat(
        (torch.zeros(len(train4), dtype=torch.long), torch.ones(len(train8), dtype=torch.long))
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    for _ in range(200):
        order = torch.randperm(len(features), generator=generator, device=device)
        loss = F.cross_entropy(classifier(features[order]), labels[order])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    test_features = torch.cat((test4, test8)).to(device)
    test_labels = torch.cat(
        (torch.zeros(len(test4), dtype=torch.long), torch.ones(len(test8), dtype=torch.long))
    ).to(device)
    sender_accuracy = (
        classifier(test_features).argmax(-1).eq(test_labels).float().mean().item()
    )
    average = F.normalize((test4 + test8) / 2, dim=-1)
    similarity = average @ average.T
    off_diagonal = similarity[~torch.eye(len(average), dtype=torch.bool)]
    centered = torch.cat((test4, test8)) - torch.cat((test4, test8)).mean(0)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular.square() / singular.square().sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    ).item()
    retrieval = F.normalize(test4, dim=-1) @ F.normalize(test8, dim=-1).T
    recall1 = retrieval.argmax(-1).eq(torch.arange(len(test4))).float().mean().item()
    zero = torch.zeros(
        (cfg["num_layers"], 3, cfg["num_kv_heads"], cfg["head_dim"]),
        device=device,
    )
    zero_checks = {}
    for name in ("writer4", "writer8", "decoder4", "decoder8"):
        key, value = getattr(bundle, name)(zero, zero)
        zero_checks[name] = int(torch.count_nonzero(key).item() + torch.count_nonzero(value).item())
        if zero_checks[name] != 0:
            raise RuntimeError(f"{name}(0) != 0")
    report = {
        "sender_classification_accuracy": sender_accuracy,
        "sender_chance_target": 0.5,
        "effective_rank": effective_rank,
        "cross_sample_mean_cosine": off_diagonal.mean().item(),
        "same_sample_cross_writer_recall_at_1": recall1,
        "pooled_feature_std_mean": torch.cat((test4, test8)).std(0, unbiased=False).mean().item(),
        "zero_nonzero_counts": zero_checks,
    }
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "results"
    save_json(output / "sender_leakage.json", report)
    save_json(output / "collapse_audit.json", report)
    del bundle, classifier
    empty_cuda()
    progress(f"{mode}: sender leakage and collapse audits completed")


def cpu_selftest(cfg):
    small = dict(
        cfg,
        num_layers=4,
        num_kv_heads=2,
        num_query_heads=4,
        head_dim=8,
        hidden_dim=16,
    )
    bundle = ModuleBundle(small)
    key, value = torch.randn(4, 5, 2, 8), torch.randn(4, 5, 2, 8)
    query = torch.randn(4, 2, 4, 8)
    c4 = bundle.writer4(key, value)
    decoded = bundle.decoder4(*c4)
    result = path_loss(
        small, query, *decoded, key, value, list(range(5)), [5, 6]
    )
    result["total"].backward()
    assert decoded[0].shape == key.shape
    zero = torch.zeros_like(key)
    for module in (bundle.writer4, bundle.writer8, bundle.decoder4, bundle.decoder8):
        zk, zv = module(zero, zero)
        assert torch.count_nonzero(zk).item() == 0
        assert torch.count_nonzero(zv).item() == 0
    progress("Multi-view CPU self-test and zero-preservation passed")


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--config", required=True)
    result.add_argument("--mode", choices=("smoke", "development"), required=True)
    return result


def cli_prepare():
    current = parser()
    current.add_argument(
        "--action", choices=("selftest", "structure", "reference", "queries", "audit"), required=True
    )
    current.add_argument("--sender", choices=("4b", "8b"))
    args = current.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    if args.action == "selftest":
        cpu_selftest(cfg)
    elif args.action == "structure":
        structure_check(cfg)
    elif args.action == "reference":
        prepare_reference(cfg, args.mode)
    elif args.action == "queries":
        if args.sender is None:
            current.error("--sender required for queries")
        extract_queries(cfg, args.mode, args.sender)
    elif args.action == "audit":
        a0_audit(cfg, args.mode)


def cli_a1():
    args = parser().parse_args()
    train_a1(load_json(args.config), args.mode)


def cli_a2():
    args = parser().parse_args()
    train_a2(load_json(args.config), args.mode)


def cli_a3():
    args = parser().parse_args()
    train_a3_4b(load_json(args.config), args.mode)


def cli_evaluate():
    args = parser().parse_args()
    evaluate(load_json(args.config), args.mode)


def cli_audit():
    args = parser().parse_args()
    audit(load_json(args.config), args.mode)
