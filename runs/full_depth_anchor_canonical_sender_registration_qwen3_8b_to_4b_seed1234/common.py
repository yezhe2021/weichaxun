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


def manifest(cfg, mode):
    return load_json(Path(cfg["r1_dir"]) / "artifacts" / mode / "manifest.json")


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
            raise RuntimeError(f"{sender} structure mismatch")
    if report["4b"]["rope_theta"] != report["8b"]["rope_theta"] or report["4b"][
        "rope_scaling"
    ] != report["8b"]["rope_scaling"]:
        raise RuntimeError("RoPE configurations differ")
    for mode in ("smoke", "formal"):
        checkpoint = Path(cfg["r1_dir"]) / "artifacts" / mode / "sparse_reader" / "best.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"R1 sparse Reader missing: {checkpoint}")
        report[f"r1_sparse_reader_{mode}"] = str(checkpoint)
    report["canonical_v1"] = "Qwen3-4B 36-layer selected pre-RoPE K/native V"
    report["writer_4b"] = "identity"
    report["reader_frozen"] = True
    save_json(Path(cfg["work_dir"]) / "artifacts" / "structure_check.json", report)
    progress("Anchor protocol, RoPE, and frozen R1 sparse Reader checks passed")


def prepare_reference(cfg, mode):
    rows = manifest(cfg, mode)
    save_json(
        Path(cfg["work_dir"]) / "artifacts" / mode / "manifest_reference.json",
        {
            "source": str(Path(cfg["r1_dir"]) / "artifacts" / mode / "manifest.json"),
            "sample_ids": {split: [x["id"] for x in samples] for split, samples in rows.items()},
            "counts": {split: len(samples) for split, samples in rows.items()},
            "position_policy": "original offsets in complete Context->Question sequence",
            "canonical_v1": "identity Qwen3-4B native KV, all 36 layers",
        },
    )
    progress(f"{mode}: exact R0/R1 sample and position reference prepared")


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

    def run(self, model, sample):
        self.values.clear()
        device = require_cuda()
        self.indices = torch.tensor(
            sample["question_position_ids"], dtype=torch.long, device=device
        )
        ids = torch.tensor([sample["full_sequence_token_ids"]], dtype=torch.long, device=device)
        model(
            ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1], device=device).unsqueeze(0),
            use_cache=False,
        )
        if len(self.values) != self.cfg["num_layers"]:
            raise RuntimeError("Question Query capture did not cover all 36 layers")
        return torch.stack([self.values[index] for index in range(self.cfg["num_layers"])])

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def extract_queries(cfg, mode):
    model = load_model(cfg["model_4b"])
    hook = QueryCapture(model, cfg)
    rows = manifest(cfg, mode)
    root = Path(cfg["work_dir"]) / "cache" / mode
    for split, samples in rows.items():
        directory = root / split / "query_4b"
        directory.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(samples), cfg["shard_size"]):
            shard = start // cfg["shard_size"]
            destination = directory / f"shard_{shard:05d}.pt"
            if destination.exists():
                continue
            records = [
                {"id": sample["id"], "query_4b": hook.run(model, sample)}
                for sample in samples[start : start + cfg["shard_size"]]
            ]
            torch.save(records, destination)
            progress(
                f"{mode}: 4B Question Query {split} shard {shard + 1}/"
                f"{math.ceil(len(samples) / cfg['shard_size'])}"
            )
    hook.close()
    del model
    empty_cuda()


class AssetStore:
    def __init__(self, cfg, mode, rows):
        self.cfg, self.mode = cfg, mode
        self.positions = {
            split: {row["id"]: index for index, row in enumerate(samples)}
            for split, samples in rows.items()
        }
        self.cache = {}

    def _load(self, key, path):
        if key not in self.cache:
            if len(self.cache) >= 8:
                self.cache.clear()
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.cache[key]

    def native(self, split, sender, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["shard_size"]
        path = (
            Path(self.cfg["r1_dir"])
            / "cache"
            / self.mode
            / split
            / sender
            / f"shard_{shard:05d}.pt"
        )
        records = self._load(("native", split, sender, shard), path)
        record = records[index % self.cfg["shard_size"]]
        if record["id"] != sample_id:
            raise RuntimeError("Native shard index mismatch")
        return record

    def query(self, split, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["shard_size"]
        path = (
            Path(self.cfg["work_dir"])
            / "cache"
            / self.mode
            / split
            / "query_4b"
            / f"shard_{shard:05d}.pt"
        )
        records = self._load(("query", split, shard), path)
        record = records[index % self.cfg["shard_size"]]
        if record["id"] != sample_id:
            raise RuntimeError("Query shard index mismatch")
        return record["query_4b"]


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden, gamma):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma), dtype=torch.float32))

    def forward(self, x):
        normalized = F.rms_norm(
            x,
            (x.shape[-1],),
            self.norm.weight.to(x.dtype),
            self.norm.eps,
        )
        hidden = F.linear(
            normalized,
            self.fc1.weight.to(x.dtype),
            None,
        )
        residual = F.linear(
            F.silu(hidden),
            self.fc2.weight.to(x.dtype),
            None,
        )
        return x + self.gamma.to(x.dtype) * residual


class FullDepthWriter8B(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers, heads = cfg["num_layers"], cfg["num_kv_heads"]
        identity = torch.eye(heads).unsqueeze(0).repeat(layers, 1, 1)
        self.head_k = nn.Parameter(identity.clone())
        self.head_v = nn.Parameter(identity.clone())
        self.k_mlps = nn.ModuleList(
            [
                ResidualMLP(
                    cfg["head_dim"], cfg["writer_hidden_dim"], cfg["writer_gamma_init"]
                )
                for _ in range(layers)
            ]
        )
        self.v_mlps = nn.ModuleList(
            [
                ResidualMLP(
                    cfg["head_dim"], cfg["writer_hidden_dim"], cfg["writer_gamma_init"]
                )
                for _ in range(layers)
            ]
        )

    def forward(self, key, value):
        mixed_k = torch.einsum("loi,lsid->lsod", self.head_k.to(key.dtype), key)
        mixed_v = torch.einsum("loi,lsid->lsod", self.head_v.to(value.dtype), value)
        output_k = torch.stack([module(mixed_k[i]) for i, module in enumerate(self.k_mlps)])
        output_v = torch.stack([module(mixed_v[i]) for i, module in enumerate(self.v_mlps)])
        return output_k, output_v


def layerwise_alignment(prediction, target, cosine_weight=1.0):
    pred, gold = prediction.float(), target.float()
    nmse = (pred - gold).square().mean(dim=(1, 2, 3)) / (
        gold.square().mean(dim=(1, 2, 3)) + 1e-6
    )
    cosine = 1 - F.cosine_similarity(
        pred.reshape(pred.shape[0], -1, pred.shape[-1]),
        gold.reshape(gold.shape[0], -1, gold.shape[-1]),
        dim=-1,
    ).mean(dim=1)
    return (nmse + cosine_weight * cosine).mean()


def repeat_kv(tensor, groups):
    return tensor.repeat_interleave(groups, dim=2)


def apply_rope_positions(tensor, positions, theta):
    device = tensor.device
    inverse_frequency = 1.0 / (
        float(theta)
        ** (
            torch.arange(0, tensor.shape[-1], 2, device=device, dtype=torch.float32)
            / tensor.shape[-1]
        )
    )
    frequencies = torch.outer(
        torch.tensor(positions, device=device, dtype=torch.float32),
        inverse_frequency,
    )
    embedding = torch.cat([frequencies, frequencies], dim=-1)
    cosine = embedding.cos().to(tensor.dtype)[None, :, None, :]
    sine = embedding.sin().to(tensor.dtype)[None, :, None, :]
    return tensor * cosine + rotate_half(tensor) * sine


def functional_alignment(
    cfg,
    query,
    pred_k,
    pred_v,
    target_k,
    target_v,
    evidence_positions,
    question_positions,
):
    groups = cfg["num_query_heads"] // cfg["num_kv_heads"]
    query = apply_rope_positions(
        query.to(pred_k.device), question_positions, cfg["rope_theta"]
    )
    target_k = apply_rope_positions(
        target_k, evidence_positions, cfg["rope_theta"]
    )
    pred_k = apply_rope_positions(pred_k, evidence_positions, cfg["rope_theta"])
    target_k_gqa = repeat_kv(target_k, groups)
    pred_k_gqa = repeat_kv(pred_k, groups)
    target_v_gqa = repeat_kv(target_v, groups)
    pred_v_gqa = repeat_kv(pred_v, groups)
    scale = math.sqrt(cfg["head_dim"])
    target_logits = torch.einsum("lqhd,lthd->lhqt", query, target_k_gqa) / scale
    pred_logits = torch.einsum("lqhd,lthd->lhqt", query, pred_k_gqa) / scale
    target_attention = target_logits.float().softmax(-1)
    pred_log_attention = pred_logits.float().log_softmax(-1)
    route = (
        target_attention
        * (target_attention.clamp_min(1e-9).log() - pred_log_attention)
    ).sum(-1).mean()
    target_output = torch.einsum(
        "lhqt,lthd->lqhd", target_attention.to(target_v_gqa.dtype), target_v_gqa
    )
    pred_attention = pred_logits.softmax(-1)
    pred_output = torch.einsum("lhqt,lthd->lqhd", pred_attention, pred_v_gqa)
    output = layerwise_alignment(pred_output, target_output, cosine_weight=1.0)
    return route, output


def align_tensor(tensor, target_length, kind):
    if kind == "zero":
        return torch.zeros(
            (tensor.shape[0], target_length, tensor.shape[2], tensor.shape[3]),
            dtype=torch.float16,
        ), target_length
    valid = min(tensor.shape[1], target_length)
    output = tensor[:, :target_length].half()
    if output.shape[1] < target_length:
        output = F.pad(output, (0, 0, 0, 0, 0, target_length - output.shape[1]))
    return output, valid


def raw_memory(store, split, sender, sample, kind):
    source_id = sample["shuffle_id"] if kind == "shuffled" else sample["id"]
    record = store.native(split, sender, source_id)
    key, valid = align_tensor(
        record["native_k"], sample["selected_token_count"], kind
    )
    value, _ = align_tensor(
        record["native_v"], sample["selected_token_count"], kind
    )
    mask = torch.zeros((1, sample["selected_token_count"]), dtype=torch.long)
    mask[:, :valid] = 1
    return key, value, mask


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        output = self.base(x)
        if self.enabled:
            hidden = F.linear(self.dropout(x), self.lora_A.to(x.dtype))
            output = output + F.linear(hidden, self.lora_B.to(x.dtype)) * self.scale
        return output


def frozen_sparse_reader(cfg, mode):
    model = load_model(cfg["model_4b"])
    device = require_cuda()
    for layer in model.model.layers:
        for name in ("q_proj", "o_proj"):
            setattr(
                layer.self_attn,
                name,
                LoRALinear(getattr(layer.self_attn, name)).to(device),
            )
    checkpoint = torch.load(
        Path(cfg["r1_dir"]) / "artifacts" / mode / "sparse_reader" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    parameters = dict(model.named_parameters())
    for name, value in checkpoint["lora"].items():
        parameters[name].data.copy_(value.to(parameters[name].device))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def set_lora(model, enabled):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def dynamic_cache(model, pre_key, value, positions):
    device = pre_key.device
    position_ids = torch.tensor([positions], dtype=torch.long, device=device)
    dummy = torch.empty(
        (1, len(positions), model.config.hidden_size),
        dtype=pre_key.dtype,
        device=device,
    )
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    post_key = (
        pre_key * cos[0][None, :, None, :]
        + rotate_half(pre_key) * sin[0][None, :, None, :]
    )
    items = [
        (
            post_key[layer].permute(1, 0, 2).unsqueeze(0),
            value[layer].permute(1, 0, 2).unsqueeze(0),
        )
        for layer in range(pre_key.shape[0])
    ]
    return DynamicCache(ddp_cache_data=items, config=model.config)


def answer_target(tok, answer, maximum):
    ids = tok(" " + answer, add_special_tokens=False).input_ids[: maximum - 1]
    ids.append(tok.eos_token_id)
    return ids


def answer_loss(
    cfg,
    model,
    tok,
    sample,
    key=None,
    value=None,
    prefix_mask=None,
    compact_positions=False,
):
    device = require_cuda()
    question = sample["question_token_ids"]
    target = answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
    current = torch.tensor([question + target[:-1]], dtype=torch.long, device=device)
    question_positions = (
        list(range(len(question)))
        if compact_positions
        else sample["question_position_ids"]
    )
    answer_positions = list(
        range(
            question_positions[-1] + 1,
            question_positions[-1] + len(target),
        )
    )
    positions = torch.tensor(
        [question_positions + answer_positions],
        dtype=torch.long,
        device=device,
    )
    kwargs = {}
    if key is not None:
        mask = torch.cat(
            [prefix_mask.to(device), torch.ones_like(current, dtype=torch.long)], 1
        )
        kwargs = {
            "past_key_values": dynamic_cache(
                model, key, value, sample["selected_position_ids"]
            ),
            "cache_position": torch.arange(
                key.shape[1], key.shape[1] + current.shape[1], device=device
            ),
        }
    else:
        mask = torch.ones_like(current)
    logits = model(
        current,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        **kwargs,
    ).logits
    selected = logits[:, len(question) - 1 : len(question) - 1 + len(target)].float()
    gold = torch.tensor(target, dtype=torch.long, device=device)
    return F.cross_entropy(selected.reshape(-1, selected.shape[-1]), gold)


def train_warmup(cfg, mode):
    seed_all(cfg["seed"])
    rows = manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    writer = FullDepthWriter8B(cfg).to(require_cuda()).train()
    optimizer = AdamW(writer.parameters(), lr=cfg["warmup_lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "warmup"
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1):
        samples = list(rows["train"])
        random.Random(cfg["seed"] + epoch).shuffle(samples)
        optimizer.zero_grad(set_to_none=True)
        for index, sample in enumerate(samples, 1):
            source = store.native("train", "8b", sample["id"])
            target = store.native("train", "4b", sample["id"])
            query = store.query("train", sample["id"]).to(require_cuda())
            source_k = source["native_k"].to(require_cuda())
            source_v = source["native_v"].to(require_cuda())
            target_k = target["native_k"].to(require_cuda())
            target_v = target["native_v"].to(require_cuda())
            with torch.autocast("cuda", dtype=torch.float16):
                pred_k, pred_v = writer(source_k, source_v)
                key_loss = layerwise_alignment(
                    pred_k, target_k, cfg["kv_cosine_weight"]
                )
                value_loss = layerwise_alignment(
                    pred_v, target_v, cfg["kv_cosine_weight"]
                )
                route, out_loss = functional_alignment(
                    cfg,
                    query,
                    pred_k,
                    pred_v,
                    target_k,
                    target_v,
                    sample["selected_position_ids"],
                    sample["question_position_ids"],
                )
                loss = (
                    key_loss
                    + value_loss
                    + cfg["route_weight"] * route
                    + cfg["warmup_output_weight"] * out_loss
                )
            scaler.scale(loss / cfg["gradient_accumulation"]).backward()
            if index % cfg["gradient_accumulation"] == 0 or index == len(samples):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            history.append(
                {
                    "sample": index,
                    "kv_k": key_loss.item(),
                    "kv_v": value_loss.item(),
                    "route_kl": route.item(),
                    "attention_output": out_loss.item(),
                    "total": loss.item(),
                }
            )
        progress(f"{mode}: full-depth Writer warm-up epoch completed")
    torch.save({"writer": writer.state_dict()}, output / "final.pt")
    save_json(output / "history.json", history)
    del writer, optimizer
    empty_cuda()


@torch.no_grad()
def validation_gap(cfg, mode, reader, tok, writer, store, samples):
    writer.eval()
    correct_values, shuffled_values = [], []
    for sample in samples:
        source_k, source_v, mask = raw_memory(
            store, "validation", "8b", sample, "correct"
        )
        wrong_k, wrong_v, wrong_mask = raw_memory(
            store, "validation", "8b", sample, "shuffled"
        )
        pred_k, pred_v = writer(source_k.to(require_cuda()), source_v.to(require_cuda()))
        wrong_k, wrong_v = writer(
            wrong_k.to(require_cuda()), wrong_v.to(require_cuda())
        )
        correct_values.append(
            answer_loss(cfg, reader, tok, sample, pred_k, pred_v, mask).item()
        )
        shuffled_values.append(
            answer_loss(
                cfg, reader, tok, sample, wrong_k, wrong_v, wrong_mask
            ).item()
        )
    correct = sum(correct_values) / len(correct_values)
    shuffled = sum(shuffled_values) / len(shuffled_values)
    return correct, shuffled, shuffled - correct


def train_functional(cfg, mode):
    seed_all(cfg["seed"])
    rows = manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    tok = tokenizer(cfg["model_4b"])
    reader = frozen_sparse_reader(cfg, mode)
    writer = FullDepthWriter8B(cfg).to(require_cuda())
    warmup = torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "warmup" / "final.pt",
        map_location="cpu",
        weights_only=False,
    )
    writer.load_state_dict(warmup["writer"])
    writer.train()
    optimizer = AdamW(writer.parameters(), lr=cfg["functional_lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    epochs = 1 if mode == "smoke" else cfg["functional_epochs"]
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "functional"
    output.mkdir(parents=True, exist_ok=True)
    history, best_gap, best_correct = [], -float("inf"), float("inf")
    for epoch in range(epochs):
        samples = list(rows["train"])
        random.Random(cfg["seed"] + epoch).shuffle(samples)
        optimizer.zero_grad(set_to_none=True)
        for index, sample in enumerate(samples, 1):
            source = store.native("train", "8b", sample["id"])
            target = store.native("train", "4b", sample["id"])
            wrong = store.native("train", "8b", sample["shuffle_id"])
            query = store.query("train", sample["id"]).to(require_cuda())
            wrong_k, valid = align_tensor(
                wrong["native_k"], sample["selected_token_count"], "shuffled"
            )
            wrong_v, _ = align_tensor(
                wrong["native_v"], sample["selected_token_count"], "shuffled"
            )
            correct_mask = torch.ones((1, sample["selected_token_count"]), dtype=torch.long)
            wrong_mask = torch.zeros_like(correct_mask)
            wrong_mask[:, :valid] = 1
            with torch.autocast("cuda", dtype=torch.float16):
                pred_k, pred_v = writer(
                    source["native_k"].to(require_cuda()),
                    source["native_v"].to(require_cuda()),
                )
                correct_nll = answer_loss(
                    cfg, reader, tok, sample, pred_k, pred_v, correct_mask
                )
                target_k = target["native_k"].to(require_cuda())
                target_v = target["native_v"].to(require_cuda())
                kv = layerwise_alignment(
                    pred_k, target_k, cfg["kv_cosine_weight"]
                ) + layerwise_alignment(
                    pred_v, target_v, cfg["kv_cosine_weight"]
                )
                _, out_loss = functional_alignment(
                    cfg,
                    query,
                    pred_k,
                    pred_v,
                    target_k,
                    target_v,
                    sample["selected_position_ids"],
                    sample["question_position_ids"],
                )
                shuffled_k, shuffled_v = writer(
                    wrong_k.to(require_cuda()), wrong_v.to(require_cuda())
                )
                shuffled_nll = answer_loss(
                    cfg,
                    reader,
                    tok,
                    sample,
                    shuffled_k,
                    shuffled_v,
                    wrong_mask,
                )
                dependence = F.relu(
                    cfg["dependence_margin"] + correct_nll - shuffled_nll
                )
                loss = (
                    correct_nll
                    + cfg["functional_kv_weight"] * kv
                    + cfg["functional_output_weight"] * out_loss
                    + cfg["dependence_weight"] * dependence
                )
            scaler.scale(loss / cfg["gradient_accumulation"]).backward()
            if index % cfg["gradient_accumulation"] == 0 or index == len(samples):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            history.append(
                {
                    "epoch": epoch + 1,
                    "sample": index,
                    "answer_ce": correct_nll.item(),
                    "shuffled_nll": shuffled_nll.item(),
                    "kv": kv.item(),
                    "attention_output": out_loss.item(),
                    "dependence": dependence.item(),
                    "total": loss.item(),
                }
            )
        correct, shuffled, gap = validation_gap(
            cfg, mode, reader, tok, writer, store, rows["validation"]
        )
        if best_gap == -float("inf") or (
            gap > best_gap and correct <= best_correct
        ):
            best_gap, best_correct = gap, correct
            torch.save(
                {
                    "writer": writer.state_dict(),
                    "epoch": epoch + 1,
                    "validation_correct_nll": correct,
                    "validation_shuffled_nll": shuffled,
                    "validation_gap": gap,
                },
                output / "best.pt",
            )
        save_json(output / "history.json", history)
        progress(f"{mode}: frozen-Reader Writer registration epoch {epoch + 1}/{epochs}")
        writer.train()
    del reader, writer, optimizer
    empty_cuda()


@torch.no_grad()
def greedy_generate(
    cfg,
    model,
    tok,
    sample,
    key=None,
    value=None,
    prefix_mask=None,
    supporting_text=False,
    compact=False,
):
    device = require_cuda()
    if supporting_text:
        prompt = sample["full_sequence_token_ids"]
        positions = list(range(len(prompt)))
    else:
        prompt = sample["question_token_ids"]
        positions = (
            list(range(len(prompt))) if compact else sample["question_position_ids"]
        )
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    kwargs = {}
    if key is not None:
        attention_mask = torch.cat(
            [prefix_mask.to(device), torch.ones_like(input_ids)], 1
        )
        kwargs = {
            "past_key_values": dynamic_cache(
                model, key, value, sample["selected_position_ids"]
            ),
            "cache_position": torch.arange(
                key.shape[1], key.shape[1] + input_ids.shape[1], device=device
            ),
        }
    else:
        attention_mask = torch.ones_like(input_ids)
    output = model(
        input_ids,
        attention_mask=attention_mask,
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
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], 1
        )
        output = model(
            token,
            attention_mask=attention_mask,
            position_ids=torch.tensor([[next_position]], dtype=torch.long, device=device),
            cache_position=torch.tensor(
                [past.get_seq_length()], dtype=torch.long, device=device
            ),
            past_key_values=past,
            use_cache=True,
        )
        past, token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    return tok.decode(generated, skip_special_tokens=True).strip()


def pairwise_rows(cfg, mode):
    path = (
        Path(cfg["r0_dir"]) / "artifacts" / mode / "evaluation" / "per_sample.jsonl"
    )
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["condition"] == "K_8b_correct_pair":
            rows[row["sample_id"]] = row
    return rows


def r1_reference_rows(cfg, mode):
    path = (
        Path(cfg["r1_dir"])
        / "artifacts"
        / mode
        / "evaluation"
        / "per_condition.jsonl"
    )
    wanted = {
        "native_4b_sparse_correct",
        "cross_8b_raw_native_correct",
    }
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["condition"] in wanted:
            rows[(row["condition"], row["sample_id"])] = row
    return rows


@torch.no_grad()
def verify_reader_protocol(cfg, mode):
    """Hard Gate: no Writer is instantiated or trained in this function."""
    seed_all(cfg["seed"])
    rows = manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    tok = tokenizer(cfg["model_4b"])
    reader = frozen_sparse_reader(cfg, mode)
    set_lora(reader, True)
    references = r1_reference_rows(cfg, mode)
    conditions = [
        ("native_4b_sparse_correct", "4b"),
        ("cross_8b_raw_native_correct", "8b"),
    ]
    report = {
        "reader_checkpoint": str(
            Path(cfg["r1_dir"])
            / "artifacts"
            / mode
            / "sparse_reader"
            / "best.pt"
        ),
        "writer_instantiated": False,
        "writer_trained": False,
        "conditions": {},
    }
    outputs = []
    for condition, sender in conditions:
        progress(f"{mode}: Reader protocol Gate evaluating {condition}")
        matches, em_values, nll_values, reference_nll = [], [], [], []
        for sample in rows["test"]:
            key, value, mask = raw_memory(
                store, "test", sender, sample, "correct"
            )
            key, value = key.to(require_cuda()), value.to(require_cuda())
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = greedy_generate(
                    cfg, reader, tok, sample, key, value, mask
                )
                nll = answer_loss(
                    cfg, reader, tok, sample, key, value, mask
                ).item()
            reference = references[(condition, sample["id"])]
            matches.append(prediction == reference["prediction"])
            em = float(
                normalize_answer(prediction)
                == normalize_answer(sample["answer"])
            )
            em_values.append(em)
            nll_values.append(nll)
            reference_nll.append(reference["gold_answer_nll"])
            outputs.append(
                {
                    "condition": condition,
                    "sample_id": sample["id"],
                    "prediction": prediction,
                    "reference_prediction": reference["prediction"],
                    "prediction_identical": prediction == reference["prediction"],
                    "em": em,
                    "gold_answer_nll": nll,
                    "reference_gold_answer_nll": reference["gold_answer_nll"],
                }
            )
        current_em = sum(em_values) / len(em_values)
        reference_em = sum(
            references[(condition, sample["id"])]["strict_semantic_accuracy"]
            for sample in rows["test"]
        ) / len(rows["test"])
        current_nll = sum(nll_values) / len(nll_values)
        expected_nll = sum(reference_nll) / len(reference_nll)
        condition_report = {
            "em": current_em,
            "r1_reference_em": reference_em,
            "em_absolute_delta": abs(current_em - reference_em),
            "gold_answer_nll": current_nll,
            "r1_reference_gold_answer_nll": expected_nll,
            "nll_absolute_delta": abs(current_nll - expected_nll),
            "all_generations_identical": all(matches),
        }
        report["conditions"][condition] = condition_report
        if (
            not condition_report["all_generations_identical"]
            or condition_report["em_absolute_delta"] > 1e-12
            or condition_report["nll_absolute_delta"] > 1e-4
        ):
            save_json(
                Path(cfg["work_dir"])
                / "artifacts"
                / mode
                / "reader_protocol_gate"
                / "report.json",
                report,
            )
            raise RuntimeError(
                f"{condition} failed exact R1 sparse-interface reproduction: "
                f"{condition_report}"
            )
    report["passed"] = True
    output = (
        Path(cfg["work_dir"])
        / "artifacts"
        / mode
        / "reader_protocol_gate"
    )
    save_json(output / "report.json", report)
    save_json(output / "per_sample.json", outputs)
    progress(
        f"{mode}: frozen R1 sparse Reader Gate passed; "
        f"4B EM={report['conditions']['native_4b_sparse_correct']['em']:.4f}, "
        f"8B raw EM={report['conditions']['cross_8b_raw_native_correct']['em']:.4f}"
    )
    del reader
    empty_cuda()


FINAL_CONDITIONS = [
    {"key": "question_only", "lora": False, "memory": None, "compact": True},
    {"key": "supporting_text", "lora": False, "memory": None, "text": True},
    {"key": "native_4b", "lora": True, "memory": ("4b", "correct")},
    {"key": "native_4b_shuffled", "lora": True, "memory": ("4b", "shuffled")},
    {"key": "raw_native_8b", "lora": True, "memory": ("8b", "correct")},
    {"key": "writer_8b_correct", "lora": True, "memory": ("writer", "correct")},
    {"key": "writer_8b_shuffled", "lora": True, "memory": ("writer", "shuffled")},
    {"key": "writer_8b_zero", "lora": True, "memory": ("writer", "zero")},
]


@torch.no_grad()
def evaluate(cfg, mode):
    seed_all(cfg["seed"])
    rows = manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    tok = tokenizer(cfg["model_4b"])
    reader = frozen_sparse_reader(cfg, mode)
    writer = FullDepthWriter8B(cfg).to(require_cuda()).eval()
    state = torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "functional" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    writer.load_state_dict(state["writer"])
    records, summary = [], {}
    joined = {
        sample["id"]: {
            "sample_id": sample["id"],
            "question": sample["question"],
            "gold_answer": sample["answer"],
        }
        for sample in rows["test"]
    }
    for condition in FINAL_CONDITIONS:
        set_lora(reader, condition["lora"])
        values = []
        progress(f"{mode}: evaluating {condition['key']}")
        for sample in rows["test"]:
            key = value = mask = None
            if condition["memory"] is not None:
                sender, kind = condition["memory"]
                if sender == "writer":
                    key, value, mask = raw_memory(
                        store, "test", "8b", sample, kind
                    )
                    key, value = writer(
                        key.to(require_cuda()), value.to(require_cuda())
                    )
                    if kind == "zero":
                        key = torch.zeros_like(key)
                        value = torch.zeros_like(value)
                else:
                    key, value, mask = raw_memory(
                        store, "test", sender, sample, kind
                    )
                    key, value = key.to(require_cuda()), value.to(require_cuda())
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = greedy_generate(
                    cfg,
                    reader,
                    tok,
                    sample,
                    key,
                    value,
                    mask,
                    supporting_text=condition.get("text", False),
                    compact=condition.get("compact", False),
                )
                if condition.get("text", False):
                    target = answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
                    prompt = sample["full_sequence_token_ids"]
                    sequence = torch.tensor([prompt + target[:-1]], device=require_cuda())
                    logits = reader(
                        sequence,
                        attention_mask=torch.ones_like(sequence),
                        position_ids=torch.arange(
                            sequence.shape[1], device=require_cuda()
                        ).unsqueeze(0),
                        use_cache=False,
                    ).logits
                    selected = logits[
                        :, len(prompt) - 1 : len(prompt) - 1 + len(target)
                    ].float()
                    nll = F.cross_entropy(
                        selected.reshape(-1, selected.shape[-1]),
                        torch.tensor(target, device=require_cuda()),
                    ).item()
                else:
                    nll = answer_loss(
                        cfg,
                        reader,
                        tok,
                        sample,
                        key,
                        value,
                        mask,
                        compact_positions=condition.get("compact", False),
                    ).item()
            em = float(
                normalize_answer(prediction) == normalize_answer(sample["answer"])
            )
            row = {
                "condition": condition["key"],
                "sample_id": sample["id"],
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "prediction": prediction,
                "em": em,
                "token_f1": token_f1(prediction, sample["answer"]),
                "strict_semantic_accuracy": em,
                "loose_semantic_accuracy": float(
                    loose_correct(prediction, sample["answer"])
                ),
                "gold_answer_nll": nll,
            }
            records.append(row)
            values.append(row)
            joined[sample["id"]][condition["key"]] = prediction
        summary[condition["key"]] = {
            metric: sum(row[metric] for row in values) / len(values)
            for metric in (
                "em",
                "token_f1",
                "strict_semantic_accuracy",
                "loose_semantic_accuracy",
                "gold_answer_nll",
            )
        }

    pair = pairwise_rows(cfg, mode)
    pair_values = []
    for sample in rows["test"]:
        row = pair[sample["id"]]
        joined[sample["id"]]["pair_reader_8b"] = row["prediction"]
        pair_values.append(row)
        records.append(
            {
                "condition": "pair_reader_8b",
                "sample_id": sample["id"],
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "prediction": row["prediction"],
                "em": row["em"],
                "token_f1": row["token_f1"],
                "strict_semantic_accuracy": row["strict_semantic_accuracy"],
                "loose_semantic_accuracy": row["loose_semantic_accuracy"],
                "gold_answer_nll": row["gold_answer_nll"],
                "source": "R0 K_8b_correct_pair",
            }
        )
    summary["pair_reader_8b"] = {
        metric: sum(row[metric] for row in pair_values) / len(pair_values)
        for metric in (
            "em",
            "token_f1",
            "strict_semantic_accuracy",
            "loose_semantic_accuracy",
            "gold_answer_nll",
        )
    }
    acc = lambda key: summary[key]["strict_semantic_accuracy"]
    aq, writer_acc, raw_acc, pair_acc = (
        acc("question_only"),
        acc("writer_8b_correct"),
        acc("raw_native_8b"),
        acc("pair_reader_8b"),
    )
    summary["core"] = {
        "A_Q": aq,
        "A_4": acc("native_4b"),
        "A_raw8": raw_acc,
        "A_writer8": writer_acc,
        "A_pair8": pair_acc,
        "writer_gain": writer_acc - raw_acc,
        "retention_pair": (
            (writer_acc - aq) / (pair_acc - aq) if pair_acc != aq else None
        ),
        "writer_correct_minus_shuffled": writer_acc
        - acc("writer_8b_shuffled"),
        "writer_correct_minus_shuffled_nll_advantage": summary[
            "writer_8b_shuffled"
        ]["gold_answer_nll"]
        - summary["writer_8b_correct"]["gold_answer_nll"],
    }
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "summary.json", summary)
    with open(output / "per_condition.jsonl", "w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(output / "full_generations.jsonl", "w", encoding="utf-8") as handle:
        for sample in rows["test"]:
            handle.write(json.dumps(joined[sample["id"]], ensure_ascii=False) + "\n")
    with open(
        output / "manual_cpw_64.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        fields = [
            "sample_id",
            "question",
            "gold_answer",
            "condition",
            "prediction",
            "manual_cpw",
            "notes",
        ]
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        for sample in rows["test"][:64]:
            for condition in [x["key"] for x in FINAL_CONDITIONS] + [
                "pair_reader_8b"
            ]:
                csv_writer.writerow(
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
    del reader, writer
    empty_cuda()
    progress(f"{mode}: full-depth anchor evaluation completed")


def cpu_selftest(cfg):
    small = dict(
        cfg,
        num_layers=4,
        num_kv_heads=2,
        num_query_heads=4,
        head_dim=8,
        writer_hidden_dim=16,
    )
    writer = FullDepthWriter8B(small)
    key = torch.randn(4, 5, 2, 8)
    value = torch.randn(4, 5, 2, 8)
    target_k, target_v = torch.randn_like(key), torch.randn_like(value)
    query = torch.randn(4, 3, 4, 8)
    pred_k, pred_v = writer(key, value)
    route, output = functional_alignment(
        small,
        query,
        pred_k,
        pred_v,
        target_k,
        target_v,
        list(range(5)),
        list(range(5, 8)),
    )
    loss = layerwise_alignment(pred_k, target_k) + route + output
    loss.backward()
    assert pred_k.shape == key.shape
    assert writer.head_k.grad is not None
    assert torch.allclose(
        writer.head_k.detach(), torch.eye(2).unsqueeze(0).repeat(4, 1, 1)
    )
    zero = torch.zeros_like(key)
    zero_k, zero_v = writer(zero, zero)
    assert torch.count_nonzero(zero_k).item() == 0
    assert torch.count_nonzero(zero_v).item() == 0
    progress("Full-depth Writer CPU self-test passed")


def common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    return parser


def cli_prepare():
    parser = common_parser()
    parser.add_argument(
        "--action", choices=("selftest", "structure", "reference", "queries"), required=True
    )
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    if args.action == "selftest":
        cpu_selftest(cfg)
    elif args.action == "structure":
        structure_check(cfg)
    elif args.action == "reference":
        prepare_reference(cfg, args.mode)
    elif args.action == "queries":
        extract_queries(cfg, args.mode)


def cli_warmup():
    args = common_parser().parse_args()
    cfg = load_json(args.config)
    train_warmup(cfg, args.mode)


def cli_functional():
    args = common_parser().parse_args()
    cfg = load_json(args.config)
    train_functional(cfg, args.mode)


def cli_evaluate():
    args = common_parser().parse_args()
    cfg = load_json(args.config)
    evaluate(cfg, args.mode)


def cli_verify():
    args = common_parser().parse_args()
    cfg = load_json(args.config)
    verify_reader_protocol(cfg, args.mode)
