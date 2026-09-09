from __future__ import annotations

import argparse
import csv
import gc
import hashlib
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
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from models import ReversibleCanonical4B


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


def manifest(cfg, mode):
    rows = load_json(
        Path(cfg["r1_dir"]) / "artifacts" / source_mode(mode) / "manifest.json"
    )
    sizes = cfg[f"{mode}_sizes"]
    selected = {
        split: [dict(sample) for sample in samples[: sizes[split]]]
        for split, samples in rows.items()
    }
    for split, samples in selected.items():
        for index, sample in enumerate(samples):
            wrong = next(
                (
                    samples[(index + offset) % len(samples)]
                    for offset in range(1, len(samples))
                    if normalize_answer(
                        samples[(index + offset) % len(samples)]["answer"]
                    )
                    != normalize_answer(sample["answer"])
                ),
                None,
            )
            if wrong is None:
                raise RuntimeError(f"{split} cannot construct answer-distinct shuffle")
            sample["source_shuffle_id"] = sample.get("shuffle_id")
            sample["shuffle_id"] = wrong["id"]
    return selected


class AssetStore:
    def __init__(self, cfg, mode, rows):
        self.cfg, self.mode = cfg, mode
        self.positions = {
            split: {sample["id"]: index for index, sample in enumerate(samples)}
            for split, samples in rows.items()
        }
        self.cache = {}

    def _load(self, key, path):
        if key not in self.cache:
            if len(self.cache) >= 12:
                self.cache.clear()
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.cache[key]

    def native(self, split, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["native_shard_size"]
        path = (
            Path(self.cfg["r1_dir"])
            / "cache"
            / source_mode(self.mode)
            / split
            / "4b"
            / f"shard_{shard:05d}.pt"
        )
        record = self._load(("native", split, shard), path)[
            index % self.cfg["native_shard_size"]
        ]
        if record["id"] != sample_id:
            raise RuntimeError("Native KV shard index mismatch")
        return record

    def query(self, split, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["query_shard_size"]
        path = (
            Path(self.cfg["a0_dir"])
            / "cache"
            / self.mode
            / split
            / "query_4b"
            / f"shard_{shard:05d}.pt"
        )
        record = self._load(("query", split, shard), path)[
            index % self.cfg["query_shard_size"]
        ]
        if record["id"] != sample_id:
            raise RuntimeError("Two-position Query shard index mismatch")
        return record


def structure_check(cfg, mode):
    conf = AutoConfig.from_pretrained(cfg["model_4b"], local_files_only=True)
    shape = (
        conf.num_hidden_layers,
        conf.num_key_value_heads,
        getattr(conf, "head_dim", conf.hidden_size // conf.num_attention_heads),
    )
    expected = (cfg["num_layers"], cfg["num_kv_heads"], cfg["head_dim"])
    if shape != expected:
        raise RuntimeError(f"4B model shape mismatch: {shape} != {expected}")
    reader = (
        Path(cfg["r1_dir"])
        / "artifacts"
        / source_mode(mode)
        / "sparse_reader"
        / "best.pt"
    )
    if not reader.exists():
        raise RuntimeError(f"R1 Sparse Reader checkpoint missing: {reader}")
    rows = manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    for split, samples in rows.items():
        for sample in samples:
            native = store.native(split, sample["id"])
            query = store.query(split, sample["id"])
            expected_native = (
                cfg["num_layers"],
                sample["selected_token_count"],
                cfg["num_kv_heads"],
                cfg["head_dim"],
            )
            if tuple(native["native_k"].shape) != expected_native:
                raise RuntimeError(f"{sample['id']} K shape mismatch")
            if tuple(native["native_v"].shape) != expected_native:
                raise RuntimeError(f"{sample['id']} V shape mismatch")
            if tuple(query["query"].shape) != (
                cfg["num_layers"],
                2,
                cfg["num_query_heads"],
                cfg["head_dim"],
            ):
                raise RuntimeError(f"{sample['id']} Query shape mismatch")
            if query["query_position_ids"] != [
                sample["question_position_ids"][-1],
                sample["question_position_ids"][-1] + 1,
            ]:
                raise RuntimeError(f"{sample['id']} Query positions mismatch")
    save_json(
        Path(cfg["work_dir"]) / "artifacts" / mode / "structure_check.json",
        {
            "passed": True,
            "model_shape": shape,
            "reader_checkpoint": str(reader),
            "native_k": "pre-RoPE",
            "native_v": True,
            "query_positions": ["last_question", "first_gold_answer"],
            "counts": {split: len(samples) for split, samples in rows.items()},
        },
    )
    progress(f"{mode}: structure and exact R1/A0 asset interface checks passed")


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
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
    for layer in model.model.layers:
        for name in ("q_proj", "o_proj"):
            setattr(
                layer.self_attn,
                name,
                LoRALinear(getattr(layer.self_attn, name)).to(require_cuda()),
            )
    checkpoint = torch.load(
        Path(cfg["r1_dir"])
        / "artifacts"
        / source_mode(mode)
        / "sparse_reader"
        / "best.pt",
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
    cosine, sine = model.model.rotary_emb(dummy, position_ids)
    post_key = (
        pre_key * cosine[0][None, :, None, :]
        + rotate_half(pre_key) * sine[0][None, :, None, :]
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
    compact=False,
):
    device = require_cuda()
    question = sample["question_token_ids"]
    target = answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
    current = torch.tensor([question + target[:-1]], dtype=torch.long, device=device)
    question_positions = (
        list(range(len(question))) if compact else sample["question_position_ids"]
    )
    answer_positions = list(
        range(question_positions[-1] + 1, question_positions[-1] + len(target))
    )
    kwargs = {}
    if key is not None:
        attention = torch.cat([prefix_mask.to(device), torch.ones_like(current)], 1)
        kwargs = {
            "past_key_values": dynamic_cache(
                model, key, value, sample["selected_position_ids"]
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
        position_ids=torch.tensor(
            [question_positions + answer_positions], dtype=torch.long, device=device
        ),
        use_cache=False,
        **kwargs,
    ).logits
    selected = logits[:, len(question) - 1 : len(question) - 1 + len(target)].float()
    return F.cross_entropy(
        selected.reshape(-1, selected.shape[-1]),
        torch.tensor(target, dtype=torch.long, device=device),
    )


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
    prompt = (
        sample["full_sequence_token_ids"]
        if supporting_text
        else sample["question_token_ids"]
    )
    positions = (
        list(range(len(prompt)))
        if supporting_text or compact
        else sample["question_position_ids"]
    )
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    kwargs = {}
    if key is not None:
        attention = torch.cat([prefix_mask.to(device), torch.ones_like(ids)], 1)
        kwargs = {
            "past_key_values": dynamic_cache(
                model, key, value, sample["selected_position_ids"]
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
        past = output.past_key_values
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    return tok.decode(generated, skip_special_tokens=True).strip()


def raw_memory(store, split, sample, kind="correct"):
    source_id = sample["shuffle_id"] if kind == "shuffled" else sample["id"]
    record = store.native(split, source_id)
    target = sample["selected_token_count"]
    valid = min(record["native_k"].shape[1], target)
    # Always detach controls from the shard cache. In particular, zero_() must
    # never mutate the cached Native tensors used by later conditions/metrics.
    key = record["native_k"][:, :target].half().clone()
    value = record["native_v"][:, :target].half().clone()
    if key.shape[1] < target:
        key = F.pad(key, (0, 0, 0, 0, 0, target - key.shape[1]))
        value = F.pad(value, (0, 0, 0, 0, 0, target - value.shape[1]))
    mask = torch.zeros((1, target), dtype=torch.long)
    mask[:, :valid] = 1
    if kind == "zero":
        key.zero_()
        value.zero_()
    return key, value, mask


def condition_row(condition, sample, prediction, nll):
    em = float(normalize_answer(prediction) == normalize_answer(sample["answer"]))
    return {
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


@torch.no_grad()
def native_baseline(cfg, mode):
    seed_all(cfg["seed"])
    if mode == "development":
        # The published 55.46875% R1 anchor is 71/128 on the complete formal
        # test split. A1 itself remains the requested 512/64/64 subset.
        rows = load_json(
            Path(cfg["r1_dir"]) / "artifacts" / "formal" / "manifest.json"
        )
        baseline_scope = "complete R1 formal test (128); A1 remains 512/64/64"
    else:
        rows = manifest(cfg, mode)
        baseline_scope = "smoke subset"
    store = AssetStore(cfg, mode, rows)
    tok = tokenizer(cfg["model_4b"])
    reader = frozen_sparse_reader(cfg, mode)
    set_lora(reader, True)
    values = []
    for sample in rows["test"]:
        key, value, mask = raw_memory(store, "test", sample)
        key, value = key.to(require_cuda()), value.to(require_cuda())
        prediction = greedy_generate(cfg, reader, tok, sample, key, value, mask)
        nll = answer_loss(cfg, reader, tok, sample, key, value, mask).item()
        values.append(condition_row("native_4b_sparse", sample, prediction, nll))
    summary = {
        "em": sum(row["em"] for row in values) / len(values),
        "token_f1": sum(row["token_f1"] for row in values) / len(values),
        "gold_answer_nll": sum(row["gold_answer_nll"] for row in values) / len(values),
        "reader_checkpoint": str(
            Path(cfg["r1_dir"])
            / "artifacts"
            / source_mode(mode)
            / "sparse_reader"
            / "best.pt"
        ),
        "current_script_execution": True,
        "evaluated_samples": len(rows["test"]),
        "baseline_scope": baseline_scope,
        "a1_test_samples": cfg[f"{mode}_sizes"]["test"],
    }
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "native_baseline"
    save_json(output / "summary.json", summary)
    save_json(output / "per_sample.json", values)
    if mode == "development" and abs(
        summary["em"] - cfg["expected_native_em"]
    ) > cfg["baseline_em_tolerance"]:
        raise RuntimeError(f"Native baseline hard gate failed: {summary}")
    progress(f"{mode}: current-script Native 4B baseline EM={summary['em']:.4f}")
    del reader
    empty_cuda()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def orthogonal(dimension, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(dimension, generator=generator)
    signs = (
        torch.randint(0, 2, (dimension,), generator=generator).float() * 2 - 1
    )
    matrix = torch.zeros(dimension, dimension, dtype=torch.float32)
    matrix[torch.arange(dimension), permutation] = signs
    q, r = torch.linalg.qr(matrix)
    diagonal_sign = torch.where(torch.diag(r) < 0, -1.0, 1.0)
    return q * diagonal_sign[None, :]


def compute_protocol(cfg, mode):
    rows = manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    shape = (cfg["num_layers"], cfg["num_kv_heads"], cfg["head_dim"])
    sum_k = torch.zeros(shape, dtype=torch.float32)
    sum_v = torch.zeros(shape, dtype=torch.float32)
    count = 0
    for index, sample in enumerate(rows["train"], 1):
        record = store.native("train", sample["id"])
        sum_k.add_(record["native_k"].float().square().sum(dim=1))
        sum_v.add_(record["native_v"].float().square().sum(dim=1))
        count += record["native_k"].shape[1]
        if index % 64 == 0 or index == len(rows["train"]):
            progress(f"{mode}: train-only RMS statistics {index}/{len(rows['train'])}")
    scale_k = (sum_k / count + cfg["scale_epsilon"]).sqrt().clamp_min(
        cfg["scale_min"]
    )
    scale_v = (sum_v / count + cfg["scale_epsilon"]).sqrt().clamp_min(
        cfg["scale_min"]
    )
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "protocol"
    output.mkdir(parents=True, exist_ok=True)
    stats_path = output / "stats.pt"
    basis_path = output / "basis.pt"
    protocol_path = output / "protocol.pt"
    torch.save(
        {
            "scale_k": scale_k,
            "scale_v": scale_v,
            "train_tokens": count,
            "train_samples": len(rows["train"]),
            "accumulation_dtype": "float32",
            "padding_included": False,
            "scale_min": cfg["scale_min"],
        },
        stats_path,
    )
    basis = {
        "rotation_k": orthogonal(cfg["head_dim"], cfg["seed"]),
        "rotation_v": orthogonal(cfg["head_dim"], cfg["seed"] + 1),
        "seed_k": cfg["seed"],
        "seed_v": cfg["seed"] + 1,
        "shared_across_layers": True,
        "basis_family": "seeded signed-permutation orthogonal basis produced by QR",
        "reason": "non-Native fixed coordinates with exact fp16 roundtrip",
    }
    torch.save(basis, basis_path)
    torch.save(
        {
            "scale_k": scale_k,
            "scale_v": scale_v,
            "rotation_k": basis["rotation_k"],
            "rotation_v": basis["rotation_v"],
        },
        protocol_path,
    )
    save_json(
        output / "sha256.json",
        {
            "stats.pt": sha256(stats_path),
            "basis.pt": sha256(basis_path),
            "protocol.pt": sha256(protocol_path),
            "validation_or_test_statistics_used": False,
        },
    )
    progress(f"{mode}: fixed RMS statistics and orthogonal bases saved")


def load_protocol(cfg, mode):
    return torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "protocol" / "protocol.pt",
        map_location="cpu",
        weights_only=False,
    )


def apply_rope(tensor, positions, theta):
    inverse = 1.0 / (
        float(theta)
        ** (
            torch.arange(0, tensor.shape[-1], 2, device=tensor.device).float()
            / tensor.shape[-1]
        )
    )
    frequency = torch.outer(
        torch.tensor(positions, device=tensor.device).float(), inverse
    )
    embedding = torch.cat((frequency, frequency), -1)
    cosine = embedding.cos()[None, :, None, :]
    sine = embedding.sin()[None, :, None, :]
    return tensor.float() * cosine + rotate_half(tensor.float()) * sine


def tensor_metrics(prediction, target):
    pred, gold = prediction.float(), target.float()
    difference = pred - gold
    layer_nmse = difference.square().mean((1, 2, 3)) / gold.square().mean(
        (1, 2, 3)
    ).clamp_min(1e-12)
    layer_cosine = F.cosine_similarity(
        pred.reshape(pred.shape[0], -1),
        gold.reshape(gold.shape[0], -1),
        dim=-1,
    )
    return {
        "nmse": (
            difference.square().mean() / gold.square().mean().clamp_min(1e-12)
        ),
        "cosine": F.cosine_similarity(pred.flatten(), gold.flatten(), dim=0),
        "max_abs_error": difference.abs().max(),
        "layer_nmse": layer_nmse,
        "layer_cosine": layer_cosine,
    }


def attention_metrics(cfg, query_record, pred_k, pred_v, native_k, native_v, sample):
    groups = cfg["num_query_heads"] // cfg["num_kv_heads"]
    query = apply_rope(
        query_record["query"].to(pred_k.device),
        query_record["query_position_ids"],
        cfg["rope_theta"],
    )
    pred_key = apply_rope(pred_k, sample["selected_position_ids"], cfg["rope_theta"])
    native_key = apply_rope(
        native_k, sample["selected_position_ids"], cfg["rope_theta"]
    )
    pred_key = pred_key.repeat_interleave(groups, dim=2)
    native_key = native_key.repeat_interleave(groups, dim=2)
    pred_value = pred_v.float().repeat_interleave(groups, dim=2)
    native_value = native_v.float().repeat_interleave(groups, dim=2)
    scale = math.sqrt(cfg["head_dim"])
    native_logits = torch.einsum("lqhd,lthd->lhqt", query, native_key) / scale
    pred_logits = torch.einsum("lqhd,lthd->lhqt", query, pred_key) / scale
    native_attention = native_logits.softmax(-1)
    pred_attention = pred_logits.softmax(-1)
    route_values = (
        native_attention
        * (
            native_attention.clamp_min(1e-12).log()
            - pred_attention.clamp_min(1e-12).log()
        )
    ).sum(-1)
    route_kl = route_values.mean()
    native_output = torch.einsum(
        "lhqt,lthd->lqhd", native_attention, native_value
    )
    pred_output = torch.einsum("lhqt,lthd->lqhd", pred_attention, pred_value)
    output_nmse = (pred_output - native_output).square().mean() / native_output.square().mean().clamp_min(
        1e-12
    )
    output_cosine = F.cosine_similarity(
        pred_output.flatten(), native_output.flatten(), dim=0
    )
    top1_values = (
        pred_attention.argmax(-1).eq(native_attention.argmax(-1)).float()
    )
    top1 = top1_values.mean()
    k = min(5, pred_attention.shape[-1])
    pred_top = pred_attention.topk(k, dim=-1).indices
    native_top = native_attention.topk(k, dim=-1).indices
    overlap_values = (
        pred_top[..., :, None].eq(native_top[..., None, :]).any(-1).float().mean(-1)
    )
    overlap = overlap_values.mean()
    pred_by_query = pred_output.permute(1, 0, 2, 3).flatten(1)
    native_by_query = native_output.permute(1, 0, 2, 3).flatten(1)
    output_cosine_by_query = F.cosine_similarity(
        pred_by_query, native_by_query, dim=-1
    )
    output_nmse_by_query = (
        (pred_output - native_output).square().mean((0, 2, 3))
        / native_output.square().mean((0, 2, 3)).clamp_min(1e-12)
    )
    return {
        "route_kl": route_kl,
        "route_kl_by_query": route_values.mean((0, 1)),
        "top1_agreement": top1,
        "top1_agreement_by_query": top1_values.mean((0, 1)),
        "top5_overlap": overlap,
        "top5_overlap_by_query": overlap_values.mean((0, 1)),
        "output_nmse": output_nmse,
        "output_nmse_by_query": output_nmse_by_query,
        "output_cosine": output_cosine,
        "output_cosine_by_query": output_cosine_by_query,
        "output_loss": output_nmse + 0.5 * (1 - output_cosine),
    }


def mean_scalar(items, key):
    return sum(float(item[key]) for item in items) / len(items)


def aggregate_roundtrip(items):
    report = {}
    for stream in ("k", "v"):
        report[stream] = {
            "nmse": mean_scalar(items, f"{stream}_nmse"),
            "cosine": mean_scalar(items, f"{stream}_cosine"),
            "max_abs_error": max(
                float(item[f"{stream}_max_abs_error"]) for item in items
            ),
        }
        report[stream]["per_layer_nmse"] = torch.stack(
            [item[f"{stream}_layer_nmse"] for item in items]
        ).mean(0).tolist()
        report[stream]["per_layer_cosine"] = torch.stack(
            [item[f"{stream}_layer_cosine"] for item in items]
        ).mean(0).tolist()
    report["attention"] = {
        key: mean_scalar(items, key)
        for key in (
            "route_kl",
            "top1_agreement",
            "top5_overlap",
            "output_nmse",
            "output_cosine",
            "output_loss",
        )
    }
    report["attention"]["query_positions"] = {
        label: {
            key: torch.stack([item[f"{key}_by_query"] for item in items])
            .mean(0)[query_index]
            .item()
            for key in (
                "route_kl",
                "top1_agreement",
                "top5_overlap",
                "output_nmse",
                "output_cosine",
            )
        }
        for query_index, label in enumerate(("last_question", "first_gold_answer"))
    }
    return report


def evaluate_roundtrip_samples(cfg, model, store, split, samples):
    device = require_cuda()
    items = []
    for sample in samples:
        native = store.native(split, sample["id"])
        key = native["native_k"].to(device)
        value = native["native_v"].to(device)
        decoded_k, decoded_v = model(key, value)
        km = tensor_metrics(decoded_k, key)
        vm = tensor_metrics(decoded_v, value)
        attention = attention_metrics(
            cfg, store.query(split, sample["id"]), decoded_k, decoded_v, key, value, sample
        )
        items.append(
            {
                "sample_id": sample["id"],
                **{f"k_{name}": tensor.detach().cpu() for name, tensor in km.items()},
                **{f"v_{name}": tensor.detach().cpu() for name, tensor in vm.items()},
                **{name: tensor.detach().cpu() for name, tensor in attention.items()},
            }
        )
    return aggregate_roundtrip(items), items


@torch.no_grad()
def functional_memory_evaluation(
    cfg, mode, split, samples, store, reader, tok, model, conditions
):
    records, summaries = [], {}
    for condition in conditions:
        values = []
        progress(f"{mode}: {split} functional condition {condition}")
        for sample in samples:
            key = value = mask = None
            if condition in ("native", "writer_bypass"):
                key, value, mask = raw_memory(store, split, sample)
                key, value = key.to(require_cuda()), value.to(require_cuda())
            elif condition == "decoded_correct":
                key, value, mask = raw_memory(store, split, sample)
                key, value = model(
                    key.to(require_cuda()), value.to(require_cuda())
                )
                key, value = key.half(), value.half()
            elif condition == "decoded_shuffled":
                key, value, mask = raw_memory(store, split, sample, "shuffled")
                key, value = model(
                    key.to(require_cuda()), value.to(require_cuda())
                )
                key, value = key.half(), value.half()
            elif condition == "decoded_zero":
                key, value, mask = raw_memory(store, split, sample, "zero")
                key, value = model(
                    key.to(require_cuda()), value.to(require_cuda())
                )
                key = torch.zeros_like(key).half()
                value = torch.zeros_like(value).half()
            prediction = greedy_generate(
                cfg, reader, tok, sample, key, value, mask
            )
            nll = answer_loss(
                cfg, reader, tok, sample, key, value, mask
            ).item()
            row = condition_row(condition, sample, prediction, nll)
            records.append(row)
            values.append(row)
        summaries[condition] = {
            "em": mean_scalar(values, "em"),
            "token_f1": mean_scalar(values, "token_f1"),
            "gold_answer_nll": mean_scalar(values, "gold_answer_nll"),
        }
    return summaries, records


@torch.no_grad()
def s0(cfg, mode):
    seed_all(cfg["seed"])
    rows = manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    model = ReversibleCanonical4B(cfg, load_protocol(cfg, mode)).to(require_cuda()).eval()
    zero = model.zero_check()
    numerical, per_sample = evaluate_roundtrip_samples(
        cfg, model, store, "validation", rows["validation"]
    )
    numerical["zero_checks"] = zero
    numerical["transform_math"] = "float32"
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "s0"
    save_json(output / "numerical_attention.json", numerical)
    save_json(
        output / "per_sample_numerical_attention.json",
        [
            {
                key: value.tolist() if isinstance(value, torch.Tensor) else value
                for key, value in item.items()
            }
            for item in per_sample
        ],
    )
    failures = []
    for stream in ("k", "v"):
        if numerical[stream]["nmse"] >= cfg["s0_nmse_max"]:
            failures.append(f"{stream} NMSE")
        if numerical[stream]["cosine"] <= cfg["s0_cosine_min"]:
            failures.append(f"{stream} cosine")
    if numerical["attention"]["route_kl"] >= cfg["s0_route_kl_max"]:
        failures.append("route KL")
    if numerical["attention"]["output_cosine"] <= cfg["s0_output_cosine_min"]:
        failures.append("attention output cosine")
    if max(zero.values()) >= 1e-6:
        failures.append("zero preservation")
    if failures:
        raise RuntimeError(f"S0 numerical hard gate failed: {failures}")

    tok = tokenizer(cfg["model_4b"])
    reader = frozen_sparse_reader(cfg, mode)
    set_lora(reader, True)
    summaries, records = functional_memory_evaluation(
        cfg,
        mode,
        "validation",
        rows["validation"],
        store,
        reader,
        tok,
        model,
        ("native", "decoded_correct", "decoded_shuffled"),
    )
    native_by_id = {
        row["sample_id"]: row
        for row in records
        if row["condition"] == "native"
    }
    decoded_by_id = {
        row["sample_id"]: row
        for row in records
        if row["condition"] == "decoded_correct"
    }
    match = sum(
        native_by_id[sample_id]["prediction"]
        == decoded_by_id[sample_id]["prediction"]
        for sample_id in native_by_id
    ) / len(native_by_id)
    functional = {
        "conditions": summaries,
        "native_decoded_em_delta": abs(
            summaries["native"]["em"] - summaries["decoded_correct"]["em"]
        ),
        "native_decoded_nll_delta": abs(
            summaries["native"]["gold_answer_nll"]
            - summaries["decoded_correct"]["gold_answer_nll"]
        ),
        "native_decoded_prediction_match": match,
        "correct_minus_shuffled": summaries["decoded_correct"]["em"]
        - summaries["decoded_shuffled"]["em"],
    }
    save_json(output / "functional_summary.json", functional)
    save_json(output / "functional_per_sample.json", records)
    if mode == "development":
        if functional["native_decoded_em_delta"] > cfg["s0_em_delta_max"]:
            failures.append("functional EM delta")
        if functional["native_decoded_nll_delta"] > cfg["s0_nll_delta_max"]:
            failures.append("functional NLL delta")
        if functional["native_decoded_prediction_match"] < cfg["s0_prediction_match_min"]:
            failures.append("prediction identity")
        if functional["correct_minus_shuffled"] < cfg["s0_shuffled_gap_min"]:
            failures.append("correct-shuffled gap")
    if failures:
        raise RuntimeError(f"S0 functional hard gate failed: {failures}")
    torch.save(
        {"model": model.state_dict(), "stage": "S0-untrained"},
        output / "untrained_exact_roundtrip.pt",
    )
    save_json(
        output / "PASSED.json",
        {
            "passed": True,
            "training_allowed": True,
            "numerical_thresholds": {
                "nmse_max": cfg["s0_nmse_max"],
                "cosine_min": cfg["s0_cosine_min"],
                "route_kl_max": cfg["s0_route_kl_max"],
                "output_cosine_min": cfg["s0_output_cosine_min"],
            },
        },
    )
    progress(f"{mode}: S0 numerical, attention, zero, and functional gates passed")
    del reader, model
    empty_cuda()


def training_components(cfg, model, store, split, sample):
    device = require_cuda()
    native = store.native(split, sample["id"])
    key = native["native_k"].to(device)
    value = native["native_v"].to(device)
    decoded_k, decoded_v = model(key, value)
    k_metrics = tensor_metrics(decoded_k, key)
    v_metrics = tensor_metrics(decoded_v, value)
    attention = attention_metrics(
        cfg,
        store.query(split, sample["id"]),
        decoded_k,
        decoded_v,
        key,
        value,
        sample,
    )
    kv_loss = k_metrics["nmse"] + v_metrics["nmse"]
    orth = model.orthogonality_loss()
    gate = model.gate_loss()
    loss = (
        kv_loss
        + 0.5 * attention["route_kl"]
        + attention["output_loss"]
        + 0.01 * orth
        + 0.001 * gate
    )
    return {
        "loss": loss,
        "kv": kv_loss,
        "route": attention["route_kl"],
        "output": attention["output_loss"],
        "output_cosine": attention["output_cosine"],
        "orth": orth,
        "gate": gate,
    }


@torch.no_grad()
def validation_score(cfg, model, store, samples):
    model.eval()
    values = [
        training_components(cfg, model, store, "validation", sample)
        for sample in samples
    ]
    report = {
        key: mean_scalar(values, key)
        for key in ("kv", "route", "output", "output_cosine", "orth", "gate")
    }
    report["score"] = report["output"] + 0.5 * report["route"] + 0.2 * report["kv"]
    return report


@torch.no_grad()
def functional_probe(cfg, model, store, reader, tok, samples):
    native, decoded = [], []
    for sample in samples:
        key, value, mask = raw_memory(store, "validation", sample)
        key, value = key.to(require_cuda()), value.to(require_cuda())
        native.append(answer_loss(cfg, reader, tok, sample, key, value, mask).item())
        decoded_key, decoded_value = model(key, value)
        decoded.append(
            answer_loss(
                cfg,
                reader,
                tok,
                sample,
                decoded_key.half(),
                decoded_value.half(),
                mask,
            ).item()
        )
    return {
        "native_nll": sum(native) / len(native),
        "decoded_nll": sum(decoded) / len(decoded),
        "decoded_minus_native_nll": sum(decoded) / len(decoded)
        - sum(native) / len(native),
    }


def checkpoint_payload(model, update, validation, probe):
    return {
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_updates": update,
        "validation": validation,
        "functional_probe": probe,
        "answer_ce_used_for_training": False,
        "cross_path_used": False,
        "sender": "qwen3-4b-only",
    }


def model_diagnostics(model):
    identity = torch.eye(
        model.writer_k.head.shape[-1], device=model.writer_k.head.device
    )
    heads = {
        "writer_k": model.writer_k.head,
        "writer_v": model.writer_v.head,
        "decoder_k": model.decoder_k.head,
        "decoder_v": model.decoder_v.head,
    }
    head_reports = {}
    for name, head in heads.items():
        identity_distance = (head - identity).square().mean((1, 2)).sqrt()
        orthogonality = (
            torch.matmul(head.transpose(-1, -2), head) - identity
        ).square().mean((1, 2)).sqrt()
        head_reports[name] = {
            "identity_distance_per_layer": identity_distance.detach().cpu().tolist(),
            "orthogonality_error_per_layer": orthogonality.detach().cpu().tolist(),
            "identity_distance_max": identity_distance.max().item(),
            "orthogonality_error_max": orthogonality.max().item(),
        }
    gates = {
        "writer_k_gamma": model.writer_k.gamma,
        "writer_v_gamma": model.writer_v.gamma,
        "writer_k_eta": model.writer_k.eta,
        "writer_v_eta": model.writer_v.eta,
        "decoder_k_beta": model.decoder_k.beta,
        "decoder_v_beta": model.decoder_v.beta,
    }
    gate_reports = {
        name: {
            "per_layer": value.detach().cpu().tolist(),
            "max_abs": value.detach().abs().max().item(),
            "rms": value.detach().square().mean().sqrt().item(),
        }
        for name, value in gates.items()
    }
    finite = all(
        torch.isfinite(parameter).all().item() for parameter in model.parameters()
    )
    return {
        "head_mixers": head_reports,
        "residual_gates": gate_reports,
        "all_parameters_finite": finite,
    }


def train_a1(cfg, mode):
    seed_all(cfg["seed"])
    s0_pass = Path(cfg["work_dir"]) / "artifacts" / mode / "s0" / "PASSED.json"
    if not s0_pass.exists():
        raise RuntimeError("S0 hard gate has not passed")
    rows = manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    model = ReversibleCanonical4B(cfg, load_protocol(cfg, mode)).to(require_cuda())
    reader = frozen_sparse_reader(cfg, mode)
    tok = tokenizer(cfg["model_4b"])
    set_lora(reader, True)
    optimizer = AdamW(
        [
            {"params": model.writer_parameters(), "lr": cfg["writer_lr"]},
            {"params": model.decoder_parameters(), "lr": cfg["decoder_lr"]},
        ],
        weight_decay=0.0,
    )
    scheduler = LambdaLR(
        optimizer,
        lambda update: min((update + 1) / cfg["warmup_updates"], 1.0),
    )
    maximum = cfg["smoke_max_updates"] if mode == "smoke" else cfg["max_updates"]
    evaluate_every = 1 if mode == "smoke" else cfg["eval_every"]
    probe_samples = rows["validation"][: cfg["functional_probe_size"]]
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)

    initial_validation = validation_score(cfg, model, store, rows["validation"])
    initial_probe = functional_probe(
        cfg, model, store, reader, tok, probe_samples
    )
    torch.save(
        checkpoint_payload(model, 0, initial_validation, initial_probe),
        output / "best.pt",
    )
    best_score = initial_validation["score"]
    best_update, bad_evals = 0, 0
    history, evaluations = [], [
        {"update": 0, **initial_validation, **initial_probe, "selected": True}
    ]
    samples = list(rows["train"])
    cursor, epoch = 0, 0
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        micro_values = []
        for _ in range(cfg["gradient_accumulation"]):
            if cursor == 0:
                random.Random(cfg["seed"] + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            model.train()
            components = training_components(cfg, model, store, "train", sample)
            (components["loss"] / cfg["gradient_accumulation"]).backward()
            micro_values.append(
                {
                    key: value.detach().item()
                    for key, value in components.items()
                }
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(
            {
                "update": update,
                "epoch": epoch,
                "writer_lr": optimizer.param_groups[0]["lr"],
                "decoder_lr": optimizer.param_groups[1]["lr"],
                **{
                    key: mean_scalar(micro_values, key)
                    for key in ("loss", "kv", "route", "output", "output_cosine", "orth", "gate")
                },
            }
        )
        if update % evaluate_every == 0 or update == maximum:
            validation = validation_score(cfg, model, store, rows["validation"])
            probe = functional_probe(cfg, model, store, reader, tok, probe_samples)
            probe_ok = (
                probe["decoded_minus_native_nll"] <= cfg["probe_nll_tolerance"]
            )
            selected = validation["score"] < best_score and probe_ok
            if selected:
                best_score, best_update, bad_evals = validation["score"], update, 0
                torch.save(
                    checkpoint_payload(model, update, validation, probe),
                    output / "best.pt",
                )
            else:
                bad_evals += 1
            evaluations.append(
                {
                    "update": update,
                    **validation,
                    **probe,
                    "functional_probe_ok": probe_ok,
                    "selected": selected,
                }
            )
            save_json(
                Path(cfg["work_dir"]) / "artifacts" / mode / "a1_history.json",
                history,
            )
            save_json(
                Path(cfg["work_dir"]) / "artifacts" / mode / "a1_evaluations.json",
                evaluations,
            )
            progress(
                f"{mode}: A1-v2 update {update}/{maximum}, "
                f"score={validation['score']:.6g}, best={best_update}"
            )
            if mode == "development" and bad_evals >= cfg["early_stop_patience"]:
                progress(f"{mode}: early stopping after {bad_evals} non-improving evals")
                break
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    save_json(
        Path(cfg["work_dir"]) / "artifacts" / mode / "a1_training_summary.json",
        {
            "completed_optimizer_updates": history[-1]["update"],
            "maximum_optimizer_updates": maximum,
            "best_update": best["optimizer_updates"],
            "best_validation": best["validation"],
            "best_functional_probe": best["functional_probe"],
            "early_stopped": history[-1]["update"] < maximum,
            "microbatch": 1,
            "gradient_accumulation": cfg["gradient_accumulation"],
        },
    )
    progress(f"{mode}: A1-v2 training completed; best update={best['optimizer_updates']}")
    del reader, model, optimizer
    empty_cuda()


@torch.no_grad()
def supporting_text_loss(cfg, reader, tok, sample):
    target = answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
    prompt = sample["full_sequence_token_ids"]
    sequence = torch.tensor(
        [prompt + target[:-1]], dtype=torch.long, device=require_cuda()
    )
    logits = reader(
        sequence,
        attention_mask=torch.ones_like(sequence),
        position_ids=torch.arange(
            sequence.shape[1], device=require_cuda()
        ).unsqueeze(0),
        use_cache=False,
    ).logits
    selected = logits[:, len(prompt) - 1 : len(prompt) - 1 + len(target)].float()
    return F.cross_entropy(
        selected.reshape(-1, selected.shape[-1]),
        torch.tensor(target, dtype=torch.long, device=require_cuda()),
    )


FINAL_CONDITIONS = (
    "question_only",
    "supporting_text",
    "native_4b",
    "decoded_correct",
    "decoded_shuffled",
    "decoded_zero",
    "writer_bypass",
)


@torch.no_grad()
def final_evaluate(cfg, mode):
    seed_all(cfg["seed"])
    rows = manifest(cfg, mode)
    store = AssetStore(cfg, mode, rows)
    regression_sample = rows["test"][0]
    cached = store.native("test", regression_sample["id"])
    cached_key = cached["native_k"].clone()
    cached_value = cached["native_v"].clone()
    raw_memory(store, "test", regression_sample, "zero")
    if not torch.equal(cached["native_k"], cached_key) or not torch.equal(
        cached["native_v"], cached_value
    ):
        raise RuntimeError("Control construction mutated the Native shard cache")
    model = ReversibleCanonical4B(cfg, load_protocol(cfg, mode)).to(require_cuda()).eval()
    checkpoint = torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    tok = tokenizer(cfg["model_4b"])
    reader = frozen_sparse_reader(cfg, mode)
    joined = {
        sample["id"]: {
            "sample_id": sample["id"],
            "question": sample["question"],
            "gold_answer": sample["answer"],
        }
        for sample in rows["test"]
    }
    records, summary = [], {}
    for condition in FINAL_CONDITIONS:
        progress(f"{mode}: final evaluation {condition}")
        set_lora(reader, condition not in ("question_only", "supporting_text"))
        values = []
        for sample in rows["test"]:
            key = value = mask = None
            if condition in ("native_4b", "writer_bypass"):
                key, value, mask = raw_memory(store, "test", sample)
                key, value = key.to(require_cuda()), value.to(require_cuda())
            elif condition == "decoded_correct":
                key, value, mask = raw_memory(store, "test", sample)
                key, value = model(
                    key.to(require_cuda()), value.to(require_cuda())
                )
                key, value = key.half(), value.half()
            elif condition == "decoded_shuffled":
                key, value, mask = raw_memory(store, "test", sample, "shuffled")
                key, value = model(
                    key.to(require_cuda()), value.to(require_cuda())
                )
                key, value = key.half(), value.half()
            elif condition == "decoded_zero":
                key, value, mask = raw_memory(store, "test", sample, "zero")
                key, value = model(
                    key.to(require_cuda()), value.to(require_cuda())
                )
                key, value = torch.zeros_like(key).half(), torch.zeros_like(value).half()
            prediction = greedy_generate(
                cfg,
                reader,
                tok,
                sample,
                key,
                value,
                mask,
                supporting_text=condition == "supporting_text",
                compact=condition == "question_only",
            )
            if condition == "supporting_text":
                nll = supporting_text_loss(cfg, reader, tok, sample).item()
            else:
                nll = answer_loss(
                    cfg,
                    reader,
                    tok,
                    sample,
                    key,
                    value,
                    mask,
                    compact=condition == "question_only",
                ).item()
            row = condition_row(condition, sample, prediction, nll)
            records.append(row)
            values.append(row)
            joined[sample["id"]][condition] = prediction
        summary[condition] = {
            metric: mean_scalar(values, metric)
            for metric in (
                "em",
                "token_f1",
                "strict_semantic_accuracy",
                "loose_semantic_accuracy",
                "gold_answer_nll",
            )
        }

    numerical, _ = evaluate_roundtrip_samples(
        cfg, model, store, "test", rows["test"]
    )
    diagnostics = model_diagnostics(model)
    qonly = summary["question_only"]["em"]
    native = summary["native_4b"]["em"]
    decoded = summary["decoded_correct"]["em"]
    shuffled = summary["decoded_shuffled"]["em"]
    retention = (
        (decoded - qonly) / (native - qonly) if native != qonly else None
    )
    gates = model.zero_check()
    core = {
        "retention_self": retention,
        "decoded_correct_minus_shuffled": decoded - shuffled,
        "native_minus_decoded": native - decoded,
        "writer_bypass_minus_native": summary["writer_bypass"]["em"] - native,
        "k_cosine": numerical["k"]["cosine"],
        "v_cosine": numerical["v"]["cosine"],
        "attention_output_cosine": numerical["attention"]["output_cosine"],
        "zero_checks": gates,
        "model_diagnostics": diagnostics,
        "best_optimizer_update": checkpoint["optimizer_updates"],
        "cross_model_stage": "not run",
    }
    failures = []
    if mode == "development":
        if numerical["k"]["cosine"] <= cfg["a1_cosine_min"]:
            failures.append("K cosine")
        if numerical["v"]["cosine"] <= cfg["a1_cosine_min"]:
            failures.append("V cosine")
        if numerical["attention"]["output_cosine"] <= cfg["a1_output_cosine_min"]:
            failures.append("attention output cosine")
        if retention is None or retention < cfg["a1_retention_min"]:
            failures.append("self retention")
        if decoded - shuffled < cfg["a1_correct_shuffled_gap_min"]:
            failures.append("correct-shuffled gap")
        if max(gates.values()) >= 1e-6:
            failures.append("zero preservation")
        if not diagnostics["all_parameters_finite"]:
            failures.append("non-finite parameter")
    core["passed"] = not failures
    core["failures"] = failures
    summary["core"] = core
    summary["post_training_numerical_attention"] = numerical
    output = Path(cfg["work_dir"]) / "artifacts" / mode / "results"
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "summary.json", summary)
    with open(output / "per_condition.jsonl", "w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(
        output / "per_sample_generations.jsonl", "w", encoding="utf-8"
    ) as handle:
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in rows["test"][:64]:
            for condition in FINAL_CONDITIONS:
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
    if failures:
        raise RuntimeError(f"A1-v2 final hard gate failed: {failures}")
    progress(f"{mode}: final A1-v2 evaluation completed and gates passed")
    del reader, model
    empty_cuda()


def cpu_selftest(cfg):
    small = dict(
        cfg,
        num_layers=4,
        num_kv_heads=2,
        num_query_heads=4,
        head_dim=8,
        trunk_hidden_dim=16,
        adapter_rank=2,
    )
    generator = torch.Generator().manual_seed(cfg["seed"])
    qk, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
    qv, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
    protocol = {
        "scale_k": torch.rand(4, 2, 8, generator=generator).clamp_min(0.1),
        "scale_v": torch.rand(4, 2, 8, generator=generator).clamp_min(0.1),
        "rotation_k": qk,
        "rotation_v": qv,
    }
    model = ReversibleCanonical4B(small, protocol)
    key = torch.randn(4, 5, 2, 8, generator=generator)
    value = torch.randn(4, 5, 2, 8, generator=generator)
    decoded_k, decoded_v = model(key, value)
    if not torch.allclose(decoded_k, key, atol=1e-5, rtol=1e-5):
        raise RuntimeError("CPU exact K roundtrip failed")
    if not torch.allclose(decoded_v, value, atol=1e-5, rtol=1e-5):
        raise RuntimeError("CPU exact V roundtrip failed")
    zero = model.zero_check()
    if max(zero.values()) != 0:
        raise RuntimeError(f"zero-preservation failed: {zero}")
    loss = (decoded_k - key).square().mean() + (decoded_v - value).square().mean()
    loss.backward()
    progress("CPU reversible roundtrip, gradients, and zero-preservation passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument(
        "action",
        choices=("selftest", "structure", "baseline", "protocol", "s0", "train", "evaluate"),
    )
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    if args.action == "selftest":
        cpu_selftest(cfg)
    elif args.action == "structure":
        structure_check(cfg, args.mode)
    elif args.action == "baseline":
        native_baseline(cfg, args.mode)
    elif args.action == "protocol":
        compute_protocol(cfg, args.mode)
    elif args.action == "s0":
        s0(cfg, args.mode)
    elif args.action == "train":
        train_a1(cfg, args.mode)
    elif args.action == "evaluate":
        final_evaluate(cfg, args.mode)


if __name__ == "__main__":
    main()
