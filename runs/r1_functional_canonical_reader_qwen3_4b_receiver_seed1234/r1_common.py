from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
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
        raise RuntimeError("CUDA is unavailable; R1 requires a working GPU")
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


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, answer):
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(answer).split()
    common = Counter(pred) & Counter(gold)
    overlap = sum(common.values())
    if not pred or not gold:
        return float(pred == gold)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def loose_correct(prediction, answer):
    pred, gold = normalize_answer(prediction), normalize_answer(answer)
    return bool(pred and gold and (gold in pred or pred in gold))


def format_raw_example(row, tok):
    support = {(str(title), int(index)) for title, index in row["supporting_facts"]}
    context_ids, selected_positions, selected_sentence_ids = [], [], []
    for title, sentences in row["context"]:
        context_ids.extend(tok(f"Document: {title}\n", add_special_tokens=False).input_ids)
        for sentence_index, sentence in enumerate(sentences):
            context_ids.extend(
                tok(f"Sentence {sentence_index}: ", add_special_tokens=False).input_ids
            )
            start = len(context_ids)
            content = tok(sentence, add_special_tokens=False).input_ids
            context_ids.extend(content)
            end = len(context_ids)
            context_ids.extend(tok("\n", add_special_tokens=False).input_ids)
            if (str(title), sentence_index) in support:
                selected_positions.extend(range(start, end))
                selected_sentence_ids.append(
                    {"title": title, "sentence_index": sentence_index, "start": start, "end": end}
                )
    question_ids = tok(
        f"\nQuestion: {row['question'].strip()}\n\nAnswer:",
        add_special_tokens=False,
    ).input_ids
    question_start = len(context_ids)
    full_ids = context_ids + question_ids
    if not selected_positions:
        raise RuntimeError(f"{row['_id']} has no selected supporting token")
    return {
        "id": row["_id"],
        "type": row.get("type"),
        "level": row.get("level"),
        "question": row["question"].strip(),
        "answer": str(row["answer"]).strip(),
        "full_context_token_ids": context_ids,
        "full_sequence_token_ids": full_ids,
        "question_token_ids": question_ids,
        "selected_token_indices": selected_positions,
        "selected_sentence_ids": selected_sentence_ids,
        "selected_position_ids": selected_positions,
        "question_position_ids": list(range(question_start, len(full_ids))),
        "selected_token_count": len(selected_positions),
    }


def length_bucket(length, bounds):
    for index, bound in enumerate(bounds):
        if length <= bound:
            return index
    return len(bounds)


def assign_shuffles(samples, cfg):
    for sample in samples:
        candidates = [
            other
            for other in samples
            if other["id"] != sample["id"]
            and normalize_answer(other["answer"]) != normalize_answer(sample["answer"])
        ]
        if not candidates:
            raise RuntimeError("Cannot construct answer-different shuffled memory")
        bucket = length_bucket(sample["selected_token_count"], cfg["length_buckets"])
        same = [
            x
            for x in candidates
            if length_bucket(x["selected_token_count"], cfg["length_buckets"]) == bucket
        ]
        pool = same or candidates
        chosen = min(
            pool,
            key=lambda x: (
                abs(x["selected_token_count"] - sample["selected_token_count"]),
                x["id"],
            ),
        )
        sample["shuffle_id"] = chosen["id"]


def prepare_manifests(cfg, mode):
    r0 = load_json(Path(cfg["r0_dir"]) / "artifacts" / mode / "splits.json")
    train_raw = {row["_id"]: row for row in load_json(cfg["hotpot_train"])}
    dev_raw = {row["_id"]: row for row in load_json(cfg["hotpot_dev"])}
    tok4 = tokenizer(cfg["model_4b"])
    tok8 = tokenizer(cfg["model_8b"])
    if tok4.get_vocab() != tok8.get_vocab():
        raise RuntimeError("4B and 8B tokenizer vocabularies differ")
    manifest = {}
    for split, r0_samples in r0.items():
        raw_map = train_raw if split == "train" else dev_raw
        current = []
        for r0_sample in r0_samples:
            sample = format_raw_example(raw_map[r0_sample["id"]], tok4)
            if sample["question"] != r0_sample["question"] or sample["answer"] != r0_sample["answer"]:
                raise RuntimeError(f"R0/raw mismatch for {sample['id']}")
            if tok4.convert_ids_to_tokens(sample["full_sequence_token_ids"]) != tok8.convert_ids_to_tokens(
                sample["full_sequence_token_ids"]
            ):
                raise RuntimeError(f"Cross-sender tokenizer mismatch for {sample['id']}")
            current.append(sample)
        assign_shuffles(current, cfg)
        manifest[split] = current
    out = Path(cfg["work_dir"]) / "artifacts" / mode
    save_json(out / "manifest.json", manifest)
    save_json(
        out / "position_audit.json",
        {
            "source": str(Path(cfg["r0_dir"]) / "artifacts" / mode / "splits.json"),
            "same_sample_ids_as_r0": {
                split: [x["id"] for x in manifest[split]] == [x["id"] for x in r0[split]]
                for split in manifest
            },
            "position_policy": "selected and question positions are original offsets in complete Context->Question sequence",
            "counts": {split: len(rows) for split, rows in manifest.items()},
            "selected_token_range": {
                split: {
                    "min": min(x["selected_token_count"] for x in rows),
                    "max": max(x["selected_token_count"] for x in rows),
                }
                for split, rows in manifest.items()
            },
        },
    )
    progress(f"{mode}: reconstructed original positions for exact R0 sample IDs")


def structure_check(cfg):
    report = {}
    for name in ("4b", "8b"):
        conf = AutoConfig.from_pretrained(cfg[f"model_{name}"], local_files_only=True)
        report[name] = {
            "layers": conf.num_hidden_layers,
            "kv_heads": conf.num_key_value_heads,
            "head_dim": getattr(conf, "head_dim", conf.hidden_size // conf.num_attention_heads),
            "rope_theta": getattr(conf, "rope_theta", None),
            "rope_scaling": getattr(conf, "rope_scaling", None),
        }
        actual = (report[name]["layers"], report[name]["kv_heads"], report[name]["head_dim"])
        expected = (cfg["num_layers"], cfg["num_kv_heads"], cfg["head_dim"])
        if actual != expected:
            raise RuntimeError(f"{name} structure {actual} != {expected}")
    if report["4b"]["rope_theta"] != report["8b"]["rope_theta"] or report["4b"][
        "rope_scaling"
    ] != report["8b"]["rope_scaling"]:
        raise RuntimeError("4B/8B RoPE configurations differ")
    checkpoint = Path(cfg["p0a_dir"]) / "checkpoints" / "shared_formal.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Frozen Writer checkpoint missing: {checkpoint}")
    report["writer_checkpoint"] = str(checkpoint)
    report["writer_frozen"] = True
    save_json(Path(cfg["work_dir"]) / "artifacts" / "structure_check.json", report)
    progress("R1 structure and frozen-Writer asset check passed")


def import_p0a(cfg):
    path = Path(cfg["p0a_dir"]) / "p0a.py"
    spec = importlib.util.spec_from_file_location("r1_frozen_p0a", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_writer(cfg):
    module = import_p0a(cfg)
    source_cfg = load_json(Path(cfg["p0a_dir"]) / "config.json")
    system = module.CanonicalSystem(source_cfg).to(require_cuda())
    state = torch.load(
        Path(cfg["p0a_dir"]) / "checkpoints" / "shared_formal.pt",
        map_location="cpu",
        weights_only=True,
    )
    system.load_state_dict(state["model"])
    system.eval()
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    return system


class NativeCapture:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.values = {}
        self.indices = None
        self.handles = [
            layer.self_attn.register_forward_pre_hook(self._hook(index), with_kwargs=True)
            for index, layer in enumerate(model.model.layers)
        ]

    def _hook(self, layer_index):
        def capture(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            batch, length, _ = hidden.shape
            heads, dim = self.cfg["num_kv_heads"], self.cfg["head_dim"]
            key = module.k_norm(module.k_proj(hidden).view(batch, length, heads, dim))
            value = module.v_proj(hidden).view(batch, length, heads, dim)
            self.values[layer_index] = (
                key[0, self.indices].detach().cpu().half().contiguous(),
                value[0, self.indices].detach().cpu().half().contiguous(),
            )

        return capture

    def run(self, model, sample):
        self.values.clear()
        device = require_cuda()
        self.indices = torch.tensor(
            sample["selected_token_indices"], dtype=torch.long, device=device
        )
        ids = torch.tensor([sample["full_sequence_token_ids"]], dtype=torch.long, device=device)
        positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        model(
            ids,
            attention_mask=torch.ones_like(ids),
            position_ids=positions,
            use_cache=False,
        )
        if len(self.values) != self.cfg["num_layers"]:
            raise RuntimeError(f"Captured {len(self.values)} layers instead of 36")
        key = torch.stack([self.values[i][0] for i in range(self.cfg["num_layers"])])
        value = torch.stack([self.values[i][1] for i in range(self.cfg["num_layers"])])
        return key, value

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def extract_sender_assets(cfg, mode, sender):
    model = load_model(cfg[f"model_{sender}"])
    writer = load_frozen_writer(cfg)
    writer_module = writer.b if sender == "4b" else writer.a
    capture = NativeCapture(model, cfg)
    manifest = load_json(Path(cfg["work_dir"]) / "artifacts" / mode / "manifest.json")
    root = Path(cfg["work_dir"]) / "cache" / mode
    shard_size = cfg["shard_size"]
    writer_layers = cfg["selected_layers"]
    for split, samples in manifest.items():
        out_dir = root / split / sender
        out_dir.mkdir(parents=True, exist_ok=True)
        for shard_start in range(0, len(samples), shard_size):
            shard_index = shard_start // shard_size
            destination = out_dir / f"shard_{shard_index:05d}.pt"
            if destination.exists():
                continue
            records = []
            for sample in samples[shard_start : shard_start + shard_size]:
                native_k, native_v = capture.run(model, sample)
                selected_k = native_k[writer_layers].to(require_cuda())
                selected_v = native_v[writer_layers].to(require_cuda())
                with torch.autocast("cuda", dtype=torch.float16):
                    canonical_k = writer_module.k(selected_k).detach().cpu().half().contiguous()
                    canonical_v = writer_module.v(selected_v).detach().cpu().half().contiguous()
                records.append(
                    {
                        "id": sample["id"],
                        "native_k": native_k,
                        "native_v": native_v,
                        "canonical_k": canonical_k,
                        "canonical_v": canonical_v,
                    }
                )
            torch.save(records, destination)
            progress(
                f"{mode}: {sender} {split} shard {shard_index + 1}/"
                f"{math.ceil(len(samples) / shard_size)}"
            )
    capture.close()
    del model, writer
    empty_cuda()


class ShardStore:
    def __init__(self, cfg, mode, manifest):
        self.cfg, self.mode, self.manifest = cfg, mode, manifest
        self.positions = {
            split: {sample["id"]: index for index, sample in enumerate(samples)}
            for split, samples in manifest.items()
        }
        self.cached_key, self.cached_records = None, None

    def get(self, split, sender, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["shard_size"]
        key = (split, sender, shard)
        if key != self.cached_key:
            path = (
                Path(self.cfg["work_dir"])
                / "cache"
                / self.mode
                / split
                / sender
                / f"shard_{shard:05d}.pt"
            )
            self.cached_records = torch.load(path, map_location="cpu", weights_only=False)
            self.cached_key = key
        record = self.cached_records[index % self.cfg["shard_size"]]
        if record["id"] != sample_id:
            raise RuntimeError(f"Shard index mismatch for {sample_id}")
        return record


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
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


def inject_lora(model, cfg):
    device = next(model.parameters()).device
    for layer in model.model.layers:
        for name in ("q_proj", "o_proj"):
            base = getattr(layer.self_attn, name)
            setattr(
                layer.self_attn,
                name,
                LoRALinear(
                    base,
                    cfg["lora_rank"],
                    cfg["lora_alpha"],
                    cfg["lora_dropout"],
                ).to(device),
            )
    return model


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def lora_state(model):
    return {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if "lora_A" in n or "lora_B" in n
    }


def load_lora(model, path):
    state = torch.load(path, map_location="cpu", weights_only=False)["lora"]
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameters[name].data.copy_(value.to(parameters[name].device))


def set_lora(model, enabled):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden, gamma):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.tensor(float(gamma), dtype=torch.float32))

    def forward(self, x):
        norm_weight = self.norm.weight.to(x.dtype)
        normalized = F.rms_norm(x, (x.shape[-1],), norm_weight, self.norm.eps)
        hidden = F.linear(
            normalized,
            self.fc1.weight.to(x.dtype),
            self.fc1.bias.to(x.dtype) if self.fc1.bias is not None else None,
        )
        residual = F.linear(
            F.silu(hidden),
            self.fc2.weight.to(x.dtype),
            self.fc2.bias.to(x.dtype) if self.fc2.bias is not None else None,
        )
        return x + self.gamma.to(x.dtype) * residual


def depth_initial_logits(target_layers, canonical_layers):
    weights = torch.zeros(len(target_layers), len(canonical_layers))
    for row, target in enumerate(target_layers):
        if target <= canonical_layers[0]:
            weights[row, 0] = 1
        elif target >= canonical_layers[-1]:
            weights[row, -1] = 1
        else:
            for index in range(len(canonical_layers) - 1):
                left, right = canonical_layers[index], canonical_layers[index + 1]
                if left <= target <= right:
                    weights[row, index] = (right - target) / (right - left)
                    weights[row, index + 1] = (target - left) / (right - left)
                    break
    return (weights + 1e-8).log()


class CanonicalTranslator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, hidden = cfg["head_dim"], cfg["translator_hidden_dim"]
        initial = depth_initial_logits(list(range(cfg["num_layers"])), cfg["selected_layers"])
        self.depth_k = nn.Parameter(initial.clone())
        self.depth_v = nn.Parameter(initial.clone())
        self.k_mlps = nn.ModuleList(
            [ResidualMLP(dim, hidden, cfg["translator_gamma_init"]) for _ in range(cfg["num_layers"])]
        )
        self.v_mlps = nn.ModuleList(
            [ResidualMLP(dim, hidden, cfg["translator_gamma_init"]) for _ in range(cfg["num_layers"])]
        )

    def forward(self, canonical_k, canonical_v):
        weight_k = self.depth_k.softmax(-1).to(canonical_k.dtype)
        weight_v = self.depth_v.softmax(-1).to(canonical_v.dtype)
        mixed_k = torch.einsum("ri,ishd->rshd", weight_k, canonical_k)
        mixed_v = torch.einsum("ri,ishd->rshd", weight_v, canonical_v)
        output_k = torch.stack([module(mixed_k[i]) for i, module in enumerate(self.k_mlps)])
        output_v = torch.stack([module(mixed_v[i]) for i, module in enumerate(self.v_mlps)])
        return output_k, output_v


def rotate_half(x):
    left, right = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-right, left), dim=-1)


def apply_receiver_rope(model, pre_key, selected_positions):
    device, dtype = pre_key.device, pre_key.dtype
    position_ids = torch.tensor([selected_positions], dtype=torch.long, device=device)
    dummy = torch.empty(
        (1, len(selected_positions), model.config.hidden_size), dtype=dtype, device=device
    )
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    cos = cos[0][None, :, None, :]
    sin = sin[0][None, :, None, :]
    return pre_key * cos + rotate_half(pre_key) * sin


def align_tensor(tensor, target_length, kind):
    if kind == "zero":
        return torch.zeros(
            (tensor.shape[0], target_length, tensor.shape[2], tensor.shape[3]),
            dtype=torch.float16,
        ), target_length
    valid = min(tensor.shape[1], target_length)
    result = tensor[:, :target_length].half()
    if result.shape[1] < target_length:
        result = F.pad(result, (0, 0, 0, 0, 0, target_length - result.shape[1]))
    return result, valid


def memory_content(store, split, sender, sample, family, kind):
    target_length = sample["selected_token_count"]
    source_id = sample["shuffle_id"] if kind == "shuffled" else sample["id"]
    record = store.get(split, sender, source_id)
    key_name = f"{family}_k"
    value_name = f"{family}_v"
    key, valid = align_tensor(record[key_name], target_length, kind)
    value, _ = align_tensor(record[value_name], target_length, kind)
    mask = torch.zeros((1, target_length), dtype=torch.long)
    mask[:, :valid] = 1
    return key, value, mask


def dynamic_cache(model, pre_key, value, positions):
    post_key = apply_receiver_rope(model, pre_key, positions)
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
    pre_key=None,
    value=None,
    prefix_mask=None,
    compact_question_positions=False,
):
    device = require_cuda()
    question = sample["question_token_ids"]
    target = answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
    current = torch.tensor([question + target[:-1]], dtype=torch.long, device=device)
    if compact_question_positions:
        question_positions = list(range(len(question)))
    else:
        question_positions = sample["question_position_ids"]
    answer_positions = list(
        range(question_positions[-1] + 1, question_positions[-1] + len(target))
    )
    position_ids = torch.tensor(
        [question_positions + answer_positions], dtype=torch.long, device=device
    )
    kwargs = {}
    if pre_key is not None:
        cache = dynamic_cache(model, pre_key, value, sample["selected_position_ids"])
        mask = torch.cat(
            [prefix_mask.to(device), torch.ones_like(current, dtype=torch.long)], dim=1
        )
        kwargs = {
            "past_key_values": cache,
            "cache_position": torch.arange(
                pre_key.shape[1], pre_key.shape[1] + current.shape[1], device=device
            ),
        }
    else:
        mask = torch.ones_like(current, dtype=torch.long)
    logits = model(
        current,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=False,
        **kwargs,
    ).logits
    selected = logits[:, len(question) - 1 : len(question) - 1 + len(target)].float()
    gold = torch.tensor(target, dtype=torch.long, device=device)
    return F.cross_entropy(selected.reshape(-1, selected.shape[-1]), gold)


def reconstruction_loss(pred_k, pred_v, target_k, target_v):
    def component(pred, target):
        target_float, pred_float = target.float(), pred.float()
        nmse = (pred_float - target_float).square().mean() / (
            target_float.square().mean() + 1e-6
        )
        cosine = 1 - F.cosine_similarity(
            pred_float.reshape(-1, pred.shape[-1]),
            target_float.reshape(-1, target.shape[-1]),
            dim=-1,
        ).mean()
        return nmse + cosine

    return component(pred_k, target_k) + component(pred_v, target_v)


def load_manifest(cfg, mode):
    return load_json(Path(cfg["work_dir"]) / "artifacts" / mode / "manifest.json")


@torch.no_grad()
def validate_memory_nll(cfg, model, tok, translator, store, samples, split, family, sender):
    model.eval()
    if translator is not None:
        translator.eval()
    values = []
    for sample in samples:
        key, value, mask = memory_content(store, split, sender, sample, family, "correct")
        key, value = key.to(require_cuda()), value.to(require_cuda())
        if translator is not None:
            key, value = translator(key, value)
        values.append(answer_loss(cfg, model, tok, sample, key, value, mask).item())
    return sum(values) / len(values)


def train_sparse_reader(cfg, mode):
    seed_all(cfg["seed"])
    manifest = load_manifest(cfg, mode)
    store = ShardStore(cfg, mode, manifest)
    tok = tokenizer(cfg["model_4b"])
    model = inject_lora(load_model(cfg["model_4b"]), cfg)
    model.train()
    params = lora_parameters(model)
    optimizer = AdamW(params, lr=cfg["sparse_reader_lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    epochs = 1 if mode == "smoke" else cfg["sparse_reader_epochs"]
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "sparse_reader"
    out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for epoch in range(epochs):
        rows = list(manifest["train"])
        random.Random(cfg["seed"] + epoch).shuffle(rows)
        optimizer.zero_grad(set_to_none=True)
        for index, sample in enumerate(rows, 1):
            key, value, mask = memory_content(store, "train", "4b", sample, "native", "correct")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = answer_loss(
                    cfg,
                    model,
                    tok,
                    sample,
                    key.to(require_cuda()),
                    value.to(require_cuda()),
                    mask,
                )
            scaler.scale(loss / cfg["gradient_accumulation"]).backward()
            if index % cfg["gradient_accumulation"] == 0 or index == len(rows):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            history.append({"epoch": epoch + 1, "sample": index, "answer_ce": loss.item()})
        val = validate_memory_nll(
            cfg, model, tok, None, store, manifest["validation"], "validation", "native", "4b"
        )
        if val < best:
            best = val
            torch.save(
                {"lora": lora_state(model), "validation_nll": val, "epoch": epoch + 1},
                out / "best.pt",
            )
        save_json(out / "history.json", history)
        progress(f"{mode}: R1-0 sparse reader epoch {epoch + 1}/{epochs}")
        model.train()
    del model, optimizer
    empty_cuda()
    evaluate_sparse_gate(cfg, mode)


@torch.no_grad()
def evaluate_sparse_gate(cfg, mode):
    seed_all(cfg["seed"])
    manifest = load_manifest(cfg, mode)
    store = ShardStore(cfg, mode, manifest)
    tok = tokenizer(cfg["model_4b"])
    model = inject_lora(load_model(cfg["model_4b"]), cfg).eval()
    load_lora(
        model,
        Path(cfg["work_dir"]) / "artifacts" / mode / "sparse_reader" / "best.pt",
    )
    conditions = {
        "question_only": ("off", None),
        "native_sparse_correct": ("on", "correct"),
        "native_sparse_shuffled": ("on", "shuffled"),
        "native_sparse_zero": ("on", "zero"),
        "native_sparse_no_kv_same_lora": ("on", None),
    }
    rows, summary = [], {}
    for name, (lora_mode, kind) in conditions.items():
        set_lora(model, lora_mode == "on")
        values = []
        for sample in manifest["validation"]:
            if kind is None:
                loss = answer_loss(
                    cfg,
                    model,
                    tok,
                    sample,
                    compact_question_positions=name == "question_only",
                )
            else:
                key, value, mask = memory_content(
                    store, "validation", "4b", sample, "native", kind
                )
                loss = answer_loss(
                    cfg,
                    model,
                    tok,
                    sample,
                    key.to(require_cuda()),
                    value.to(require_cuda()),
                    mask,
                )
            values.append(loss.item())
            rows.append({"condition": name, "sample_id": sample["id"], "gold_answer_nll": loss.item()})
        summary[name] = {"gold_answer_nll": sum(values) / len(values)}
    summary["correct_minus_shuffled_nll_advantage"] = (
        summary["native_sparse_shuffled"]["gold_answer_nll"]
        - summary["native_sparse_correct"]["gold_answer_nll"]
    )
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "sparse_gate"
    save_json(out / "summary.json", summary)
    save_json(out / "per_sample_nll.json", rows)
    del model
    empty_cuda()
    progress(f"{mode}: R1-0 sparse-native gate evaluation completed")


def train_translator_warmup(cfg, mode):
    seed_all(cfg["seed"])
    manifest = load_manifest(cfg, mode)
    store = ShardStore(cfg, mode, manifest)
    translator = CanonicalTranslator(cfg).to(require_cuda()).train()
    optimizer = AdamW(translator.parameters(), lr=cfg["reconstruction_lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    epochs = 1
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "translator_warmup"
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(epochs):
        rows = list(manifest["train"])
        random.Random(cfg["seed"] + epoch).shuffle(rows)
        optimizer.zero_grad(set_to_none=True)
        for index, sample in enumerate(rows, 1):
            record = store.get("train", "4b", sample["id"])
            canonical_k = record["canonical_k"].to(require_cuda())
            canonical_v = record["canonical_v"].to(require_cuda())
            target_k = record["native_k"].to(require_cuda())
            target_v = record["native_v"].to(require_cuda())
            with torch.autocast("cuda", dtype=torch.float16):
                pred_k, pred_v = translator(canonical_k, canonical_v)
                loss = reconstruction_loss(pred_k, pred_v, target_k, target_v)
            scaler.scale(loss / cfg["gradient_accumulation"]).backward()
            if index % cfg["gradient_accumulation"] == 0 or index == len(rows):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            history.append({"epoch": epoch + 1, "sample": index, "reconstruction_loss": loss.item()})
        progress(f"{mode}: R1-1 reconstruction warm-up epoch completed")
    torch.save({"translator": translator.state_dict()}, out / "final.pt")
    save_json(out / "history.json", history)
    del translator, optimizer
    empty_cuda()


def train_self_canonical(cfg, mode):
    seed_all(cfg["seed"])
    manifest = load_manifest(cfg, mode)
    store = ShardStore(cfg, mode, manifest)
    tok = tokenizer(cfg["model_4b"])
    model = inject_lora(load_model(cfg["model_4b"]), cfg)
    load_lora(
        model,
        Path(cfg["work_dir"]) / "artifacts" / mode / "sparse_reader" / "best.pt",
    )
    translator = CanonicalTranslator(cfg).to(require_cuda())
    warmup = torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "translator_warmup" / "final.pt",
        map_location="cpu",
        weights_only=False,
    )
    translator.load_state_dict(warmup["translator"])
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "self_canonical"
    out.mkdir(parents=True, exist_ok=True)
    history = []

    # Phase B1: short answer-CE segment with LoRA frozen.
    for parameter in lora_parameters(model):
        parameter.requires_grad_(False)
    translator.train()
    model.train()
    optimizer = AdamW(translator.parameters(), lr=cfg["functional_translator_lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    warm_rows = manifest["train"][: min(cfg["functional_translator_only_steps"], len(manifest["train"]))]
    for index, sample in enumerate(warm_rows, 1):
        record = store.get("train", "4b", sample["id"])
        with torch.autocast("cuda", dtype=torch.float16):
            key, value = translator(
                record["canonical_k"].to(require_cuda()),
                record["canonical_v"].to(require_cuda()),
            )
            mask = torch.ones((1, sample["selected_token_count"]), dtype=torch.long)
            loss = answer_loss(cfg, model, tok, sample, key, value, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        history.append({"phase": "translator_only", "sample": index, "answer_ce": loss.item()})
    del optimizer

    # Phase B2: joint functional registration.
    for parameter in lora_parameters(model):
        parameter.requires_grad_(True)
    optimizer = AdamW(
        [
            {"params": translator.parameters(), "lr": cfg["functional_translator_lr"]},
            {"params": lora_parameters(model), "lr": cfg["functional_lora_lr"]},
        ],
        weight_decay=0.0,
    )
    epochs = 1 if mode == "smoke" else cfg["functional_epochs"]
    best = float("inf")
    for epoch in range(epochs):
        rows = list(manifest["train"])
        random.Random(cfg["seed"] + epoch).shuffle(rows)
        optimizer.zero_grad(set_to_none=True)
        for index, sample in enumerate(rows, 1):
            correct = store.get("train", "4b", sample["id"])
            shuffled = store.get("train", "4b", sample["shuffle_id"])
            shuffled_k, valid = align_tensor(
                shuffled["canonical_k"], sample["selected_token_count"], "shuffled"
            )
            shuffled_v, _ = align_tensor(
                shuffled["canonical_v"], sample["selected_token_count"], "shuffled"
            )
            correct_mask = torch.ones((1, sample["selected_token_count"]), dtype=torch.long)
            shuffled_mask = torch.zeros_like(correct_mask)
            shuffled_mask[:, :valid] = 1
            with torch.autocast("cuda", dtype=torch.float16):
                pred_k, pred_v = translator(
                    correct["canonical_k"].to(require_cuda()),
                    correct["canonical_v"].to(require_cuda()),
                )
                answer_ce = answer_loss(
                    cfg, model, tok, sample, pred_k, pred_v, correct_mask
                )
                rec = reconstruction_loss(
                    pred_k,
                    pred_v,
                    correct["native_k"].to(require_cuda()),
                    correct["native_v"].to(require_cuda()),
                )
                wrong_k, wrong_v = translator(
                    shuffled_k.to(require_cuda()), shuffled_v.to(require_cuda())
                )
                shuffled_nll = answer_loss(
                    cfg, model, tok, sample, wrong_k, wrong_v, shuffled_mask
                )
                dependence = F.relu(
                    cfg["dependence_margin"] + answer_ce - shuffled_nll
                )
                loss = (
                    answer_ce
                    + cfg["reconstruction_weight"] * rec
                    + cfg["dependence_weight"] * dependence
                )
            scaler.scale(loss / cfg["gradient_accumulation"]).backward()
            if index % cfg["gradient_accumulation"] == 0 or index == len(rows):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(translator.parameters()) + lora_parameters(model), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            history.append(
                {
                    "phase": "joint",
                    "epoch": epoch + 1,
                    "sample": index,
                    "answer_ce": answer_ce.item(),
                    "reconstruction": rec.item(),
                    "dependence": dependence.item(),
                    "shuffled_nll": shuffled_nll.item(),
                    "total": loss.item(),
                }
            )
        val = validate_memory_nll(
            cfg,
            model,
            tok,
            translator,
            store,
            manifest["validation"],
            "validation",
            "canonical",
            "4b",
        )
        if val < best:
            best = val
            torch.save(
                {
                    "translator": translator.state_dict(),
                    "lora": lora_state(model),
                    "validation_nll": val,
                    "epoch": epoch + 1,
                },
                out / "best.pt",
            )
        save_json(out / "history.json", history)
        progress(f"{mode}: R1-1 functional epoch {epoch + 1}/{epochs}")
        model.train()
        translator.train()
    del model, translator, optimizer
    empty_cuda()


def load_reader_checkpoint(model, translator, path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    translator.load_state_dict(state["translator"])
    parameters = dict(model.named_parameters())
    for name, value in state["lora"].items():
        parameters[name].data.copy_(value.to(parameters[name].device))


def build_condition_memory(cfg, model, translator, store, split, sample, condition):
    if condition["memory"] is None:
        return None, None, None
    sender, family, kind = condition["memory"]
    key, value, mask = memory_content(store, split, sender, sample, family, kind)
    key, value = key.to(require_cuda()), value.to(require_cuda())
    if family == "canonical":
        key, value = translator(key, value)
    return key, value, mask


@torch.no_grad()
def greedy_generate(
    cfg,
    model,
    tok,
    sample,
    pre_key=None,
    value=None,
    prefix_mask=None,
    supporting_text=False,
    compact_positions=False,
):
    device = require_cuda()
    if supporting_text:
        prompt = sample["full_sequence_token_ids"]
        positions = list(range(len(prompt)))
    else:
        prompt = sample["question_token_ids"]
        positions = (
            list(range(len(prompt))) if compact_positions else sample["question_position_ids"]
        )
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    position_ids = torch.tensor([positions], dtype=torch.long, device=device)
    kwargs = {}
    if pre_key is not None:
        prefix_len = pre_key.shape[1]
        cache = dynamic_cache(model, pre_key, value, sample["selected_position_ids"])
        attention_mask = torch.cat(
            [prefix_mask.to(device), torch.ones_like(input_ids, dtype=torch.long)], 1
        )
        kwargs["past_key_values"] = cache
        kwargs["cache_position"] = torch.arange(
            prefix_len, prefix_len + input_ids.shape[1], device=device
        )
    else:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    output = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        **kwargs,
    )
    generated = []
    past = output.past_key_values
    next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    next_position = positions[-1] + 1
    for _ in range(cfg["max_new_tokens"]):
        token = int(next_token.item())
        if token == tok.eos_token_id:
            break
        generated.append(token)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], 1
        )
        compact_cache_position = torch.tensor(
            [past.get_seq_length()], dtype=torch.long, device=device
        )
        output = model(
            next_token,
            attention_mask=attention_mask,
            position_ids=torch.tensor([[next_position]], dtype=torch.long, device=device),
            cache_position=compact_cache_position,
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    return tok.decode(generated, skip_special_tokens=True).strip()


FINAL_CONDITIONS = [
    {"key": "question_only", "lora": "off", "reader": None, "memory": None, "compact": True},
    {"key": "supporting_text", "lora": "off", "reader": None, "memory": None, "text": True},
    {"key": "native_4b_sparse_correct", "lora": "sparse", "reader": None, "memory": ("4b", "native", "correct")},
    {"key": "native_4b_sparse_shuffled", "lora": "sparse", "reader": None, "memory": ("4b", "native", "shuffled")},
    {"key": "native_4b_sparse_zero", "lora": "sparse", "reader": None, "memory": ("4b", "native", "zero")},
    {"key": "self_4b_canonical_correct", "lora": "functional", "reader": "translator", "memory": ("4b", "canonical", "correct")},
    {"key": "self_4b_canonical_shuffled", "lora": "functional", "reader": "translator", "memory": ("4b", "canonical", "shuffled")},
    {"key": "self_4b_canonical_zero", "lora": "functional", "reader": "translator", "memory": ("4b", "canonical", "zero")},
    {"key": "self_reader_no_memory", "lora": "functional", "reader": "translator", "memory": None},
    {"key": "cross_8b_raw_native_correct", "lora": "sparse", "reader": None, "memory": ("8b", "native", "correct")},
    {"key": "cross_8b_canonical_correct", "lora": "functional", "reader": "translator", "memory": ("8b", "canonical", "correct")},
    {"key": "cross_8b_canonical_shuffled", "lora": "functional", "reader": "translator", "memory": ("8b", "canonical", "shuffled")},
    {"key": "cross_8b_canonical_zero", "lora": "functional", "reader": "translator", "memory": ("8b", "canonical", "zero")},
]


@torch.no_grad()
def evaluate_final(cfg, mode):
    seed_all(cfg["seed"])
    manifest = load_manifest(cfg, mode)
    store = ShardStore(cfg, mode, manifest)
    tok = tokenizer(cfg["model_4b"])
    model = inject_lora(load_model(cfg["model_4b"]), cfg).eval()
    translator = CanonicalTranslator(cfg).to(require_cuda()).eval()
    sparse_path = Path(cfg["work_dir"]) / "artifacts" / mode / "sparse_reader" / "best.pt"
    functional_path = Path(cfg["work_dir"]) / "artifacts" / mode / "self_canonical" / "best.pt"
    rows, summary, joined = [], {}, {x["id"]: {"sample_id": x["id"], "question": x["question"], "gold_answer": x["answer"]} for x in manifest["test"]}
    current_lora = None
    for condition in FINAL_CONDITIONS:
        if condition["lora"] != current_lora:
            if condition["lora"] == "sparse":
                load_lora(model, sparse_path)
                set_lora(model, True)
            elif condition["lora"] == "functional":
                load_reader_checkpoint(model, translator, functional_path)
                set_lora(model, True)
            else:
                set_lora(model, False)
            current_lora = condition["lora"]
        values = []
        progress(f"{mode}: evaluating {condition['key']}")
        for sample in manifest["test"]:
            key, value, mask = build_condition_memory(
                cfg, model, translator, store, "test", sample, condition
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
                    supporting_text=condition.get("text", False),
                    compact_positions=condition.get("compact", False),
                )
                if condition.get("text", False):
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
                    gold = torch.tensor(target, device=require_cuda())
                    nll = F.cross_entropy(selected.reshape(-1, selected.shape[-1]), gold).item()
                else:
                    nll = answer_loss(
                        cfg,
                        model,
                        tok,
                        sample,
                        key,
                        value,
                        mask,
                        compact_question_positions=condition.get("compact", False),
                    ).item()
            em = float(normalize_answer(prediction) == normalize_answer(sample["answer"]))
            record = {
                "condition": condition["key"],
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
            rows.append(record)
            values.append(record)
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
    accuracy = lambda key: summary[key]["strict_semantic_accuracy"]
    aq = accuracy("question_only")
    an = accuracy("native_4b_sparse_correct")
    ass = accuracy("self_4b_canonical_correct")
    ac = accuracy("cross_8b_canonical_correct")
    raw8 = accuracy("cross_8b_raw_native_correct")
    summary["core_retention"] = {
        "A_Q": aq,
        "A_N": an,
        "A_S": ass,
        "A_C": ac,
        "A_raw8": raw8,
        "retention_self": (ass - aq) / (an - aq) if an != aq else None,
        "retention_cross": (ac - aq) / (ass - aq) if ass != aq else None,
        "writer_gain": ac - raw8,
        "native_correct_minus_shuffled": an - accuracy("native_4b_sparse_shuffled"),
        "self_correct_minus_shuffled": ass - accuracy("self_4b_canonical_shuffled"),
        "cross_correct_minus_shuffled": ac - accuracy("cross_8b_canonical_shuffled"),
    }
    summary["r0_pairwise_upper_bound_source"] = str(
        Path(cfg["r0_dir"]) / "artifacts" / "formal" / "evaluation" / "summary.json"
    )
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "summary.json", summary)
    with open(out / "per_condition.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / "full_generations.jsonl", "w", encoding="utf-8") as handle:
        for sample in manifest["test"]:
            handle.write(json.dumps(joined[sample["id"]], ensure_ascii=False) + "\n")
    with open(out / "manual_cpw_64.csv", "w", encoding="utf-8-sig", newline="") as handle:
        fields = ["sample_id", "question", "gold_answer", "condition", "prediction", "manual_cpw", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in manifest["test"][:64]:
            for condition in FINAL_CONDITIONS:
                writer.writerow(
                    {
                        "sample_id": sample["id"],
                        "question": sample["question"],
                        "gold_answer": sample["answer"],
                        "condition": condition["key"],
                        "prediction": joined[sample["id"]][condition["key"]],
                        "manual_cpw": "",
                        "notes": "",
                    }
                )
    del model, translator
    empty_cuda()
    progress(f"{mode}: R1 final self/cross evaluation completed")


def cpu_selftest(cfg):
    small = dict(
        cfg,
        num_layers=4,
        selected_layers=[0, 1, 2, 3],
        num_kv_heads=2,
        head_dim=8,
        translator_hidden_dim=16,
    )
    translator = CanonicalTranslator(small)
    key = torch.randn(4, 5, 2, 8)
    value = torch.randn(4, 5, 2, 8)
    output_k, output_v = translator(key, value)
    (output_k.sum() + output_v.sum()).backward()
    assert output_k.shape == (4, 5, 2, 8)
    assert translator.depth_k.grad is not None
    aligned, valid = align_tensor(key.half(), 7, "shuffled")
    assert aligned.shape == (4, 7, 2, 8) and valid == 5
    progress("R1 CPU self-test passed")


def common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    return parser


def cli_prepare():
    parser = common_args()
    parser.add_argument(
        "--action", choices=("manifest", "structure", "selftest", "extract"), required=True
    )
    parser.add_argument("--sender", choices=("4b", "8b"))
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    if args.action == "manifest":
        prepare_manifests(cfg, args.mode)
    elif args.action == "structure":
        structure_check(cfg)
    elif args.action == "selftest":
        cpu_selftest(cfg)
    elif args.action == "extract":
        if not args.sender:
            parser.error("--sender is required for extract")
        extract_sender_assets(cfg, args.mode, args.sender)


def cli_sparse():
    args = common_args().parse_args()
    cfg = load_json(args.config)
    train_sparse_reader(cfg, args.mode)


def cli_warmup():
    args = common_args().parse_args()
    cfg = load_json(args.config)
    train_translator_warmup(cfg, args.mode)


def cli_functional():
    args = common_args().parse_args()
    cfg = load_json(args.config)
    train_self_canonical(cfg, args.mode)


def cli_evaluate():
    args = common_args().parse_args()
    cfg = load_json(args.config)
    evaluate_final(cfg, args.mode)
