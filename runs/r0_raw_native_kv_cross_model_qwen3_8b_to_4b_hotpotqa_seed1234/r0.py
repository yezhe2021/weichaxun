#!/home/yezhe/data/miniconda3/envs/attnkv/bin/python
"""R0: raw native KV cross-model usability, with no canonical/translator modules."""
import argparse
import csv
import json
import math
import os
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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; R0 requires a working GPU")
    return torch.device("cuda")


def empty_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gold_evidence(row):
    contexts = {title: sentences for title, sentences in row["context"]}
    lines = []
    for title, sentence_id in row["supporting_facts"]:
        sentences = contexts.get(title, [])
        if 0 <= sentence_id < len(sentences):
            text = sentences[sentence_id].strip()
            if text:
                lines.append(f"{title}: {text}")
    if not lines:
        raise ValueError(f"No supporting sentence found for {row['_id']}")
    return "\n".join(lines)


def prefix_text(sample):
    return f"Evidence:\n{sample['evidence']}\n\nQuestion:\n"


def suffix_text(sample):
    return f"{sample['question']}\n\nAnswer:"


def full_prompt(sample):
    return prefix_text(sample) + suffix_text(sample)


def convert_row(row):
    return {
        "id": row["_id"],
        "type": row.get("type"),
        "level": row.get("level"),
        "question": row["question"].strip(),
        "answer": str(row["answer"]).strip(),
        "evidence": gold_evidence(row),
        "supporting_facts": row["supporting_facts"],
    }


def choose_rows(rows, size, seed, excluded=None):
    excluded = set(excluded or [])
    candidates = [row for row in rows if row["_id"] not in excluded]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = []
    for row in candidates:
        try:
            selected.append(convert_row(row))
        except (KeyError, ValueError, IndexError):
            continue
        if len(selected) == size:
            break
    if len(selected) != size:
        raise RuntimeError(f"Requested {size} usable rows, found {len(selected)}")
    return selected


def prepare_data(cfg, mode):
    sizes = cfg[f"{mode}_sizes"]
    train_rows = load_json(cfg["hotpot_train"])
    dev_rows = load_json(cfg["hotpot_dev"])
    train = choose_rows(train_rows, sizes["train"], cfg["seed"])
    validation = choose_rows(dev_rows, sizes["validation"], cfg["seed"] + 1)
    test = choose_rows(
        dev_rows, sizes["test"], cfg["seed"] + 2, {x["id"] for x in validation}
    )
    tok = AutoTokenizer.from_pretrained(cfg["model_4b"], local_files_only=True)
    all_splits = {"train": train, "validation": validation, "test": test}
    for split, samples in all_splits.items():
        for sample in samples:
            sample["prefix_tokens"] = len(
                tok(prefix_text(sample), add_special_tokens=False).input_ids
            )
        assign_shuffles(samples, cfg)
    out = Path(cfg["work_dir"]) / "artifacts" / mode
    save_json(out / "splits.json", all_splits)
    save_json(
        out / "sampling_report.json",
        {
            "seed": cfg["seed"],
            "policy": "unfiltered random sample; train from official train, validation/test disjoint from official dev",
            "counts": {k: len(v) for k, v in all_splits.items()},
            "type_counts": {
                k: dict(Counter(x.get("type") for x in v)) for k, v in all_splits.items()
            },
        },
    )
    progress(f"{mode}: prepared {sizes['train']}/{sizes['validation']}/{sizes['test']}")


def length_bucket(length, bounds):
    for idx, bound in enumerate(bounds):
        if length <= bound:
            return idx
    return len(bounds)


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def assign_shuffles(samples, cfg):
    bounds = cfg["length_buckets"]
    for current in samples:
        valid = [
            other
            for other in samples
            if other["id"] != current["id"]
            and normalize_answer(other["answer"]) != normalize_answer(current["answer"])
        ]
        if not valid:
            raise RuntimeError("Cannot construct answer-different shuffled control")
        bucket = length_bucket(current["prefix_tokens"], bounds)
        same_bucket = [
            x for x in valid if length_bucket(x["prefix_tokens"], bounds) == bucket
        ]
        pool = same_bucket or valid
        chosen = min(
            pool,
            key=lambda x: (abs(x["prefix_tokens"] - current["prefix_tokens"]), x["id"]),
        )
        current["shuffle_id"] = chosen["id"]


def tokenizer_signature(tok):
    return {
        "class": tok.__class__.__name__,
        "vocab_size": tok.vocab_size,
        "len": len(tok),
        "bos": tok.bos_token_id,
        "eos": tok.eos_token_id,
        "pad": tok.pad_token_id,
        "special_tokens": tok.special_tokens_map,
    }


def structure_check(cfg):
    reports = {}
    for name, path in (("4b", cfg["model_4b"]), ("8b", cfg["model_8b"])):
        conf = AutoConfig.from_pretrained(path, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        reports[name] = {
            "path": path,
            "num_hidden_layers": conf.num_hidden_layers,
            "num_key_value_heads": conf.num_key_value_heads,
            "head_dim": getattr(conf, "head_dim", conf.hidden_size // conf.num_attention_heads),
            "rope_theta": getattr(conf, "rope_theta", None),
            "rope_scaling": getattr(conf, "rope_scaling", None),
            "tokenizer": tokenizer_signature(tok),
        }
    for name in ("4b", "8b"):
        row = reports[name]
        expected = (cfg["num_layers"], cfg["num_kv_heads"], cfg["head_dim"])
        actual = (row["num_hidden_layers"], row["num_key_value_heads"], row["head_dim"])
        if actual != expected:
            raise RuntimeError(f"{name} KV structure {actual} != expected {expected}")
    for field in ("rope_theta", "rope_scaling", "tokenizer"):
        if reports["4b"][field] != reports["8b"][field]:
            raise RuntimeError(f"4B/8B incompatible {field}")
    save_json(Path(cfg["work_dir"]) / "artifacts" / "structure_check.json", reports)
    progress("R0 structure check passed")


def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(require_cuda()).eval()


def cache_to_cpu(cache):
    return [
        (layer.keys.detach().cpu().half().contiguous(), layer.values.detach().cpu().half().contiguous())
        for layer in cache.layers
    ]


def dynamic_cache(items, config, device):
    data = [(k.to(device), v.to(device)) for k, v in items]
    return DynamicCache(ddp_cache_data=data, config=config)


def target_ids(tok, answer, max_tokens):
    ids = tok(" " + answer, add_special_tokens=False).input_ids[: max_tokens - 1]
    ids.append(tok.eos_token_id)
    return ids


def teacher_inputs(tok, sample, max_tokens, device):
    prompt = tok(suffix_text(sample), add_special_tokens=False).input_ids
    target = target_ids(tok, sample["answer"], max_tokens)
    current = torch.tensor([prompt + target[:-1]], dtype=torch.long, device=device)
    return current, target, len(prompt)


def answer_logits(logits, prompt_len, target_len):
    return logits[:, prompt_len - 1 : prompt_len - 1 + target_len].float()


@torch.no_grad()
def replay_gate(cfg, mode):
    device = require_cuda()
    samples = load_json(Path(cfg["work_dir"]) / "artifacts" / mode / "splits.json")
    sample = samples["validation"][0]
    tok = AutoTokenizer.from_pretrained(cfg["model_4b"], local_files_only=True)
    model = load_model(cfg["model_4b"])
    prefix = tok(prefix_text(sample), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    current, target, prompt_len = teacher_inputs(tok, sample, cfg["max_answer_tokens"], device)
    full = torch.cat([prefix, current], dim=1)
    full_logits = model(full, attention_mask=torch.ones_like(full), use_cache=False).logits
    full_selected = full_logits[
        :, prefix.shape[1] + prompt_len - 1 : prefix.shape[1] + prompt_len - 1 + len(target)
    ].float()
    prefill = model(prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
    cpu = cache_to_cpu(prefill.past_key_values)
    mask = torch.ones((1, prefix.shape[1] + current.shape[1]), device=device, dtype=torch.long)
    cached_logits = model(
        current,
        attention_mask=mask,
        past_key_values=dynamic_cache(cpu, model.config, device),
        use_cache=False,
    ).logits
    cached_selected = answer_logits(cached_logits, prompt_len, len(target))
    gold = torch.tensor(target, dtype=torch.long, device=device)
    full_nll = F.cross_entropy(full_selected.reshape(-1, full_selected.shape[-1]), gold)
    cached_nll = F.cross_entropy(cached_selected.reshape(-1, cached_selected.shape[-1]), gold)
    top1 = (full_selected.argmax(-1) == cached_selected.argmax(-1)).float().mean().item()
    p = F.log_softmax(full_selected, -1)
    q = F.log_softmax(cached_selected, -1)
    kl = F.kl_div(q, p.exp(), reduction="batchmean").item() / len(target)

    prompt_ids = torch.tensor(
        [tok(full_prompt(sample), add_special_tokens=False).input_ids], device=device
    )
    with torch.autocast("cuda", dtype=torch.float16):
        normal = model.generate(
            prompt_ids, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        suffix = torch.tensor(
            [tok(suffix_text(sample), add_special_tokens=False).input_ids], device=device
        )
        cache_gen = dynamic_cache(cpu, model.config, device)
        gen_mask = torch.ones((1, prefix.shape[1] + suffix.shape[1]), device=device, dtype=torch.long)
        cached = model.generate(
            suffix, attention_mask=gen_mask, past_key_values=cache_gen,
            max_new_tokens=cfg["max_new_tokens"], do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    normal_new = normal[0, prompt_ids.shape[1]:].cpu().tolist()
    cached_new = cached[0, suffix.shape[1]:].cpu().tolist()
    report = {
        "sample_id": sample["id"],
        "top1_agreement": top1,
        "full_nll": full_nll.item(),
        "cached_nll": cached_nll.item(),
        "absolute_nll_delta": abs(full_nll.item() - cached_nll.item()),
        "mean_token_kl": kl,
        "greedy_tokens_identical": normal_new == cached_new,
    }
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "r0_0_replay.json", report)
    if (
        top1 < cfg["replay_top1_min"]
        or report["absolute_nll_delta"] >= cfg["replay_nll_delta_max"]
        or kl >= cfg["replay_kl_max"]
        or not report["greedy_tokens_identical"]
    ):
        raise RuntimeError(f"R0-0 replay hard gate failed: {report}")
    del model, prefill
    empty_cuda()
    progress(f"{mode}: R0-0 cache replay gate passed")


@torch.no_grad()
def extract_caches(cfg, mode, sender):
    device = require_cuda()
    path = cfg[f"model_{sender}"]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = load_model(path)
    splits = load_json(Path(cfg["work_dir"]) / "artifacts" / mode / "splits.json")
    root = Path(cfg["work_dir"]) / "artifacts" / mode / "cache" / sender
    for split, samples in splits.items():
        (root / split).mkdir(parents=True, exist_ok=True)
        for idx, sample in enumerate(samples, 1):
            dest = root / split / f"{sample['id']}.pt"
            if dest.exists():
                continue
            ids = tok(
                prefix_text(sample), return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            result = model(ids, attention_mask=torch.ones_like(ids), use_cache=True)
            items = cache_to_cpu(result.past_key_values)
            if len(items) != cfg["num_layers"]:
                raise RuntimeError(f"{sample['id']}: extracted {len(items)} layers")
            torch.save(
                {
                    "sample_id": sample["id"],
                    "sender": sender,
                    "token_ids": ids.cpu(),
                    "cache_length": ids.shape[1],
                    "attention_mask": torch.ones((1, ids.shape[1]), dtype=torch.long),
                    "evidence": sample["evidence"],
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "layers": items,
                },
                dest,
            )
            if idx % 32 == 0:
                progress(f"{mode}: {sender} cache {split} {idx}/{len(samples)}")
    del model
    empty_cuda()
    progress(f"{mode}: complete 36-layer {sender} native cache extraction")


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        result = self.base(x)
        if self.enabled:
            hidden = F.linear(self.dropout(x), self.lora_A.to(x.dtype))
            result = result + F.linear(hidden, self.lora_B.to(x.dtype)) * self.scale
        return result


def inject_lora(model, cfg):
    for param in model.parameters():
        param.requires_grad_(False)
    device = next(model.parameters()).device
    count = 0
    for layer in model.model.layers:
        for name in ("q_proj", "o_proj"):
            base = getattr(layer.self_attn, name)
            wrapped = LoRALinear(
                base, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"]
            ).to(device)
            setattr(layer.self_attn, name, wrapped)
            count += 1
    if count != cfg["num_layers"] * 2:
        raise RuntimeError(f"Attached LoRA to {count} modules")
    return model


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def set_lora_enabled(model, enabled):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def lora_state(model):
    return {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if "lora_A" in n or "lora_B" in n
    }


def load_lora_state(model, path):
    state = torch.load(path, map_location="cpu", weights_only=False)["lora"]
    current = dict(model.named_parameters())
    for name, value in state.items():
        current[name].data.copy_(value.to(current[name].device))


def cache_path(cfg, mode, sender, split, sample_id):
    return (
        Path(cfg["work_dir"]) / "artifacts" / mode / "cache" / sender / split / f"{sample_id}.pt"
    )


def load_cache_record(cfg, mode, sender, split, sample_id):
    return torch.load(
        cache_path(cfg, mode, sender, split, sample_id), map_location="cpu", weights_only=False
    )


def aligned_items(record, target_length, kind):
    output = []
    if kind == "zero":
        valid = target_length
    else:
        valid = min(record["cache_length"], target_length)
    for key, value in record["layers"]:
        if kind == "zero":
            new_k = torch.zeros(
                (key.shape[0], key.shape[1], target_length, key.shape[3]), dtype=torch.float16
            )
            new_v = torch.zeros_like(new_k)
        else:
            new_k = key[:, :, :target_length].half()
            new_v = value[:, :, :target_length].half()
            if new_k.shape[2] < target_length:
                pad = target_length - new_k.shape[2]
                new_k = F.pad(new_k, (0, 0, 0, pad))
                new_v = F.pad(new_v, (0, 0, 0, pad))
        output.append((new_k, new_v))
    prefix_mask = torch.zeros((1, target_length), dtype=torch.long)
    prefix_mask[:, :valid] = 1
    return output, prefix_mask


def cache_for_condition(cfg, mode, sender, split, sample, kind):
    correct = load_cache_record(cfg, mode, sender, split, sample["id"])
    if kind == "correct":
        record = correct
    elif kind == "shuffled":
        record = load_cache_record(cfg, mode, sender, split, sample["shuffle_id"])
    elif kind == "zero":
        record = correct
    else:
        raise ValueError(kind)
    return aligned_items(record, correct["cache_length"], kind)


def loss_for_sample(cfg, model, tok, mode, sender, split, sample, kind="correct"):
    device = require_cuda()
    items, prefix_mask = cache_for_condition(cfg, mode, sender, split, sample, kind)
    current, target, prompt_len = teacher_inputs(tok, sample, cfg["max_answer_tokens"], device)
    mask = torch.cat(
        [prefix_mask.to(device), torch.ones_like(current, dtype=torch.long)], dim=1
    )
    output = model(
        current,
        attention_mask=mask,
        past_key_values=dynamic_cache(items, model.config, device),
        use_cache=False,
    )
    selected = answer_logits(output.logits, prompt_len, len(target))
    gold = torch.tensor(target, dtype=torch.long, device=device)
    return F.cross_entropy(selected.reshape(-1, selected.shape[-1]), gold)


@torch.no_grad()
def validation_nll(cfg, model, tok, mode, sender, samples):
    model.eval()
    values = []
    for sample in samples:
        with torch.autocast("cuda", dtype=torch.float16):
            values.append(loss_for_sample(cfg, model, tok, mode, sender, "validation", sample).item())
    return sum(values) / len(values)


def train_reader(cfg, mode, sender, reader_name):
    seed_all(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model_4b"], local_files_only=True)
    model = inject_lora(load_model(cfg["model_4b"]), cfg)
    model.train()
    trainable = lora_parameters(model)
    optimizer = AdamW(trainable, lr=cfg["learning_rate"], weight_decay=0.0)
    splits = load_json(Path(cfg["work_dir"]) / "artifacts" / mode / "splits.json")
    epochs = 1 if mode == "smoke" else cfg["epochs"]
    total_updates = max(1, math.ceil(len(splits["train"]) * epochs / cfg["gradient_accumulation"]))
    warmup = max(1, round(total_updates * cfg["warmup_ratio"]))
    scaler = torch.amp.GradScaler("cuda")
    out = Path(cfg["work_dir"]) / "artifacts" / mode / reader_name
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    progress(f"{mode}: training {reader_name} from identical seed-1234 initialization")
    for epoch in range(epochs):
        order = list(splits["train"])
        random.Random(cfg["seed"] + epoch).shuffle(order)
        for idx, sample in enumerate(order, 1):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_for_sample(cfg, model, tok, mode, sender, "train", sample)
                scaled_loss = loss / cfg["gradient_accumulation"]
            scaler.scale(scaled_loss).backward()
            last = idx == len(order)
            if idx % cfg["gradient_accumulation"] == 0 or last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                scale = min(1.0, global_step / warmup)
                for group in optimizer.param_groups:
                    group["lr"] = cfg["learning_rate"] * scale
            history.append({"epoch": epoch + 1, "sample": idx, "answer_ce": loss.item()})
        val = validation_nll(cfg, model, tok, mode, sender, splits["validation"])
        save_json(out / "history.json", history)
        if val < best:
            best = val
            torch.save(
                {"lora": lora_state(model), "epoch": epoch + 1, "validation_nll": val},
                out / "best.pt",
            )
        progress(f"{mode}: {reader_name} epoch {epoch + 1}/{epochs} completed")
        model.train()
    del model, optimizer
    empty_cuda()


def token_f1(prediction, gold):
    pred = normalize_answer(prediction).split()
    truth = normalize_answer(gold).split()
    common = Counter(pred) & Counter(truth)
    overlap = sum(common.values())
    if not pred or not truth:
        return float(pred == truth)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(truth)
    return 2 * precision * recall / (precision + recall)


def loose_correct(prediction, gold):
    pred = normalize_answer(prediction)
    truth = normalize_answer(gold)
    return bool(pred and truth and (truth in pred or pred in truth))


@torch.no_grad()
def generate_condition(cfg, model, tok, mode, sample, condition):
    device = require_cuda()
    set_lora_enabled(model, condition["lora"] != "off")
    if condition["memory"] == "supporting_text":
        ids = tok(full_prompt(sample), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    elif condition["memory"] == "none":
        ids = tok(suffix_text(sample), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    else:
        sender, kind = condition["memory"].split("_", 1)
        items, prefix_mask = cache_for_condition(cfg, mode, sender, "test", sample, kind)
        ids = tok(suffix_text(sample), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        mask = torch.cat([prefix_mask.to(device), torch.ones_like(ids)], 1)
        kwargs = {
            "input_ids": ids,
            "attention_mask": mask,
            "past_key_values": dynamic_cache(items, model.config, device),
        }
    with torch.autocast("cuda", dtype=torch.float16):
        output = model.generate(
            **kwargs, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    prediction = tok.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return prediction


@torch.no_grad()
def nll_condition(cfg, model, tok, mode, sample, condition):
    device = require_cuda()
    set_lora_enabled(model, condition["lora"] != "off")
    target = target_ids(tok, sample["answer"], cfg["max_answer_tokens"])
    if condition["memory"] == "supporting_text":
        prompt = tok(full_prompt(sample), add_special_tokens=False).input_ids
        current = torch.tensor([prompt + target[:-1]], device=device)
        logits = model(current, attention_mask=torch.ones_like(current), use_cache=False).logits
        selected = logits[:, len(prompt)-1:len(prompt)-1+len(target)].float()
    elif condition["memory"] == "none":
        prompt = tok(suffix_text(sample), add_special_tokens=False).input_ids
        current = torch.tensor([prompt + target[:-1]], device=device)
        logits = model(current, attention_mask=torch.ones_like(current), use_cache=False).logits
        selected = logits[:, len(prompt)-1:len(prompt)-1+len(target)].float()
    else:
        sender, kind = condition["memory"].split("_", 1)
        items, prefix_mask = cache_for_condition(cfg, mode, sender, "test", sample, kind)
        current, _, prompt_len = teacher_inputs(tok, sample, cfg["max_answer_tokens"], device)
        mask = torch.cat([prefix_mask.to(device), torch.ones_like(current)], 1)
        logits = model(
            current, attention_mask=mask,
            past_key_values=dynamic_cache(items, model.config, device), use_cache=False
        ).logits
        selected = answer_logits(logits, prompt_len, len(target))
    gold = torch.tensor(target, device=device)
    return F.cross_entropy(selected.reshape(-1, selected.shape[-1]), gold).item()


CONDITIONS = [
    {"key": "A_question_only", "memory": "none", "lora": "off"},
    {"key": "B_supporting_text", "memory": "supporting_text", "lora": "off"},
    {"key": "C_4b_native_lora_off", "memory": "4b_correct", "lora": "off"},
    {"key": "D_4b_correct_self", "memory": "4b_correct", "lora": "self"},
    {"key": "E_4b_shuffled_self", "memory": "4b_shuffled", "lora": "self"},
    {"key": "F_4b_zero_self", "memory": "4b_zero", "lora": "self"},
    {"key": "G_no_kv_self", "memory": "none", "lora": "self"},
    {"key": "H_8b_correct_self", "memory": "8b_correct", "lora": "self"},
    {"key": "I_8b_shuffled_self", "memory": "8b_shuffled", "lora": "self"},
    {"key": "J_8b_zero_self", "memory": "8b_zero", "lora": "self"},
    {"key": "K_8b_correct_pair", "memory": "8b_correct", "lora": "pair"},
    {"key": "L_8b_shuffled_pair", "memory": "8b_shuffled", "lora": "pair"},
    {"key": "M_8b_zero_pair", "memory": "8b_zero", "lora": "pair"},
]


def cpu_selftest():
    module = LoRALinear(nn.Linear(5, 7), 2, 4, 0.05)
    output = module(torch.randn(3, 5))
    output.sum().backward()
    record = {
        "cache_length": 3,
        "layers": [
            (
                torch.randn(1, 8, 3, 128).half(),
                torch.randn(1, 8, 3, 128).half(),
            )
            for _ in range(36)
        ],
    }
    items, mask = aligned_items(record, 5, "shuffled")
    assert output.shape == (3, 7)
    assert module.lora_A.grad is not None and module.lora_B.grad is not None
    assert len(items) == 36 and items[0][0].shape == (1, 8, 5, 128)
    assert mask.tolist() == [[1, 1, 1, 0, 0]]
    progress("CPU self-test passed: LoRA backward and shuffled-cache alignment")


def evaluate(cfg, mode):
    seed_all(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model_4b"], local_files_only=True)
    model = inject_lora(load_model(cfg["model_4b"]), cfg)
    model.eval()
    samples = load_json(Path(cfg["work_dir"]) / "artifacts" / mode / "splits.json")["test"]
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = {}
    for condition in CONDITIONS:
        values = []
        if condition["lora"] in ("self", "pair"):
            checkpoint = (
                Path(cfg["work_dir"]) / "artifacts" / mode /
                ("lora_self" if condition["lora"] == "self" else "lora_pair") / "best.pt"
            )
            load_lora_state(model, checkpoint)
        progress(f"{mode}: evaluating {condition['key']}")
        for sample in samples:
            prediction = generate_condition(cfg, model, tok, mode, sample, condition)
            nll = nll_condition(cfg, model, tok, mode, sample, condition)
            em = float(normalize_answer(prediction) == normalize_answer(sample["answer"]))
            record = {
                "condition": condition["key"], "sample_id": sample["id"],
                "type": sample.get("type"), "question": sample["question"],
                "evidence": sample["evidence"], "gold_answer": sample["answer"],
                "prediction": prediction, "em": em,
                "token_f1": token_f1(prediction, sample["answer"]),
                "strict_semantic_accuracy": em,
                "loose_semantic_accuracy": float(loose_correct(prediction, sample["answer"])),
                "gold_answer_nll": nll,
            }
            rows.append(record)
            values.append(record)
        summaries[condition["key"]] = {
            key: sum(x[key] for x in values) / len(values)
            for key in (
                "em", "token_f1", "strict_semantic_accuracy",
                "loose_semantic_accuracy", "gold_answer_nll"
            )
        }
    def acc(key):
        return summaries[key]["strict_semantic_accuracy"]
    summaries["primary_deltas"] = {
        "self_correct_minus_shuffled": acc("D_4b_correct_self") - acc("E_4b_shuffled_self"),
        "self_correct_minus_question_only": acc("D_4b_correct_self") - acc("A_question_only"),
        "zero_shot_correct_minus_shuffled": acc("H_8b_correct_self") - acc("I_8b_shuffled_self"),
        "pair_correct_minus_shuffled": acc("K_8b_correct_pair") - acc("L_8b_shuffled_pair"),
        "A_self": acc("D_4b_correct_self"),
        "A_zero": acc("H_8b_correct_self"),
        "A_pair": acc("K_8b_correct_pair"),
    }
    save_json(out / "summary.json", summaries)
    with open(out / "per_sample.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manual_samples = samples[:64]
    by_key = {(x["condition"], x["sample_id"]): x for x in rows}
    with open(out / "manual_cpw_64.csv", "w", encoding="utf-8-sig", newline="") as f:
        fields = ["sample_id", "question", "gold_answer", "condition", "prediction", "manual_cpw", "notes"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in manual_samples:
            for condition in CONDITIONS:
                row = by_key[(condition["key"], sample["id"])]
                writer.writerow({
                    "sample_id": sample["id"], "question": sample["question"],
                    "gold_answer": sample["answer"], "condition": condition["key"],
                    "prediction": row["prediction"], "manual_cpw": "", "notes": "",
                })
    del model
    empty_cuda()
    progress(f"{mode}: final evaluation completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "replay", "extract", "train", "evaluate"):
        p = sub.add_parser(name)
        if name != "prepare":
            p.add_argument("--mode", choices=("smoke", "formal"), required=True)
        if name == "extract":
            p.add_argument("--sender", choices=("4b", "8b"), required=True)
        if name == "train":
            p.add_argument("--reader", choices=("self", "pair"), required=True)
    sub.add_parser("structure")
    sub.add_parser("selftest")
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    if args.command == "prepare":
        prepare_data(cfg, "smoke")
        prepare_data(cfg, "formal")
    elif args.command == "structure":
        structure_check(cfg)
    elif args.command == "selftest":
        cpu_selftest()
    elif args.command == "replay":
        replay_gate(cfg, args.mode)
    elif args.command == "extract":
        extract_caches(cfg, args.mode, args.sender)
    elif args.command == "train":
        sender = "4b" if args.reader == "self" else "8b"
        train_reader(cfg, args.mode, sender, f"lora_{args.reader}")
    elif args.command == "evaluate":
        evaluate(cfg, args.mode)


if __name__ == "__main__":
    main()
