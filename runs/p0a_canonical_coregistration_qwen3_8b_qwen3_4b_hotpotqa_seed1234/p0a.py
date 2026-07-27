from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not start model execution")
    return torch.device("cuda")


def empty_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(require_cuda())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def token_hash(ids):
    return hashlib.sha256(torch.as_tensor(ids, dtype=torch.int64).numpy().tobytes()).hexdigest()


def encode_example(tok, row, cfg):
    support = {(str(title), int(index)) for title, index in row["supporting_facts"]}
    context_ids = []
    sentence_spans = []
    for title, sentences in row["context"]:
        context_ids.extend(tok(f"Document: {title}\n", add_special_tokens=False).input_ids)
        for index, sentence in enumerate(sentences):
            prefix = tok(f"Sentence {index}: ", add_special_tokens=False).input_ids
            content = tok(sentence, add_special_tokens=False).input_ids
            newline = tok("\n", add_special_tokens=False).input_ids
            context_ids.extend(prefix)
            start = len(context_ids)
            context_ids.extend(content)
            end = len(context_ids)
            context_ids.extend(newline)
            sentence_spans.append(
                {
                    "title": title,
                    "sentence_index": index,
                    "start": start,
                    "end": end,
                    "gold": (str(title), index) in support,
                }
            )
    question_prefix = tok("\nQuestion: ", add_special_tokens=False).input_ids
    question_ids = tok(row["question"], add_special_tokens=False).input_ids[: cfg["max_question_tokens"]]
    if len(context_ids) > cfg["max_context_tokens"] or not question_ids:
        return None
    if not sentence_spans or not any(x["gold"] and x["end"] > x["start"] for x in sentence_spans):
        return None
    full_ids = context_ids + question_prefix + question_ids
    question_start = len(context_ids) + len(question_prefix)
    return {
        "id": row["_id"],
        "type": row.get("type", "unknown"),
        "question": row["question"],
        "context_ids": context_ids,
        "full_ids": full_ids,
        "context_length": len(context_ids),
        "question_start": question_start,
        "question_end": len(full_ids),
        "sentence_spans": sentence_spans,
        "gold_token_count": sum(x["end"] - x["start"] for x in sentence_spans if x["gold"]),
    }


def balanced_examples(rows, tok, cfg, count, rng):
    groups = {"bridge": [], "comparison": []}
    shuffled = list(rows)
    rng.shuffle(shuffled)
    target_each = count // 2
    for row in shuffled:
        kind = row.get("type")
        if kind not in groups or len(groups[kind]) >= target_each:
            continue
        encoded = encode_example(tok, row, cfg)
        if encoded is not None:
            groups[kind].append(encoded)
        if all(len(v) >= target_each for v in groups.values()):
            break
    if sum(map(len, groups.values())) != count:
        raise RuntimeError(f"Could not build balanced {count}-example HotpotQA split")
    result = groups["bridge"] + groups["comparison"]
    rng.shuffle(result)
    return result


def prepare_manifest(cfg):
    path = Path(cfg["work_dir"]) / "manifests" / "dataset.json"
    if path.exists():
        return load_json(path)
    tok_a = load_tokenizer(cfg["model_a"])
    tok_b = load_tokenizer(cfg["model_b"])
    train_raw = json.load(open(cfg["hotpot_train"], encoding="utf-8"))
    val_raw = json.load(open(cfg["hotpot_validation"], encoding="utf-8"))
    rng = random.Random(cfg["seed"])
    train = balanced_examples(train_raw, tok_a, cfg, cfg["train_size"], rng)
    validation = balanced_examples(val_raw, tok_a, cfg, cfg["validation_size"], rng)
    for sample in train + validation:
        a = tok_a.convert_ids_to_tokens(sample["full_ids"])
        b = tok_b.convert_ids_to_tokens(sample["full_ids"])
        if a != b:
            raise RuntimeError(f"Tokenizer vocabulary mismatch for {sample['id']}")
    manifest = {
        "seed": cfg["seed"],
        "selected_layers": cfg["selected_layers"],
        "train": train,
        "validation": validation,
    }
    save_json(path, manifest)
    progress("Dataset manifest prepared")
    return manifest


class NativeHook:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.selected = set(cfg["selected_layers"])
        self.values = {}
        self.handles = []
        for index, block in enumerate(model.model.layers):
            if index in self.selected:
                self.handles.append(
                    block.self_attn.register_forward_pre_hook(self._make(index), with_kwargs=True)
                )

    def _make(self, layer):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            b, t, _ = hidden.shape
            kh = self.cfg["num_kv_heads"]
            qh = self.cfg["num_query_heads"]
            d = self.cfg["head_dim"]
            k = module.k_norm(module.k_proj(hidden).view(b, t, kh, d))[0].detach().cpu()
            v = module.v_proj(hidden).view(b, t, kh, d)[0].detach().cpu()
            q = module.q_norm(module.q_proj(hidden).view(b, t, qh, d))[0].detach().cpu()
            q = q.view(t, kh, qh // kh, d).mean(dim=2)
            self.values[layer] = (k, v, q)
        return hook

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def capture(model, hook, ids):
    hook.clear()
    device = require_cuda()
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(tensor)
    positions = torch.arange(tensor.shape[1], device=device).unsqueeze(0)
    model(input_ids=tensor, attention_mask=mask, position_ids=positions, use_cache=False)
    return {layer: hook.values[layer] for layer in sorted(hook.values)}


@torch.no_grad()
def context_invariance_sanity(cfg):
    manifest = prepare_manifest(cfg)
    sample = manifest["train"][0]
    report = {}
    for role, path in (("a", cfg["model_a"]), ("b", cfg["model_b"])):
        progress(f"Context invariance sanity ({role}) started")
        model = load_model(path)
        hook = NativeHook(model, cfg)
        alone = capture(model, hook, sample["context_ids"])
        full = capture(model, hook, sample["full_ids"])
        k_mse, v_mse, k_cos, v_cos, k_relative_rmse, v_relative_rmse = [], [], [], [], [], []
        n = sample["context_length"]
        for layer in cfg["selected_layers"]:
            ak, av, _ = alone[layer]
            fk, fv, _ = full[layer]
            fk, fv = fk[:n], fv[:n]
            current_k_mse = F.mse_loss(ak.float(), fk.float()).item()
            current_v_mse = F.mse_loss(av.float(), fv.float()).item()
            k_mse.append(current_k_mse)
            v_mse.append(current_v_mse)
            k_relative_rmse.append(
                math.sqrt(current_k_mse / max(ak.float().square().mean().item(), 1e-12))
            )
            v_relative_rmse.append(
                math.sqrt(current_v_mse / max(av.float().square().mean().item(), 1e-12))
            )
            k_cos.append(F.cosine_similarity(ak.float().reshape(-1, 128), fk.float().reshape(-1, 128), dim=-1).mean().item())
            v_cos.append(F.cosine_similarity(av.float().reshape(-1, 128), fv.float().reshape(-1, 128), dim=-1).mean().item())
        role_report = {
            "k_mse_max": max(k_mse),
            "v_mse_max": max(v_mse),
            "k_cosine_min": min(k_cos),
            "v_cosine_min": min(v_cos),
            "k_relative_rmse_max": max(k_relative_rmse),
            "v_relative_rmse_max": max(v_relative_rmse),
        }
        role_report["passed"] = (
            role_report["k_cosine_min"] >= cfg["sanity_min_cosine"]
            and role_report["v_cosine_min"] >= cfg["sanity_min_cosine"]
            and role_report["k_relative_rmse_max"] <= cfg["sanity_max_relative_rmse"]
            and role_report["v_relative_rmse_max"] <= cfg["sanity_max_relative_rmse"]
        )
        report[role] = role_report
        hook.close()
        del model
        empty_cuda()
        if not role_report["passed"]:
            save_json(Path(cfg["work_dir"]) / "metrics" / "context_invariance.json", report)
            raise RuntimeError(f"Context invariance sanity failed for sender {role}")
        progress(f"Context invariance sanity ({role}) passed")
    report["passed"] = True
    save_json(Path(cfg["work_dir"]) / "metrics" / "context_invariance.json", report)


@torch.no_grad()
def extract_sender(cfg, role, limit=None):
    assert role in {"a", "b"}
    manifest = prepare_manifest(cfg)
    path = cfg["model_a"] if role == "a" else cfg["model_b"]
    tok = load_tokenizer(path)
    other = load_tokenizer(cfg["model_b"] if role == "a" else cfg["model_a"])
    model = load_model(path)
    hook = NativeHook(model, cfg)
    root = Path(cfg["work_dir"]) / "cache" / role
    samples = []
    for split in ("train", "validation"):
        chosen = manifest[split][:limit] if limit else manifest[split]
        samples.extend((split, x) for x in chosen)
    progress(f"Native extraction {role}: {len(samples)} samples")
    for index, (split, sample) in enumerate(samples, 1):
        out = root / split / f"{sample['id']}.pt"
        if out.exists():
            continue
        ids = sample["full_ids"]
        if tok.convert_ids_to_tokens(ids) != other.convert_ids_to_tokens(ids):
            raise RuntimeError(f"Tokenizer mismatch for {sample['id']}")
        values = capture(model, hook, ids)
        n = sample["context_length"]
        qs, qe = sample["question_start"], sample["question_end"]
        k = torch.stack([values[layer][0][:n] for layer in cfg["selected_layers"]]).half()
        v = torch.stack([values[layer][1][:n] for layer in cfg["selected_layers"]]).half()
        q = torch.stack([values[layer][2][qs:qe] for layer in cfg["selected_layers"]]).half()
        if k.shape[2:] != (cfg["num_kv_heads"], cfg["head_dim"]) or q.shape[2:] != (
            cfg["num_kv_heads"],
            cfg["head_dim"],
        ):
            raise RuntimeError(f"Unexpected native layout for {sample['id']}")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "id": sample["id"],
                "token_hash": token_hash(ids),
                "context_length": n,
                "question_length": qe - qs,
                "k": k,
                "v": v,
                "q": q,
            },
            out,
        )
        if index % 4 == 0 or index == len(samples):
            progress(f"Native extraction {role}: {index}/{len(samples)}")
    hook.close()
    del model
    empty_cuda()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


class ResidualAdapter(nn.Module):
    def __init__(self, layers, heads, dim, hidden):
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.layer_embedding = nn.Parameter(torch.zeros(layers, dim))
        self.head_embedding = nn.Parameter(torch.zeros(heads, dim))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.output_norm = RMSNorm(dim)
        nn.init.normal_(self.layer_embedding, std=0.02)
        nn.init.normal_(self.head_embedding, std=0.02)

    def forward(self, x):
        # x is layer × token × head × dim.
        z = self.input_norm(x)
        z = z + self.layer_embedding[:, None, None, :] + self.head_embedding[None, None, :, :]
        y = x + self.gamma.to(x.dtype) * self.fc2(F.gelu(self.fc1(z)))
        return self.output_norm(y)


class SenderAdapters(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        args = (len(cfg["selected_layers"]), cfg["num_kv_heads"], cfg["head_dim"], cfg["adapter_hidden_dim"])
        self.k = ResidualAdapter(*args)
        self.v = ResidualAdapter(*args)
        self.q = ResidualAdapter(*args)


class CanonicalSystem(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.a = SenderAdapters(cfg)
        self.b = SenderAdapters(cfg)

    def transform(self, role, item):
        module = self.a if role == "a" else self.b
        return module.k(item["k"]), module.v(item["v"]), module.q(item["q"])


class CacheDataset(Dataset):
    def __init__(self, cfg, split, limit=None):
        self.cfg = cfg
        manifest = prepare_manifest(cfg)
        self.samples = manifest[split][:limit] if limit else manifest[split]
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        root = Path(self.cfg["work_dir"]) / "cache"
        a = torch.load(root / "a" / self.split / f"{sample['id']}.pt", map_location="cpu", weights_only=True)
        b = torch.load(root / "b" / self.split / f"{sample['id']}.pt", map_location="cpu", weights_only=True)
        if a["id"] != b["id"] or a["token_hash"] != b["token_hash"]:
            raise RuntimeError(f"Cross-sender cache mismatch for {sample['id']}")
        return {"sample": sample, "a": a, "b": b}


def collate_list(items):
    return items


def to_device(item, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}


def retrieval_scores(q, k):
    logits = torch.einsum("luhd,lthd->luht", q, k) / math.sqrt(q.shape[-1])
    return logits.mean(dim=(0, 1, 2))


def multi_positive_loss(scores, gold_mask):
    if not bool(gold_mask.any()):
        raise RuntimeError("A sample has no gold supporting token")
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[gold_mask], dim=0)


def readout(q, k, v):
    logits = torch.einsum("luhd,lthd->luht", q, k) / math.sqrt(q.shape[-1])
    attention = logits.softmax(dim=-1)
    result = torch.einsum("luht,lthd->luhd", attention, v)
    return F.normalize(result.float().mean(dim=(0, 1, 2)), dim=-1)


def pooled_q(q):
    return F.normalize(q.float().mean(dim=(0, 1, 2)), dim=-1)


def support_vectors(k, sentence_spans):
    result = []
    for span in sentence_spans:
        if span["gold"] and span["end"] > span["start"]:
            result.append(F.normalize(k[:, span["start"] : span["end"]].float().mean(dim=(0, 1, 2)), dim=-1))
    return result


def symmetric_infonce(x, y, temperature):
    if len(x) < 2:
        return x[0].new_zeros(()) if x else torch.tensor(0.0, device=require_cuda())
    x = F.normalize(torch.stack(x), dim=-1)
    y = F.normalize(torch.stack(y), dim=-1)
    labels = torch.arange(x.shape[0], device=x.device)
    logits = x @ y.T / temperature
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def gold_mask(sample, length, device):
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    for span in sample["sentence_spans"]:
        if span["gold"]:
            mask[span["start"] : span["end"]] = True
    return mask


def batch_loss(system, batch, cfg, mode):
    rets = {"aa": [], "ab": [], "ba": [], "bb": []}
    q_a, q_b, k_a, k_b = [], [], [], []
    r_aa, r_ab, r_ba, r_bb = [], [], [], []
    for row in batch:
        a = to_device(row["a"], require_cuda())
        b = to_device(row["b"], require_cuda())
        ka, va, qa = system.transform("a", a)
        kb, vb, qb = system.transform("b", b)
        mask = gold_mask(row["sample"], ka.shape[1], ka.device)
        grids = {
            "aa": retrieval_scores(qa, ka),
            "ab": retrieval_scores(qa, kb),
            "ba": retrieval_scores(qb, ka),
            "bb": retrieval_scores(qb, kb),
        }
        for name, scores in grids.items():
            rets[name].append(multi_positive_loss(scores, mask))
        if mode == "shared":
            q_a.append(pooled_q(qa)); q_b.append(pooled_q(qb))
            k_a.extend(support_vectors(ka, row["sample"]["sentence_spans"]))
            k_b.extend(support_vectors(kb, row["sample"]["sentence_spans"]))
            r_aa.append(readout(qa, ka, va)); r_ab.append(readout(qa, kb, vb))
            r_ba.append(readout(qb, ka, va)); r_bb.append(readout(qb, kb, vb))
    means = {k: torch.stack(v).mean() for k, v in rets.items()}
    if mode == "private":
        return 0.5 * (means["aa"] + means["bb"])
    retrieval = 0.5 * (means["aa"] + means["bb"]) + means["ab"] + means["ba"]
    lq = symmetric_infonce(q_a, q_b, cfg["temperature"])
    lk = symmetric_infonce(k_a, k_b, cfg["temperature"])
    lv = symmetric_infonce(r_aa, r_ab, cfg["temperature"]) + symmetric_infonce(r_bb, r_ba, cfg["temperature"])
    return retrieval + cfg["lambda_q"] * lq + cfg["lambda_k"] * lk + cfg["lambda_v"] * lv


def save_checkpoint(path, system, extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": system.state_dict(), "extra": extra}, path)


def train_model(cfg, mode, smoke=False):
    device = require_cuda()
    seed_all(cfg["seed"] + (1 if mode == "shared" else 0))
    limit = cfg["smoke_train_size"] if smoke else None
    dataset = CacheDataset(cfg, "train", limit)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_list,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    system = CanonicalSystem(cfg).to(device)
    optimizer = torch.optim.AdamW(system.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    epochs = 1 if smoke else cfg["formal_epochs"]
    max_steps = cfg["smoke_steps"] if smoke else None
    tag = f"{mode}_{'smoke' if smoke else 'formal'}"
    history = []
    step = 0
    progress(f"Training {tag} started")
    for epoch in range(1, epochs + 1):
        for batch in loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = batch_loss(system, batch, cfg, mode)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {tag}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(system.parameters(), cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            history.append({"step": step, "epoch": epoch, "loss": loss.detach().float().item()})
            progress(f"Training {tag}: step {step}")
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
    root = Path(cfg["work_dir"])
    save_checkpoint(root / "checkpoints" / f"{tag}.pt", system, {"mode": mode, "smoke": smoke, "steps": step})
    save_json(root / "metrics" / f"{tag}_history.json", history)
    del system, optimizer
    empty_cuda()
    progress(f"Training {tag} completed")


def average_precision(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    hits, total = 0, 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            hits += 1
            total += hits / rank
    return total / positives


def sample_metrics(scores, sample):
    sentence_scores, labels = [], []
    for span in sample["sentence_spans"]:
        if span["end"] <= span["start"]:
            continue
        sentence_scores.append(scores[span["start"] : span["end"]].float().mean().item())
        labels.append(bool(span["gold"]))
    order = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)
    gold_count = max(1, sum(labels))
    result = {}
    for k in (1, 5):
        hits = sum(labels[i] for i in order[:k])
        result[f"support_recall_at_{k}"] = hits / gold_count
        result[f"support_hit_at_{k}"] = float(hits > 0)
    first = next((rank for rank, i in enumerate(order, 1) if labels[i]), None)
    result["mrr"] = 1.0 / first if first else 0.0
    result["support_auprc"] = average_precision(sentence_scores, labels)
    return result


def aggregate(rows):
    if not rows:
        return {}
    return {key: sum(x[key] for x in rows) / len(rows) for key in rows[0]}


@torch.no_grad()
def evaluate_system(cfg, checkpoint, mode, split="validation", limit=None):
    device = require_cuda()
    dataset = CacheDataset(cfg, split, limit)
    system = None
    if checkpoint:
        system = CanonicalSystem(cfg).to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        system.load_state_dict(state["model"])
        system.eval()
    grid_rows = {x: [] for x in ("aa", "ab", "ba", "bb")}
    shuffled_rows = {"ab": [], "ba": []}
    cached = [dataset[i] for i in range(len(dataset))]
    progress(f"Evaluation {mode}: {len(cached)} samples")
    for index, row in enumerate(cached):
        a = to_device(row["a"], device)
        b = to_device(row["b"], device)
        with torch.autocast("cuda", dtype=torch.float16):
            if system is None:
                ka, qa, kb, qb = a["k"], a["q"], b["k"], b["q"]
            else:
                ka, _, qa = system.transform("a", a)
                kb, _, qb = system.transform("b", b)
            scores = {
                "aa": retrieval_scores(qa, ka),
                "ab": retrieval_scores(qa, kb),
                "ba": retrieval_scores(qb, ka),
                "bb": retrieval_scores(qb, kb),
            }
            for name in scores:
                grid_rows[name].append(sample_metrics(scores[name], row["sample"]))
            other = cached[(index + 1) % len(cached)]
            oa = to_device(other["a"], device)
            ob = to_device(other["b"], device)
            if system is None:
                shuffled_ab = retrieval_scores(qa, ob["k"])
                shuffled_ba = retrieval_scores(qb, oa["k"])
            else:
                okb, _, _ = system.transform("b", ob)
                oka, _, _ = system.transform("a", oa)
                shuffled_ab = retrieval_scores(qa, okb)
                shuffled_ba = retrieval_scores(qb, oka)
            shuffled_rows["ab"].append(sample_metrics(shuffled_ab, other["sample"]))
            shuffled_rows["ba"].append(sample_metrics(shuffled_ba, other["sample"]))
        if (index + 1) % 8 == 0 or index + 1 == len(cached):
            progress(f"Evaluation {mode}: {index + 1}/{len(cached)}")
    result = {
        "mode": mode,
        "grid": {name: aggregate(rows) for name, rows in grid_rows.items()},
        "shuffled": {name: aggregate(rows) for name, rows in shuffled_rows.items()},
    }
    del system
    empty_cuda()
    return result


def final_evaluation(cfg):
    root = Path(cfg["work_dir"])
    results = {
        "raw_native": evaluate_system(cfg, None, "raw_native"),
        "private_writer": evaluate_system(cfg, root / "checkpoints" / "private_formal.pt", "private_writer"),
        "shared_canonical": evaluate_system(cfg, root / "checkpoints" / "shared_formal.pt", "shared_canonical"),
    }
    save_json(root / "metrics" / "final_evaluation.json", results)
    progress("Final evaluation completed")


def cpu_self_test(cfg):
    seed_all(cfg["seed"])
    small = dict(cfg, selected_layers=[0, 1], num_kv_heads=2, head_dim=8, adapter_hidden_dim=16)
    system = CanonicalSystem(small)
    x = {"k": torch.randn(2, 7, 2, 8), "v": torch.randn(2, 7, 2, 8), "q": torch.randn(2, 3, 2, 8)}
    k, v, q = system.transform("a", x)
    assert k.shape == x["k"].shape and v.shape == x["v"].shape and q.shape == x["q"].shape
    scores = retrieval_scores(q, k)
    assert scores.shape == (7,) and torch.isfinite(scores).all()
    _ = readout(q, k, v)
    progress("CPU structural self-test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["prepare", "cpu_self_test", "sanity", "extract_a", "extract_b"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.command == "prepare":
        prepare_manifest(cfg)
    elif args.command == "cpu_self_test":
        cpu_self_test(cfg)
    elif args.command == "sanity":
        context_invariance_sanity(cfg)
    elif args.command == "extract_a":
        extract_sender(cfg, "a", args.limit)
    elif args.command == "extract_b":
        extract_sender(cfg, "b", args.limit)


if __name__ == "__main__":
    main()
