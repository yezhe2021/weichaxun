from __future__ import annotations

import argparse
import gc
import importlib.util
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
from transformers import AutoModel, AutoTokenizer


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
        raise RuntimeError("CUDA is unavailable; do not start P0-A2")
    return torch.device("cuda")


def empty_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_p0a(cfg):
    path = Path(cfg["source_p0a_dir"]) / "p0a.py"
    spec = importlib.util.spec_from_file_location("source_p0a", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_config(cfg):
    return load_json(Path(cfg["source_p0a_dir"]) / "config.json")


def manifest(cfg):
    return load_json(Path(cfg["source_p0a_dir"]) / "manifests" / "dataset.json")


def raw_rows_by_id(path):
    return {row["_id"]: row for row in json.load(open(path, encoding="utf-8"))}


def sentence_texts(sample, raw):
    row = raw[sample["id"]]
    lookup = {(str(title), index): text for title, sentences in row["context"] for index, text in enumerate(sentences)}
    result = []
    for span in sample["sentence_spans"]:
        key = (str(span["title"]), int(span["sentence_index"]))
        if key not in lookup:
            raise RuntimeError(f"Missing raw sentence {key} for {sample['id']}")
        result.append(lookup[key])
    return result


@torch.no_grad()
def extract_text_embeddings(cfg, limit=None):
    device = require_cuda()
    data = manifest(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"], local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        cfg["text_encoder"],
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    raw = {
        "train": raw_rows_by_id(cfg["hotpot_train"]),
        "validation": raw_rows_by_id(cfg["hotpot_validation"]),
    }
    root = Path(cfg["work_dir"]) / "cache" / "sentence_embeddings"
    all_samples = []
    for split in ("train", "validation"):
        chosen = data[split][:limit] if limit else data[split]
        all_samples.extend((split, sample) for sample in chosen)
    progress(f"Frozen sentence embedding extraction: {len(all_samples)} samples")
    for index, (split, sample) in enumerate(all_samples, 1):
        out = root / split / f"{sample['id']}.pt"
        if out.exists():
            continue
        texts = sentence_texts(sample, raw[split])
        vectors = []
        for start in range(0, len(texts), cfg["text_batch_size"]):
            batch = tokenizer(
                texts[start : start + cfg["text_batch_size"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg["text_max_tokens"],
            ).to(device)
            hidden = model(**batch, use_cache=False).last_hidden_state.float()
            mask = batch.attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(F.normalize(pooled, dim=-1).half().cpu())
        embeddings = torch.cat(vectors)
        labels = torch.tensor([bool(x["gold"]) for x in sample["sentence_spans"]], dtype=torch.bool)
        if embeddings.shape[0] != labels.shape[0] or not bool(labels.any()):
            raise RuntimeError(f"Sentence embedding alignment failed for {sample['id']}")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"id": sample["id"], "embeddings": embeddings, "labels": labels}, out)
        if index % 4 == 0 or index == len(all_samples):
            progress(f"Frozen sentence embedding extraction: {index}/{len(all_samples)}")
    del model
    empty_cuda()


def load_source_item(cfg, role, split, sample_id):
    return torch.load(
        Path(cfg["source_p0a_dir"]) / "cache" / role / split / f"{sample_id}.pt",
        map_location="cpu",
        weights_only=True,
    )


def align_v(v, target_tokens):
    current = v.shape[1]
    if current == target_tokens:
        return v
    if current > target_tokens:
        return v[:, :target_tokens]
    pad = v.new_zeros((v.shape[0], target_tokens - current, v.shape[2], v.shape[3]))
    return torch.cat([v, pad], dim=1)


def fixed_attention(q, k):
    logits = torch.einsum("luhd,lthd->luht", q, k) / math.sqrt(q.shape[-1])
    return logits.softmax(dim=-1)


def fixed_readout(attention, v):
    result = torch.einsum("luht,lthd->luhd", attention, v)
    return F.normalize(result.float().mean(dim=(0, 1, 2)), dim=-1)


def nearest_wrong_indices(samples):
    lengths = [int(x["context_length"]) for x in samples]
    result = []
    for i, length in enumerate(lengths):
        candidates = [(abs(length - other), j) for j, other in enumerate(lengths) if j != i]
        result.append(min(candidates)[1])
    return result


def load_canonical_system(cfg, mode, p0a, device):
    if mode == "raw_native":
        return None
    source_cfg = source_config(cfg)
    system = p0a.CanonicalSystem(source_cfg).to(device)
    name = "private_formal.pt" if mode == "private_writer" else "shared_formal.pt"
    checkpoint = torch.load(
        Path(cfg["source_p0a_dir"]) / "checkpoints" / name,
        map_location="cpu",
        weights_only=True,
    )
    system.load_state_dict(checkpoint["model"])
    system.eval()
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    return system


@torch.no_grad()
def precompute_readouts(cfg, mode, limit=None):
    device = require_cuda()
    p0a = load_p0a(cfg)
    data = manifest(cfg)
    source_cfg = source_config(cfg)
    system = load_canonical_system(cfg, mode, p0a, device)
    root = Path(cfg["work_dir"]) / "cache" / "readouts" / mode
    total = 0
    for split in ("train", "validation"):
        samples = data[split][:limit] if limit else data[split]
        wrong_indices = nearest_wrong_indices(samples)
        for index, sample in enumerate(samples):
            out = root / split / f"{sample['id']}.pt"
            if out.exists():
                existing = torch.load(out, map_location="cpu", weights_only=True)
                expected_wrong_id = samples[wrong_indices[index]]["id"]
                if existing.get("wrong_id") == expected_wrong_id:
                    total += 1
                    continue
            a = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in load_source_item(cfg, "a", split, sample["id"]).items()}
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in load_source_item(cfg, "b", split, sample["id"]).items()}
            wrong = samples[wrong_indices[index]]
            wrong_a = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in load_source_item(cfg, "a", split, wrong["id"]).items()}
            wrong_b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in load_source_item(cfg, "b", split, wrong["id"]).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                if system is None:
                    ka, va, qa = a["k"], a["v"], a["q"]
                    kb, vb, qb = b["k"], b["v"], b["q"]
                    wrong_va, wrong_vb = wrong_a["v"], wrong_b["v"]
                else:
                    ka, va, qa = system.transform("a", a)
                    kb, vb, qb = system.transform("b", b)
                    _, wrong_va, _ = system.transform("a", wrong_a)
                    _, wrong_vb, _ = system.transform("b", wrong_b)
                attention_a = fixed_attention(qa, ka)
                attention_b = fixed_attention(qb, kb)
                values = {
                    "a_self": fixed_readout(attention_a, va),
                    "a_cross": fixed_readout(attention_a, vb),
                    "a_shuffled": fixed_readout(attention_a, align_v(wrong_vb, va.shape[1])),
                    "b_self": fixed_readout(attention_b, vb),
                    "b_cross": fixed_readout(attention_b, va),
                    "b_shuffled": fixed_readout(attention_b, align_v(wrong_va, vb.shape[1])),
                }
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "id": sample["id"],
                    "wrong_id": wrong["id"],
                    **{name: value.half().cpu() for name, value in values.items()},
                },
                out,
            )
            total += 1
            if total % 8 == 0:
                progress(f"Readout precompute {mode}: {total}")
    del system
    empty_cuda()
    progress(f"Readout precompute {mode}: completed {total}")


class ProbeDataset(Dataset):
    def __init__(self, cfg, mode, split, limit=None):
        self.cfg = cfg
        self.mode = mode
        self.split = split
        samples = manifest(cfg)[split]
        self.samples = samples[:limit] if limit else samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        work = Path(self.cfg["work_dir"]) / "cache"
        readouts = torch.load(
            work / "readouts" / self.mode / self.split / f"{sample['id']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        sentences = torch.load(
            work / "sentence_embeddings" / self.split / f"{sample['id']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if readouts["id"] != sentences["id"]:
            raise RuntimeError(f"Probe data alignment failed for {sample['id']}")
        return {"sample": sample, "readouts": readouts, "sentences": sentences}


def collate_list(rows):
    return rows


class LinearContentProbe(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.linear = nn.Linear(128, text_dim)

    def forward(self, readout):
        return F.normalize(self.linear(readout.float()), dim=-1)


def probe_loss(probe, batch, sender, cfg, device):
    losses = []
    key = f"{sender}_self"
    for row in batch:
        readout = row["readouts"][key].to(device)
        candidates = F.normalize(row["sentences"]["embeddings"].float().to(device), dim=-1)
        labels = row["sentences"]["labels"].to(device)
        scores = probe(readout) @ candidates.T / cfg["probe_temperature"]
        losses.append(torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[labels], dim=0))
    return torch.stack(losses).mean()


def train_probe(cfg, mode, sender, smoke=False):
    device = require_cuda()
    seed_all(cfg["seed"] + (0 if sender == "a" else 1))
    limit = cfg["smoke_train_size"] if smoke else None
    dataset = ProbeDataset(cfg, mode, "train", limit)
    loader = DataLoader(
        dataset,
        batch_size=cfg["probe_batch_size"],
        shuffle=True,
        collate_fn=collate_list,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    first = dataset[0]
    text_dim = first["sentences"]["embeddings"].shape[-1]
    probe = LinearContentProbe(text_dim).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=cfg["probe_learning_rate"],
        weight_decay=cfg["probe_weight_decay"],
    )
    epochs = cfg["smoke_epochs"] if smoke else cfg["formal_epochs"]
    history = []
    tag = f"{mode}_{sender}_{'smoke' if smoke else 'formal'}"
    progress(f"Probe training {tag} started")
    for epoch in range(1, epochs + 1):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = probe_loss(probe, batch, sender, cfg, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite probe loss in {tag}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg["gradient_clip"])
            optimizer.step()
            history.append({"epoch": epoch, "loss": loss.detach().item()})
        progress(f"Probe training {tag}: epoch {epoch}/{epochs}")
    out = Path(cfg["work_dir"])
    checkpoint = out / "checkpoints" / f"{tag}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": probe.state_dict(), "text_dim": text_dim}, checkpoint)
    save_json(out / "metrics" / f"{tag}_history.json", history)
    del probe, optimizer
    empty_cuda()


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


def retrieval_metrics(scores, labels):
    scores = scores.detach().float().cpu().tolist()
    labels = labels.detach().bool().cpu().tolist()
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    gold = max(1, sum(labels))
    result = {}
    for k in (1, 5):
        hits = sum(labels[i] for i in order[:k])
        result[f"support_recall_at_{k}"] = hits / gold
        result[f"support_hit_at_{k}"] = float(hits > 0)
    first = next((rank for rank, i in enumerate(order, 1) if labels[i]), None)
    result["mrr"] = 1.0 / first if first else 0.0
    result["support_auprc"] = average_precision(scores, labels)
    return result


def aggregate(rows):
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


@torch.no_grad()
def evaluate_probe(cfg, mode, sender, smoke=False):
    device = require_cuda()
    limit = cfg["smoke_validation_size"] if smoke else None
    dataset = ProbeDataset(cfg, mode, "validation", limit)
    tag = f"{mode}_{sender}_{'smoke' if smoke else 'formal'}"
    state = torch.load(
        Path(cfg["work_dir"]) / "checkpoints" / f"{tag}.pt",
        map_location="cpu",
        weights_only=True,
    )
    probe = LinearContentProbe(state["text_dim"]).to(device)
    probe.load_state_dict(state["model"])
    probe.eval()
    conditions = {name: [] for name in ("self", "cross", "shuffled")}
    for index in range(len(dataset)):
        row = dataset[index]
        candidates = F.normalize(row["sentences"]["embeddings"].float().to(device), dim=-1)
        labels = row["sentences"]["labels"].to(device)
        for condition in conditions:
            readout = row["readouts"][f"{sender}_{condition}"].to(device)
            scores = probe(readout) @ candidates.T
            conditions[condition].append(retrieval_metrics(scores, labels))
    result = {
        "mode": mode,
        "probe_trained_sender": sender,
        "probe_training_source": f"{sender}_self_only",
        "conditions": {name: aggregate(rows) for name, rows in conditions.items()},
    }
    for condition in result["conditions"].values():
        if not all(math.isfinite(value) for value in condition.values()):
            raise RuntimeError(f"Non-finite probe metric in {tag}")
    del probe
    empty_cuda()
    return result


def cpu_self_test(cfg):
    attention = torch.softmax(torch.randn(2, 3, 2, 7), dim=-1)
    value = torch.randn(2, 7, 2, 8)
    result = fixed_readout(attention, value)
    assert result.shape == (8,) and torch.isfinite(result).all()
    probe = LinearContentProbe(16)
    assert probe(torch.randn(128)).shape == (16,)
    progress("P0-A2 CPU structural self-test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "command",
        choices=["cpu_self_test", "text_embeddings", "readouts", "train_probe", "evaluate_probe"],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=["raw_native", "private_writer", "shared_canonical"])
    parser.add_argument("--sender", choices=["a", "b"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.command == "cpu_self_test":
        cpu_self_test(cfg)
    elif args.command == "text_embeddings":
        extract_text_embeddings(cfg, args.limit)
    elif args.command == "readouts":
        if not args.mode:
            parser.error("--mode is required")
        precompute_readouts(cfg, args.mode, args.limit)
    elif args.command == "train_probe":
        if not args.mode or not args.sender:
            parser.error("--mode and --sender are required")
        train_probe(cfg, args.mode, args.sender, args.smoke)
    elif args.command == "evaluate_probe":
        if not args.mode or not args.sender:
            parser.error("--mode and --sender are required")
        result = evaluate_probe(cfg, args.mode, args.sender, args.smoke)
        suffix = "smoke" if args.smoke else "formal"
        save_json(
            Path(cfg["work_dir"]) / "metrics" / f"{args.mode}_{args.sender}_{suffix}_evaluation.json",
            result,
        )


if __name__ == "__main__":
    main()
