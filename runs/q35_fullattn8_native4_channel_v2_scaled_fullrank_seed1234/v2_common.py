from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
from writer_v2 import ScaledFullRankWriter


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def source_mode(mode):
    return "smoke" if mode == "smoke" else "formal"


def query_mode(mode):
    return "smoke" if mode == "smoke" else "development"


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rows_for(cfg, mode):
    rows = load_json(
        Path(cfg["r1_dir"]) / "artifacts" / source_mode(mode) / "manifest.json"
    )
    sizes = cfg["smoke_sizes"] if mode == "smoke" else cfg["sizes"]
    selected = {k: list(rows[k][:sizes[k]]) for k in sizes}
    for split, samples in selected.items():
        for sample in samples:
            sample["selected_position_ids"] = sample["selected_position_ids"][
                :cfg["max_evidence_tokens"]
            ]
            sample["selected_token_count"] = len(sample["selected_position_ids"])
        for index, sample in enumerate(samples):
            candidates = [
                x for x in samples if x["id"] != sample["id"]
                and x["answer"].strip().lower() != sample["answer"].strip().lower()
            ]
            sample["shuffle_id"] = candidates[(17 * index + 7) % len(candidates)]["id"]
    return selected


def r1_module(cfg):
    path = str(Path(cfg["r1_dir"]))
    if path not in sys.path:
        sys.path.insert(0, path)
    import r1_common
    return r1_common


class Stores:
    def __init__(self, cfg, mode, rows):
        self.cfg, self.mode = cfg, mode
        self.positions = {
            split: {x["id"]: i for i, x in enumerate(samples)}
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
        if sender == "q35":
            shard_size = self.cfg["shard_size"]
            path = Path(self.cfg["work_dir"]) / "cache" / self.mode / split / "q35" / f"shard_{index // shard_size:05d}.pt"
        else:
            shard_size = 32
            path = Path(self.cfg["r1_dir"]) / "cache" / source_mode(self.mode) / split / sender / f"shard_{index // shard_size:05d}.pt"
        record = self._load((split, sender, index // shard_size), path)[index % shard_size]
        if record["id"] != sample_id:
            raise RuntimeError("shard mismatch")
        return record

    def query(self, split, sample_id):
        index = self.positions[split][sample_id]
        size = self.cfg["query_shard_size"]
        path = Path(self.cfg["a0_dir"]) / "cache" / query_mode(self.mode) / split / "query_4b" / f"shard_{index // size:05d}.pt"
        return self._load(("query", split, index // size), path)[index % size]

    def memory(self, split, sender, sample, kind="correct"):
        source_id = sample["shuffle_id"] if kind == "shuffled" else sample["id"]
        record = self.native(split, sender, source_id)
        target = sample["selected_token_count"]
        valid = min(target, record["native_k"].shape[1])
        key, value = record["native_k"][:, :target].clone(), record["native_v"][:, :target].clone()
        if valid < target:
            key = F.pad(key, (0, 0, 0, 0, 0, target - valid))
            value = F.pad(value, (0, 0, 0, 0, 0, target - valid))
        mask = torch.zeros(1, target, dtype=torch.long)
        mask[:, :valid] = 1
        if kind == "zero":
            key.zero_(); value.zero_()
        return key, value, mask


def load_scales(cfg, mode):
    return torch.load(
        Path(cfg["work_dir"]) / "artifacts" / mode / "scales.pt",
        map_location="cpu", weights_only=False,
    )


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


def normalized_target(writer, key, value):
    tokens = key.shape[1]
    return (
        key.float().reshape(36, tokens, 1024) / writer.scale_target_k[:, None],
        value.float().reshape(36, tokens, 1024) / writer.scale_target_v[:, None],
    )


def layer_metrics(pred_k, pred_v, gold_k, gold_v, mask=None):
    if mask is not None:
        pred_k, pred_v = pred_k[:, mask], pred_v[:, mask]
        gold_k, gold_v = gold_k[:, mask], gold_v[:, mask]
    rows = []
    for layer in range(pred_k.shape[0]):
        def metric(pred, gold):
            error = (pred - gold).square().mean()
            nmse = error / gold.square().mean().clamp_min(1e-8)
            cosine = F.cosine_similarity(pred.flatten(), gold.flatten(), 0)
            return nmse, cosine
        kn, kc = metric(pred_k[layer], gold_k[layer])
        vn, vc = metric(pred_v[layer], gold_v[layer])
        rows.append({
            "layer": layer, "k_nmse": kn, "v_nmse": vn,
            "k_cosine": kc, "v_cosine": vc,
        })
    return rows


def apply_rope(x, positions, theta):
    inverse = 1.0 / (
        float(theta) ** (
            torch.arange(0, x.shape[-1], 2, device=x.device).float()
            / x.shape[-1]
        )
    )
    frequency = torch.outer(
        torch.tensor(positions, device=x.device).float(), inverse
    )
    embedding = torch.cat((frequency, frequency), -1)[:, None]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x.float() * embedding.cos() + rotated.float() * embedding.sin()


def attention_diagnostics(cfg, query, pred_k, pred_v, gold_k, gold_v, positions):
    rows = []
    for layer in range(pred_k.shape[0]):
        target_layer = (
            cfg["target_depth_anchors"][layer]
            if pred_k.shape[0] == 8 else layer
        )
        q = query["query"][target_layer].to(pred_k.device)
        pk = apply_rope(pred_k[layer], positions, cfg["rope_theta"])
        gk = apply_rope(gold_k[layer], positions, cfg["rope_theta"])
        pk = pk.repeat_interleave(4, 1)
        gk = gk.repeat_interleave(4, 1)
        pv = pred_v[layer].float().repeat_interleave(4, 1)
        gv = gold_v[layer].float().repeat_interleave(4, 1)
        q = apply_rope(
            q, query["query_position_ids"], cfg["rope_theta"]
        )
        logits_p = torch.einsum("qhd,thd->hqt", q, pk) / math.sqrt(128)
        logits_g = torch.einsum("qhd,thd->hqt", q, gk) / math.sqrt(128)
        attn_p, attn_g = logits_p.softmax(-1), logits_g.softmax(-1)
        route = (
            attn_g * (
                attn_g.clamp_min(1e-12).log()
                - attn_p.clamp_min(1e-12).log()
            )
        ).sum(-1).mean()
        out_p = torch.einsum("hqt,thd->qhd", attn_p, pv)
        out_g = torch.einsum("hqt,thd->qhd", attn_g, gv)
        rows.append({
            "route_kl": route,
            "attention_output_cosine": F.cosine_similarity(
                out_p.flatten(), out_g.flatten(), 0
            ),
        })
    return rows


def representation_loss(rows):
    return torch.stack([
        row["k_smooth_l1"] + row["v_smooth_l1"]
        + 0.1 * (1 - row["k_cosine"]) + 0.1 * (1 - row["v_cosine"])
        for row in rows
    ]).mean()


def loss_rows(pred_k, pred_v, gold_k, gold_v, mask):
    pred_k, pred_v = pred_k[:, mask], pred_v[:, mask]
    gold_k, gold_v = gold_k[:, mask], gold_v[:, mask]
    rows = []
    for layer in range(pred_k.shape[0]):
        rows.append({
            "k_smooth_l1": F.smooth_l1_loss(pred_k[layer], gold_k[layer]),
            "v_smooth_l1": F.smooth_l1_loss(pred_v[layer], gold_v[layer]),
            "k_cosine": F.cosine_similarity(
                pred_k[layer].flatten(), gold_k[layer].flatten(), 0
            ),
            "v_cosine": F.cosine_similarity(
                pred_v[layer].flatten(), gold_v[layer].flatten(), 0
            ),
        })
    return rows


def predict(cfg, writer, store, split, sample, kind="correct"):
    source_k, source_v, prefix = store.memory(split, "q35", sample, kind)
    output = writer(source_k.to(cuda()), source_v.to(cuda()))
    return output, prefix


@torch.no_grad()
def validate_rep(cfg, writer, store, samples, stage, split="validation"):
    writer.eval()
    all_rows = []
    anchors = cfg["target_depth_anchors"]
    for sample in samples:
        source_k, source_v, prefix = store.memory(split, "q35", sample)
        target_k, target_v, _ = store.memory(split, "4b", sample)
        source_k, source_v = source_k.to(cuda()), source_v.to(cuda())
        target_k, target_v = target_k.to(cuda()), target_v.to(cuda())
        gold_k, gold_v = normalized_target(writer, target_k, target_v)
        if stage == "a1":
            pred_k, pred_v = writer.features(source_k, source_v)
            gold_k, gold_v = gold_k[anchors], gold_v[anchors]
            raw_pred_k = pred_k * writer.scale_target_k[anchors, None]
            raw_pred_v = pred_v * writer.scale_target_v[anchors, None]
            raw_gold_k = target_k[anchors].float().reshape(
                8, target_k.shape[1], 8, 128
            )
            raw_gold_v = target_v[anchors].float().reshape_as(raw_gold_k)
        else:
            pred_k, pred_v = writer.standardized_output(source_k, source_v)
            raw_pred_k = pred_k * writer.scale_target_k[:, None]
            raw_pred_v = pred_v * writer.scale_target_v[:, None]
            raw_gold_k = target_k.float().reshape(36, target_k.shape[1], 8, 128)
            raw_gold_v = target_v.float().reshape_as(raw_gold_k)
        mask = prefix[0].bool().to(cuda())
        metrics = layer_metrics(pred_k, pred_v, gold_k, gold_v, mask)
        diagnostics = attention_diagnostics(
            cfg, store.query(split, sample["id"]),
            raw_pred_k[:, mask].reshape(raw_pred_k.shape[0], -1, 8, 128),
            raw_pred_v[:, mask].reshape(raw_pred_v.shape[0], -1, 8, 128),
            raw_gold_k[:, mask], raw_gold_v[:, mask],
            [p for p, keep in zip(sample["selected_position_ids"], mask.tolist()) if keep],
        )
        for metric, diagnostic in zip(metrics, diagnostics):
            metric.update(diagnostic)
        all_rows.append(metrics)
    result = []
    for layer in range(len(all_rows[0])):
        row = {"layer": layer, "target_layer": anchors[layer] if stage == "a1" else layer}
        for key in (
            "k_nmse", "v_nmse", "k_cosine", "v_cosine",
            "route_kl", "attention_output_cosine",
        ):
            row[key] = sum(float(x[layer][key]) for x in all_rows) / len(all_rows)
        row["representation_score"] = (
            row["k_nmse"] + row["v_nmse"]
            + 0.1 * (2 - row["k_cosine"] - row["v_cosine"])
        )
        result.append(row)
    return result


def checkpoint_writer(path, writer, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"writer": writer.state_dict(), **metadata}, path)


def load_writer(cfg, mode, variant, checkpoint=None):
    writer = ScaledFullRankWriter(cfg, load_scales(cfg, mode), variant).to(cuda())
    if checkpoint:
        writer.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["writer"]
        )
    return writer


def accumulation(cfg, mode):
    return cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]
