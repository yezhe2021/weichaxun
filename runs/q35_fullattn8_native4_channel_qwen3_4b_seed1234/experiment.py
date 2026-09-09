from __future__ import annotations

import argparse
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
from writer import Qwen35FullAttentionWriter, relative_depth_matrix


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    rows = load_json(Path(cfg["r1_dir"]) / "artifacts" / source_mode(mode) / "manifest.json")
    sizes = cfg["smoke_sizes"] if mode == "smoke" else cfg["sizes"]
    selected = {split: list(rows[split][:sizes[split]]) for split in sizes}
    limit = cfg["max_evidence_tokens"]
    for split, samples in selected.items():
        for sample in samples:
            sample["selected_position_ids"] = sample["selected_position_ids"][:limit]
            sample["selected_token_count"] = len(sample["selected_position_ids"])
        for index, sample in enumerate(samples):
            candidates = [
                x for x in samples
                if x["id"] != sample["id"]
                and x["answer"].strip().lower() != sample["answer"].strip().lower()
            ]
            sample["shuffle_id"] = candidates[(index * 17 + 7) % len(candidates)]["id"]
    return selected


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_maps(cfg):
    train = {row["_id"]: row for row in load_json(cfg["hotpot_train"])}
    dev = {row["_id"]: row for row in load_json(cfg["hotpot_dev"])}
    return train, dev


def encoded_segments(raw, tokenizer):
    """Reproduce R1's segment-wise tokenization while retaining global char offsets."""
    ids, offsets, cursor = [], [], 0
    support = {(str(title), int(index)) for title, index in raw["supporting_facts"]}
    support_chars = []

    def add(text, selected=False):
        nonlocal cursor
        encoded = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        start_index = len(ids)
        ids.extend(encoded.input_ids)
        offsets.extend([(cursor + a, cursor + b) for a, b in encoded.offset_mapping])
        if selected:
            support_chars.extend(range(start_index, len(ids)))
        cursor += len(text)

    for title, sentences in raw["context"]:
        add(f"Document: {title}\n")
        for index, sentence in enumerate(sentences):
            add(f"Sentence {index}: ")
            add(sentence, (str(title), index) in support)
            add("\n")
    return ids, offsets, support_chars


def overlap_alignment(target_spans, source_spans):
    rows, used = [], set()
    for ta, tb in target_spans:
        width = max(tb - ta, 1)
        values = {}
        for index, (sa, sb) in enumerate(source_spans):
            overlap = max(0, min(tb, sb) - max(ta, sa))
            if overlap:
                values[index] = overlap / width
                used.add(index)
        if not values:
            raise RuntimeError(f"no Qwen3.5 token overlap for target span {(ta, tb)}")
        rows.append(values)
    selected = sorted(used)
    remap = {old: new for new, old in enumerate(selected)}
    matrix = torch.zeros(len(rows), len(selected), dtype=torch.float32)
    for target, values in enumerate(rows):
        for source, weight in values.items():
            matrix[target, remap[source]] = weight
        total = matrix[target].sum()
        if not torch.isfinite(total) or total <= 0:
            raise RuntimeError("invalid token alignment row")
        matrix[target] /= total
    return selected, matrix


class Q35Capture:
    def __init__(self, model, cfg):
        self.cfg, self.values, self.indices = cfg, {}, None
        self.layer_ids = cfg["q35_full_attention_layers"]
        self.handles = [
            model.model.layers[index].self_attn.register_forward_pre_hook(
                self._hook(index), with_kwargs=True
            )
            for index in self.layer_ids
        ]

    def _hook(self, layer_index):
        def capture(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            batch, length, _ = hidden.shape
            heads, dim = self.cfg["source_kv_heads"], self.cfg["source_head_dim"]
            key = module.k_norm(module.k_proj(hidden).view(batch, length, heads, dim))
            value = module.v_proj(hidden).view(batch, length, heads, dim)
            self.values[layer_index] = (
                key[0, self.indices].detach(),
                value[0, self.indices].detach(),
            )
        return capture

    @torch.no_grad()
    def run(self, model, ids, source_indices, alignment):
        self.values.clear()
        device = require_cuda()
        self.indices = torch.tensor(source_indices, dtype=torch.long, device=device)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        positions = torch.arange(len(ids), device=device).unsqueeze(0)
        model(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=positions,
            use_cache=False,
        )
        if set(self.values) != set(self.layer_ids):
            raise RuntimeError(f"captured layers {sorted(self.values)}")
        weights = alignment.to(device=device, dtype=torch.float32)
        keys, values = [], []
        for layer in self.layer_ids:
            key, value = self.values[layer]
            keys.append(torch.einsum("ts,shd->thd", weights, key.float()))
            values.append(torch.einsum("ts,shd->thd", weights, value.float()))
        key = torch.stack(keys).reshape(
            len(self.layer_ids), len(alignment), self.cfg["target_kv_heads"],
            self.cfg["target_head_dim"]
        )
        value = torch.stack(values).reshape_as(key)
        return key.cpu().half().contiguous(), value.cpu().half().contiguous()

    def close(self):
        for handle in self.handles:
            handle.remove()


def selftest(cfg):
    base = relative_depth_matrix(cfg["target_depth_anchors"], cfg["target_layers"])
    assert base.shape == (36, 8)
    assert torch.allclose(base.sum(-1), torch.ones(36))
    assert base[3, 0] == 1 and base[35, 7] == 1
    spans = [(0, 2), (2, 4)]
    indices, matrix = overlap_alignment(spans, [(0, 1), (1, 3), (3, 4)])
    assert indices == [0, 1, 2] and torch.allclose(matrix.sum(-1), torch.ones(2))
    for variant in ("s0", "s1", "s2"):
        writer = Qwen35FullAttentionWriter(cfg, variant)
        assert not any(
            isinstance(module, torch.nn.Linear) and module.bias is not None
            for module in writer.modules()
        )
        assert writer.zero_check(torch.device("cpu")) == 0.0
        x = torch.randn(8, 2, 8, 128)
        out = writer(x, x)
        assert out[0].shape == (36, 2, 8, 128)
    progress("CPU selftest passed: alignment, depth map, dimensions, bias=False, Writer(0)=0")


def audit(cfg, mode):
    rows = rows_for(cfg, mode)
    q35 = AutoConfig.from_pretrained(cfg["model_q35"], local_files_only=True).text_config
    q4 = AutoConfig.from_pretrained(cfg["model_4b"], local_files_only=True)
    actual_full = [i for i, kind in enumerate(q35.layer_types) if kind == "full_attention"]
    expected_full = cfg["q35_full_attention_layers"]
    if actual_full != expected_full:
        raise RuntimeError(f"Qwen3.5 full attention layers {actual_full} != {expected_full}")
    source_shape = (q35.num_key_value_heads, q35.head_dim)
    target_shape = (q4.num_key_value_heads, q4.head_dim)
    if source_shape != (4, 256) or target_shape != (8, 128):
        raise RuntimeError(f"unexpected KV shapes: {source_shape} -> {target_shape}")
    tok35 = AutoTokenizer.from_pretrained(cfg["model_q35"], local_files_only=True, use_fast=True)
    tok4 = AutoTokenizer.from_pretrained(cfg["model_4b"], local_files_only=True, use_fast=True)
    if len(tok35) == len(tok4):
        raise RuntimeError("expected distinct Qwen3.5 and Qwen3 tokenizers")
    reader = Path(cfg["r1_dir"]) / "artifacts" / source_mode(mode) / "sparse_reader" / "best.pt"
    if not reader.exists():
        raise RuntimeError(f"R1 sparse Reader missing: {reader}")
    query_root = Path(cfg["a0_dir"]) / "cache" / query_mode(mode)
    if not query_root.exists():
        raise RuntimeError(f"4B query cache missing: {query_root}")
    report = {
        "passed": True,
        "sample_count": {k: len(v) for k, v in rows.items()},
        "sender": "Qwen3.5-4B evidence-only, 8 full-attention layers only",
        "excluded": ["24 DeltaNet pseudo-KV", "DeltaNet convolution state",
                     "DeltaNet recurrent state", "32-layer hidden-state residual"],
        "full_attention_layers": actual_full,
        "source_pre_rope_k": True,
        "source_native_v": True,
        "source_shape": [8, "T_q35", 4, 256],
        "channel_shape": [36, "T_q4", 8, 128],
        "token_alignment": "deterministic character-offset overlap; Qwen3-4B target slots",
        "reader_checkpoint": str(reader),
        "reader_checkpoint_sha256": sha256(reader),
        "reader_frozen": True,
        "question_never_enters_sender": True,
        "hard_gates_enforced": False,
    }
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "protocol_audit.json", report)
    progress(f"{mode}: protocol audit passed")


@torch.no_grad()
def extract(cfg, mode):
    rows = rows_for(cfg, mode)
    raw_train, raw_dev = load_raw_maps(cfg)
    tok4 = AutoTokenizer.from_pretrained(cfg["model_4b"], local_files_only=True, use_fast=True)
    tok35 = AutoTokenizer.from_pretrained(cfg["model_q35"], local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_q35"], local_files_only=True, dtype=torch.float16,
        low_cpu_mem_usage=True, device_map={"": 0}
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = Q35Capture(model, cfg)
    root = Path(cfg["work_dir"]) / "cache" / mode
    alignment_audit = []
    for split, samples in rows.items():
        raw_map = raw_train if split == "train" else raw_dev
        out_dir = root / split / "q35"
        out_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(samples), cfg["shard_size"]):
            shard = start // cfg["shard_size"]
            destination = out_dir / f"shard_{shard:05d}.pt"
            if destination.exists():
                continue
            records = []
            for sample in samples[start:start + cfg["shard_size"]]:
                raw = raw_map[sample["id"]]
                ids4, offsets4, selected4 = encoded_segments(raw, tok4)
                if ids4 != sample["full_context_token_ids"]:
                    raise RuntimeError(f"R1/Qwen3 context mismatch: {sample['id']}")
                target_indices = sample["selected_position_ids"]
                if target_indices != selected4[:len(target_indices)]:
                    raise RuntimeError(f"support token mismatch: {sample['id']}")
                ids35, offsets35, _ = encoded_segments(raw, tok35)
                target_spans = [offsets4[index] for index in target_indices]
                source_indices, matrix = overlap_alignment(target_spans, offsets35)
                key, value = capture.run(model, ids35, source_indices, matrix)
                records.append({
                    "id": sample["id"], "native_k": key, "native_v": value,
                    "source_tokens": len(source_indices),
                    "target_tokens": len(target_indices),
                })
                alignment_audit.append({
                    "id": sample["id"], "split": split,
                    "q35_context_tokens": len(ids35),
                    "aligned_q35_tokens": len(source_indices),
                    "target_q4_tokens": len(target_indices),
                    "row_sum_min": matrix.sum(-1).min().item(),
                    "row_sum_max": matrix.sum(-1).max().item(),
                })
            temporary = destination.with_suffix(".tmp")
            torch.save(records, temporary)
            temporary.replace(destination)
            progress(f"{mode}: Qwen3.5 extract {split} shard {shard + 1}/{math.ceil(len(samples)/cfg['shard_size'])}")
    capture.close()
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "alignment_audit.json", alignment_audit)
    del model
    torch.cuda.empty_cache()


class Stores:
    def __init__(self, cfg, mode, rows):
        self.cfg, self.mode = cfg, mode
        self.positions = {
            split: {sample["id"]: i for i, sample in enumerate(samples)}
            for split, samples in rows.items()
        }
        self.cache = {}

    def _load(self, key, path):
        if key not in self.cache:
            if len(self.cache) > 12:
                self.cache.clear()
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.cache[key]

    def native4(self, split, sample_id):
        index = self.positions[split][sample_id]
        shard = index // 32
        path = Path(self.cfg["r1_dir"]) / "cache" / source_mode(self.mode) / split / "4b" / f"shard_{shard:05d}.pt"
        record = self._load(("4b", split, shard), path)[index % 32]
        if record["id"] != sample_id:
            raise RuntimeError("4B shard mismatch")
        return record

    def q35(self, split, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["shard_size"]
        path = Path(self.cfg["work_dir"]) / "cache" / self.mode / split / "q35" / f"shard_{shard:05d}.pt"
        record = self._load(("q35", split, shard), path)[index % self.cfg["shard_size"]]
        if record["id"] != sample_id:
            raise RuntimeError("Qwen3.5 shard mismatch")
        return record

    def query4(self, split, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["query_shard_size"]
        path = Path(self.cfg["a0_dir"]) / "cache" / query_mode(self.mode) / split / "query_4b" / f"shard_{shard:05d}.pt"
        record = self._load(("query", split, shard), path)[index % self.cfg["query_shard_size"]]
        if record["id"] != sample_id:
            raise RuntimeError("query shard mismatch")
        return record

    def memory(self, split, family, sample, kind="correct"):
        source_id = sample["shuffle_id"] if kind == "shuffled" else sample["id"]
        record = self.native4(split, source_id) if family == "4b" else self.q35(split, source_id)
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


def apply_rope(x, positions, theta):
    inverse = 1.0 / (
        float(theta) ** (
            torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1]
        )
    )
    frequency = torch.outer(torch.tensor(positions, device=x.device).float(), inverse)
    embedding = torch.cat((frequency, frequency), -1)[None, :, None]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x.float() * embedding.cos() + rotated.float() * embedding.sin()


def representation_losses(cfg, query, prediction, target, positions):
    pred_k, pred_v = prediction
    gold_k, gold_v = target

    def nmse(pred, gold):
        return ((pred.float() - gold.float()).square().mean(1) /
                gold.float().square().mean(1).clamp_min(1e-8)).mean()

    def cosine(pred, gold):
        p = pred.float().permute(0, 2, 1, 3).reshape(36, 8, -1)
        g = gold.float().permute(0, 2, 1, 3).reshape(36, 8, -1)
        return F.cosine_similarity(p, g, -1).mean()

    groups = cfg["target_query_heads"] // cfg["target_kv_heads"]
    q = apply_rope(query["query"].to(pred_k.device), query["query_position_ids"], cfg["rope_theta"])
    pk = apply_rope(pred_k, positions, cfg["rope_theta"]).repeat_interleave(groups, 2)
    gk = apply_rope(gold_k, positions, cfg["rope_theta"]).repeat_interleave(groups, 2)
    pv = pred_v.float().repeat_interleave(groups, 2)
    gv = gold_v.float().repeat_interleave(groups, 2)
    logits_p = torch.einsum("lqhd,lthd->lhqt", q, pk) / math.sqrt(128)
    logits_g = torch.einsum("lqhd,lthd->lhqt", q, gk) / math.sqrt(128)
    attn_p, attn_g = logits_p.softmax(-1), logits_g.softmax(-1)
    route = (attn_g * (attn_g.clamp_min(1e-12).log() -
                       attn_p.clamp_min(1e-12).log())).sum(-1).mean()
    out_p = torch.einsum("lhqt,lthd->lqhd", attn_p, pv)
    out_g = torch.einsum("lhqt,lthd->lqhd", attn_g, gv)
    out_nmse = (out_p - out_g).square().mean() / out_g.square().mean().clamp_min(1e-8)
    out_cos = F.cosine_similarity(out_p.flatten(), out_g.flatten(), 0)
    k_nmse, v_nmse = nmse(pred_k, gold_k), nmse(pred_v, gold_v)
    k_cos, v_cos = cosine(pred_k, gold_k), cosine(pred_v, gold_v)
    result = {
        "k_nmse": k_nmse, "v_nmse": v_nmse, "k_cosine": k_cos,
        "v_cosine": v_cos, "route_kl": route,
        "output_nmse": out_nmse, "output_cosine": out_cos,
    }
    result["stage_a"] = (
        0.5 * (k_nmse + v_nmse) + 0.25 * (2 - k_cos - v_cos)
        + 0.5 * route + out_nmse + 0.5 * (1 - out_cos)
    )
    return result


def writer_values(cfg, writer, store, split, sample, kind="correct"):
    k35, v35, mask = store.memory(split, "q35", sample, kind)
    k4, v4, _ = store.memory(split, "4b", sample, "correct")
    prediction = writer(k35.to(require_cuda()), v35.to(require_cuda()))
    values = representation_losses(
        cfg, store.query4(split, sample["id"]), prediction,
        (k4.to(require_cuda()), v4.to(require_cuda())),
        sample["selected_position_ids"],
    )
    return prediction, mask, values


def predict_memory(writer, store, split, sample, kind="correct"):
    key, value, mask = store.memory(split, "q35", sample, kind)
    return writer(key.to(require_cuda()), value.to(require_cuda())), mask


@torch.no_grad()
def validate_representation(cfg, writer, store, samples, split="validation"):
    writer.eval()
    rows = []
    for sample in samples:
        _, _, values = writer_values(cfg, writer, store, split, sample)
        rows.append({key: value.item() for key, value in values.items()})
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def schedule(optimizer, cfg, maximum):
    warmup = min(cfg["warmup_updates"], maximum)
    return LambdaLR(optimizer, lambda step: min((step + 1) / max(warmup, 1), 1.0))


def accumulation(cfg, mode):
    return cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]


def stage_a(cfg, mode, variant):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    writer = Qwen35FullAttentionWriter(cfg, variant).to(require_cuda())
    if writer.zero_check(require_cuda()) != 0:
        raise RuntimeError("Writer(0) != 0")
    optimizer = AdamW(writer.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["stage_a_updates"]
    scheduler = schedule(optimizer, cfg, maximum)
    interval = 1 if mode == "smoke" else cfg["eval_interval"]
    grad_acc = accumulation(cfg, mode)
    out = Path(cfg["work_dir"]) / "artifacts" / mode / variant / "stage_a"
    out.mkdir(parents=True, exist_ok=True)
    best, history, evaluations = -float("inf"), [], []
    samples, cursor, epoch = list(rows["train"]), 0, 0
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        micro = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            writer.train()
            _, _, values = writer_values(cfg, writer, store, "train", sample)
            (values["stage_a"] / grad_acc).backward()
            micro.append({k: v.detach().item() for k, v in values.items()})
        grad_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]).item()
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        history.append({
            "update": update, "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0],
            **{k: sum(x[k] for x in micro) / len(micro) for k in micro[0]},
        })
        if update % interval == 0 or update == maximum:
            validation = validate_representation(cfg, writer, store, rows["validation"])
            selected = validation["output_cosine"] > best
            if selected:
                best = validation["output_cosine"]
                torch.save({"writer": writer.state_dict(), "update": update,
                            "validation": validation}, out / "best.pt")
            evaluations.append({"update": update, **validation, "selected": selected})
            save_json(out / "history.json", history)
            save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}/{variant}: stage A {update}/{maximum}")
    save_json(out / "summary.json", {"completed": True, "best_output_cosine": best})


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
    nlls = []
    for sample in samples[:min(16, len(samples))]:
        prediction, mask = predict_memory(
            writer, store, "validation", sample, "correct"
        )
        nlls.append(r1.answer_loss(
            cfg, reader, tok, sample, prediction[0].half(), prediction[1].half(), mask
        ).item())
    return sum(nlls) / len(nlls)


def stage_b(cfg, mode, variant):
    """Functional-only: no KV consistency term contributes any gradient."""
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    writer = Qwen35FullAttentionWriter(cfg, variant).to(require_cuda())
    stage_a_path = Path(cfg["work_dir"]) / "artifacts" / mode / variant / "stage_a" / "best.pt"
    writer.load_state_dict(torch.load(stage_a_path, map_location="cpu", weights_only=False)["writer"])
    r1, reader, tok = load_reader(cfg, mode)
    optimizer = AdamW(writer.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    maximum = cfg["smoke_updates"] if mode == "smoke" else cfg["stage_b_updates"]
    scheduler = schedule(optimizer, cfg, maximum)
    interval = 1 if mode == "smoke" else cfg["eval_interval"]
    grad_acc = accumulation(cfg, mode)
    out = Path(cfg["work_dir"]) / "artifacts" / mode / variant / "stage_b"
    out.mkdir(parents=True, exist_ok=True)
    best, history, evaluations = float("inf"), [], []
    samples, cursor, epoch = list(rows["train"]), 0, 0
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, maximum + 1):
        micro = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + 10000 + epoch).shuffle(samples)
                epoch += 1
            sample = samples[cursor]
            cursor = (cursor + 1) % len(samples)
            writer.train()
            correct, mask = predict_memory(
                writer, store, "train", sample, "correct"
            )
            shuffled, wrong_mask = predict_memory(
                writer, store, "train", sample, "shuffled"
            )
            correct_nll = r1.answer_loss(
                cfg, reader, tok, sample, correct[0].half(), correct[1].half(), mask
            )
            shuffled_nll = r1.answer_loss(
                cfg, reader, tok, sample, shuffled[0].half(), shuffled[1].half(), wrong_mask
            )
            dependence = F.relu(cfg["dependence_margin"] + correct_nll - shuffled_nll)
            loss = correct_nll + cfg["dependence_weight"] * dependence
            (loss / grad_acc).backward()
            micro.append({
                "functional_loss": loss.detach().item(),
                "answer_nll": correct_nll.detach().item(),
                "shuffled_nll": shuffled_nll.detach().item(),
                "dependence": dependence.detach().item(),
                "kv_loss_weight": 0.0,
            })
        grad_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg["gradient_clip"]).item()
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        history.append({
            "update": update, "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0],
            **{k: sum(x[k] for x in micro) / len(micro) for k in micro[0]},
        })
        if update % interval == 0 or update == maximum:
            validation_nll = validate_answer(cfg, writer, store, reader, tok, r1, rows["validation"])
            representation = validate_representation(cfg, writer, store, rows["validation"])
            selected = validation_nll < best
            if selected:
                best = validation_nll
                torch.save({
                    "writer": writer.state_dict(), "update": update,
                    "validation_nll": validation_nll,
                    "selection_metric": "functional_answer_nll",
                    "kv_diagnostic_only": representation,
                }, out / "best.pt")
            evaluations.append({
                "update": update, "validation_nll": validation_nll,
                **representation, "selected": selected,
            })
            save_json(out / "history.json", history)
            save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}/{variant}: stage B functional-only {update}/{maximum}")
    save_json(out / "summary.json", {
        "completed": True, "best_validation_nll": best,
        "selection_metric": "functional_answer_nll", "kv_loss_weight": 0.0,
    })
    del reader
    torch.cuda.empty_cache()


def aggregate(rows):
    summary = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            key: sum(float(row[key]) for row in selected) / len(selected)
            for key in ("em", "token_f1", "nll")
        }
        summary[condition]["count"] = len(selected)
    return summary


def evaluate(cfg, mode):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    r1, reader, tok = load_reader(cfg, mode)
    writers = {"s0": Qwen35FullAttentionWriter(cfg, "s0").to(require_cuda()).eval()}
    for variant in cfg["train_variants"]:
        writer = Qwen35FullAttentionWriter(cfg, variant).to(require_cuda()).eval()
        checkpoint = torch.load(
            Path(cfg["work_dir"]) / "artifacts" / mode / variant / "stage_b" / "best.pt",
            map_location="cpu", weights_only=False,
        )
        writer.load_state_dict(checkpoint["writer"])
        writers[variant] = writer
    conditions = [
        ("question_only", "none", "correct", True),
        ("supporting_text", "text", "correct", False),
        ("native4_correct", "4b", "correct", True),
        ("native4_shuffled", "4b", "shuffled", True),
        ("native4_zero", "4b", "zero", True),
        ("reader_off", "off", "correct", False),
    ]
    for variant in ("s0", "s1", "s2"):
        for kind in ("correct", "shuffled", "zero"):
            conditions.append((f"q35_{variant}_{kind}", variant, kind, True))
    output = []
    for condition, family, kind, lora_enabled in conditions:
        r1.set_lora(reader, lora_enabled)
        progress(f"{mode}: evaluate {condition}")
        for sample in rows["test"]:
            key = value = mask = None
            supporting = family == "text"
            compact = family in {"none", "off"}
            if family == "4b":
                key, value, mask = store.memory("test", "4b", sample, kind)
                key, value = key.to(require_cuda()), value.to(require_cuda())
            elif family in writers:
                source_k, source_v, mask = store.memory("test", "q35", sample, kind)
                with torch.no_grad():
                    key, value = writers[family](
                        source_k.to(require_cuda()), source_v.to(require_cuda())
                    )
                if kind == "zero":
                    key = torch.zeros_like(key)
                    value = torch.zeros_like(value)
            with torch.no_grad():
                prediction = r1.greedy_generate(
                    cfg, reader, tok, sample,
                    None if key is None else key.half(),
                    None if value is None else value.half(),
                    mask, supporting_text=supporting, compact_positions=compact,
                )
                if supporting:
                    target = r1.answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
                    prompt = sample["full_sequence_token_ids"]
                    sequence = torch.tensor([prompt + target[:-1]], device=require_cuda())
                    logits = reader(
                        sequence, attention_mask=torch.ones_like(sequence),
                        position_ids=torch.arange(sequence.shape[1], device=require_cuda()).unsqueeze(0),
                        use_cache=False,
                    ).logits
                    selected = logits[:, len(prompt)-1:len(prompt)-1+len(target)].float()
                    nll = F.cross_entropy(
                        selected.reshape(-1, selected.shape[-1]),
                        torch.tensor(target, device=require_cuda()),
                    ).item()
                else:
                    nll = r1.answer_loss(
                        cfg, reader, tok, sample,
                        None if key is None else key.half(),
                        None if value is None else value.half(),
                        mask, compact_question_positions=compact,
                    ).item()
            output.append({
                "sample_id": sample["id"], "type": sample.get("type", "unknown"),
                "condition": condition, "answer": sample["answer"],
                "prediction": prediction,
                "em": float(r1.normalize_answer(prediction) == r1.normalize_answer(sample["answer"])),
                "token_f1": r1.token_f1(prediction, sample["answer"]), "nll": nll,
                "manual_c_p_w": "",
            })
    summary = aggregate(output)
    for variant, writer in writers.items():
        summary[f"q35_{variant}_representation_diagnostic"] = validate_representation(
            cfg, writer, store, rows["test"], split="test"
        )
        correct = summary[f"q35_{variant}_correct"]
        shuffled = summary[f"q35_{variant}_shuffled"]
        summary[f"q35_{variant}_functional_diagnostic"] = {
            "correct_shuffled_em_gap": correct["em"] - shuffled["em"],
            "correct_shuffled_nll_gap": shuffled["nll"] - correct["nll"],
            "diagnostic_only": True,
        }
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "summary.json", summary)
    save_json(out / "per_sample.json", output)
    with (out / "manual_c_p_w.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader(); writer.writerows(output)
    save_json(out / "completion.json", {
        "completed": True, "hard_gates_enforced": False,
        "sender_uses_question": False, "reader_frozen": True,
        "stage_b_kv_loss_weight": 0.0,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("action", choices=(
        "selftest", "audit", "extract", "stage_a", "stage_b", "evaluate"
    ))
    parser.add_argument("--variant", choices=("s1", "s2"))
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.action in {"stage_a", "stage_b"} and not args.variant:
        parser.error("--variant is required")
    if args.action == "selftest":
        selftest(cfg)
    elif args.action == "audit":
        audit(cfg, args.mode)
    elif args.action == "extract":
        extract(cfg, args.mode)
    elif args.action == "stage_a":
        stage_a(cfg, args.mode, args.variant)
    elif args.action == "stage_b":
        stage_b(cfg, args.mode, args.variant)
    else:
        evaluate(cfg, args.mode)


if __name__ == "__main__":
    main()
