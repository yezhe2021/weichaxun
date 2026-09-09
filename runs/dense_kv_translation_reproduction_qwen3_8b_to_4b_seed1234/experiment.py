from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def progress(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; stop without starting the experiment")
    return torch.device("cuda")


def sha_ids(ids):
    return hashlib.sha256(torch.as_tensor(ids, dtype=torch.int64).numpy().tobytes()).hexdigest()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_gold(answer):
    m = re.findall(r"####\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?)", answer or "")
    return m[-1].replace(",", "") if m else None


def parse_generated(text):
    m = re.findall(r"####\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?)", text or "")
    if not m:
        return None
    value = m[-1].replace(",", "")
    try:
        return str(int(value)) if float(value).is_integer() else str(float(value))
    except ValueError:
        return value


def format_question(question):
    return (
        "Solve the following math problem carefully. End with exactly "
        "'#### <numeric answer>'.\n\nQuestion: " + question.strip() + "\n\nAnswer:"
    )


def make_splits(cfg):
    out = Path(cfg["work_dir"]) / "manifests"
    split_path = out / "splits.json"
    if split_path.exists():
        return load_json(split_path)
    train = read_jsonl(cfg["gsm8k_train"])
    test = read_jsonl(cfg["gsm8k_test"])
    rng = random.Random(cfg["seed"])
    rng.shuffle(train)
    rng.shuffle(test)
    ntr, nv, nt = cfg["train_size"], cfg["validation_size"], cfg["test_size"]
    if len(train) < ntr + nv or len(test) < nt:
        raise RuntimeError("GSM8K files do not contain enough isolated samples")
    result = {
        "seed": cfg["seed"],
        "train": [dict(id=f"train-{i:04d}", **x) for i, x in enumerate(train[:ntr])],
        "validation": [
            dict(id=f"validation-{i:04d}", **x) for i, x in enumerate(train[ntr : ntr + nv])
        ],
        "test": [dict(id=f"test-{i:04d}", **x) for i, x in enumerate(test[:nt])],
    }
    save_json(split_path, result)
    return result


def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path, device=True):
    kwargs = {
        "local_files_only": True,
        "torch_dtype": torch.float16,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if device:
        model.to(require_cuda())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def empty_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def inverse_rope(x, cos, sin):
    denom = (cos.square() + sin.square()).clamp_min(1e-12).unsqueeze(1)
    return (x * cos.unsqueeze(1) - rotate_half(x) * sin.unsqueeze(1)) / denom


def cache_to_cpu(cache):
    return [
        (layer.keys.detach().cpu().clone(), layer.values.detach().cpu().clone())
        for layer in cache.layers
    ]


def cache_from_tensors(items, config, device, requires_grad=False):
    data = []
    for k, v in items:
        k = k.to(device=device, dtype=torch.float16)
        v = v.to(device=device, dtype=torch.float16)
        if requires_grad:
            k.requires_grad_(True)
            v.requires_grad_(True)
        data.append((k, v))
    return DynamicCache(ddp_cache_data=data, config=config)


@torch.no_grad()
def gate0(cfg):
    progress("Gate 0: native cache injection and RoPE round-trip started")
    device = require_cuda()
    splits = make_splits(cfg)
    sample = splits["validation"][0]
    tok = load_tokenizer(cfg["receiver_model"])
    prompt = format_question(sample["question"])
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg["max_source_tokens"]).input_ids
    if ids.shape[1] < 2:
        raise RuntimeError("Gate 0 prompt is unexpectedly short")
    ids = ids.to(device)
    answer_ids = tok(sample["answer"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    model = load_model(cfg["receiver_model"])
    prefix = ids[:, :-1]
    prefix_mask = torch.ones_like(prefix)
    prefill = model(prefix, attention_mask=prefix_mask, use_cache=True)
    cpu_cache = cache_to_cpu(prefill.past_key_values)
    seq = torch.cat([ids, answer_ids[:, :-1]], dim=1)
    normal = model(seq, attention_mask=torch.ones_like(seq), use_cache=False).logits
    full_sequence_logits = normal[
        :, ids.shape[1] - 1 : ids.shape[1] - 1 + answer_ids.shape[1]
    ].float()
    normal_gen = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    normal_suffix = normal_gen[0, ids.shape[1] :].cpu()

    # The oracle reference must use the same incremental attention shape as an
    # externally supplied prefix. Comparing a one-shot full-sequence attention
    # kernel to an incremental cached kernel is not bitwise stable in V100 FP16.
    injected_input = torch.cat([ids[:, -1:], answer_ids[:, :-1]], dim=1)
    incremental_mask = torch.ones(
        (1, prefix.shape[1] + injected_input.shape[1]), device=device, dtype=torch.long
    )
    native_cache = cache_from_tensors(cpu_cache, model.config, device)
    native_logits = model(
        injected_input,
        attention_mask=incremental_mask,
        past_key_values=native_cache,
        use_cache=False,
    ).logits.float()
    native_generation_cache = cache_from_tensors(cpu_cache, model.config, device)
    native_gen_mask = torch.ones((1, prefix.shape[1] + 1), device=device, dtype=torch.long)
    native_gen = model.generate(
        ids[:, -1:],
        attention_mask=native_gen_mask,
        past_key_values=native_generation_cache,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    native_suffix = native_gen[0, 1:].cpu()
    del prefill, model
    empty_cuda()

    model = load_model(cfg["receiver_model"])
    injected_cache = cache_from_tensors(cpu_cache, model.config, device)
    injected = model(
        injected_input,
        attention_mask=incremental_mask,
        past_key_values=injected_cache,
        use_cache=False,
    ).logits.float()
    logit_diff = (native_logits - injected).abs()
    gold = answer_ids
    full_sequence_nll = F.cross_entropy(
        full_sequence_logits.reshape(-1, full_sequence_logits.shape[-1]), gold.reshape(-1)
    ).item()
    native_nll = F.cross_entropy(
        native_logits.reshape(-1, native_logits.shape[-1]), gold.reshape(-1)
    ).item()
    injected_nll = F.cross_entropy(injected.reshape(-1, injected.shape[-1]), gold.reshape(-1)).item()

    gen_cache = cache_from_tensors(cpu_cache, model.config, device)
    gen_mask = torch.ones((1, prefix.shape[1] + 1), device=device, dtype=torch.long)
    injected_gen = model.generate(
        ids[:, -1:],
        attention_mask=gen_mask,
        past_key_values=gen_cache,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    injected_suffix = injected_gen[0, 1:].cpu()

    layer = model.model.layers[0].self_attn
    hidden = model.model.embed_tokens(ids)
    pre_k = layer.k_norm(layer.k_proj(hidden).view(1, ids.shape[1], -1, cfg["head_dim"])).transpose(1, 2)
    pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, pos)
    round_trip = inverse_rope(apply_rope(pre_k, cos, sin), cos, sin)
    rope_mse = F.mse_loss(round_trip.float(), pre_k.float()).item()
    rope_cos = F.cosine_similarity(
        round_trip.float().reshape(-1, cfg["head_dim"]),
        pre_k.float().reshape(-1, cfg["head_dim"]),
        dim=-1,
    ).mean().item()

    top1_equal = bool(torch.equal(native_logits.argmax(-1).cpu(), injected.argmax(-1).cpu()))
    generation_equal = bool(
        torch.equal(native_suffix, injected_suffix) and torch.equal(normal_suffix, injected_suffix)
    )
    passed = (
        logit_diff.max().item() <= cfg["gate0_logit_atol"]
        and top1_equal
        and abs(native_nll - injected_nll) <= cfg["gate0_nll_atol"]
        and generation_equal
        and rope_cos >= cfg["gate0_rope_min_cosine"]
        and rope_mse <= cfg["gate0_rope_max_mse"]
    )
    report = {
        "passed": passed,
        "first_token_max_abs_logit_difference": logit_diff[:, 0].max().item(),
        "all_teacher_forced_max_abs_logit_difference": logit_diff.max().item(),
        "top1_equal": top1_equal,
        "full_sequence_teacher_forcing_nll": full_sequence_nll,
        "native_incremental_teacher_forcing_nll": native_nll,
        "injected_teacher_forcing_nll": injected_nll,
        "free_running_generation_equal": generation_equal,
        "rope_round_trip_cosine": rope_cos,
        "rope_round_trip_mse": rope_mse,
    }
    save_json(Path(cfg["work_dir"]) / "metrics" / "gate0.json", report)
    del model
    empty_cuda()
    if not passed:
        raise RuntimeError("Gate 0 failed; translator training is forbidden")
    progress("Gate 0: passed")
    return report


class KVHook:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.values = {}
        self.handles = []
        for index, block in enumerate(model.model.layers):
            self.handles.append(
                block.self_attn.register_forward_pre_hook(self._make(index), with_kwargs=True)
            )

    def _make(self, index):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            b, t, _ = hidden.shape
            shape = (b, t, self.cfg["num_kv_heads"], self.cfg["head_dim"])
            k = module.k_norm(module.k_proj(hidden).view(shape))[0].detach().cpu()
            v = module.v_proj(hidden).view(shape)[0].detach().cpu()
            self.values[index] = (k, v)
        return hook

    def clear(self):
        self.values.clear()

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def extract(cfg, role):
    assert role in {"sender", "receiver"}
    device = require_cuda()
    splits = make_splits(cfg)
    sender_tok = load_tokenizer(cfg["sender_model"])
    receiver_tok = load_tokenizer(cfg["receiver_model"])
    model_path = cfg["sender_model"] if role == "sender" else cfg["receiver_model"]
    tok = sender_tok if role == "sender" else receiver_tok
    model = load_model(model_path)
    hook = KVHook(model, cfg)
    root = Path(cfg["work_dir"]) / "kv" / role
    index_rows = []
    total_samples = sum(len(splits[x]) for x in ("train", "validation", "test"))
    completed_samples = 0
    progress(f"KV extraction ({role}): started for {total_samples} samples")
    for split in ("train", "validation", "test"):
        for sample in splits[split]:
            prompt = format_question(sample["question"])
            s = sender_tok(prompt, return_tensors="pt", truncation=True, max_length=cfg["max_source_tokens"])
            r = receiver_tok(prompt, return_tensors="pt", truncation=True, max_length=cfg["max_source_tokens"])
            if not torch.equal(s.input_ids, r.input_ids):
                raise RuntimeError(f"Sender/receiver token mismatch for {sample['id']}")
            if not torch.equal(s.attention_mask, r.attention_mask):
                raise RuntimeError(f"Sender/receiver padding mask mismatch for {sample['id']}")
            encoded = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg["max_source_tokens"])
            ids = encoded.input_ids.to(device)
            position_ids = torch.arange(ids.shape[1], device=device).unsqueeze(0)
            hook.clear()
            model(
                input_ids=ids,
                attention_mask=encoded.attention_mask.to(device),
                position_ids=position_ids,
                use_cache=False,
            )
            if len(hook.values) != cfg["num_layers"]:
                raise RuntimeError(f"Expected {cfg['num_layers']} hooked layers, got {len(hook.values)}")
            token_hash = sha_ids(encoded.input_ids[0])
            for layer_idx in range(cfg["num_layers"]):
                k, v = hook.values[layer_idx]
                if tuple(k.shape[1:]) != (cfg["num_kv_heads"], cfg["head_dim"]):
                    raise RuntimeError(f"Unexpected K layout {tuple(k.shape)}")
                for start in range(0, ids.shape[1], cfg["token_chunk_size"]):
                    end = min(ids.shape[1], start + cfg["token_chunk_size"])
                    rel = Path(split) / sample["id"] / f"layer_{layer_idx:02d}" / f"tokens_{start:04d}_{end:04d}.pt"
                    path = root / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "k": k[start:end].half().contiguous(),
                            "v": v[start:end].half().contiguous(),
                            "sample_id": sample["id"],
                            "layer": layer_idx,
                            "start": start,
                            "end": end,
                            "token_hash": token_hash,
                            "position_ids": torch.arange(start, end, dtype=torch.long),
                            "attention_mask": encoded.attention_mask[0, start:end].clone(),
                        },
                        path,
                    )
                    index_rows.append({"split": split, "relative_path": str(rel), "sample_id": sample["id"]})
            completed_samples += 1
            if completed_samples % 8 == 0 or completed_samples == total_samples:
                progress(f"KV extraction ({role}): {completed_samples}/{total_samples} samples")
    save_json(root / "index.json", index_rows)
    hook.close()
    del model
    empty_cuda()
    progress(f"KV extraction ({role}): completed")


class DenseKVTranslator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        l, g, d, h = (
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["translator_hidden_dim"],
        )
        self.cfg = cfg
        self.k_w1 = nn.Parameter(torch.empty(l, g, h, d))
        self.k_b1 = nn.Parameter(torch.empty(l, g, h))
        self.k_w2 = nn.Parameter(torch.empty(l, g, d, h))
        self.k_b2 = nn.Parameter(torch.empty(l, g, d))
        self.v_w1 = nn.Parameter(torch.empty(l, g, h, d))
        self.v_b1 = nn.Parameter(torch.empty(l, g, h))
        self.v_w2 = nn.Parameter(torch.empty(l, g, d, h))
        self.v_b2 = nn.Parameter(torch.empty(l, g, d))
        self.alpha = nn.Parameter(torch.zeros(l, g))
        self.reset_parameters()

    def reset_parameters(self):
        for w1, b1, w2, b2 in (
            (self.k_w1, self.k_b1, self.k_w2, self.k_b2),
            (self.v_w1, self.v_b1, self.v_w2, self.v_b2),
        ):
            for layer in range(self.cfg["num_layers"]):
                for group in range(self.cfg["num_kv_heads"]):
                    nn.init.kaiming_uniform_(w1[layer, group], a=math.sqrt(5))
                    nn.init.kaiming_uniform_(w2[layer, group], a=math.sqrt(5))
                    bound1 = 1 / math.sqrt(self.cfg["head_dim"])
                    bound2 = 1 / math.sqrt(self.cfg["translator_hidden_dim"])
                    nn.init.uniform_(b1[layer, group], -bound1, bound1)
                    nn.init.uniform_(b2[layer, group], -bound2, bound2)

    def _one(self, x, layer, w1, b1, w2, b2):
        # x: token × KV-group × head-dim; each group uses independent parameters.
        y = torch.einsum("tgd,ghd->tgh", x, w1[layer]) + b1[layer].unsqueeze(0)
        y = F.gelu(y)
        y = torch.einsum("tgh,gdh->tgd", y, w2[layer]) + b2[layer].unsqueeze(0)
        return y * torch.sigmoid(self.alpha[layer]).view(1, -1, 1)

    def forward_layer(self, k, v, layer):
        return (
            self._one(k, layer, self.k_w1, self.k_b1, self.k_w2, self.k_b2),
            self._one(v, layer, self.v_w1, self.v_b1, self.v_w2, self.v_b2),
        )


class PairDataset(Dataset):
    def __init__(self, cfg, split, limit_samples=None):
        root = Path(cfg["work_dir"]) / "kv"
        sender_rows = load_json(root / "sender" / "index.json")
        receiver_rows = load_json(root / "receiver" / "index.json")
        rmap = {x["relative_path"]: x for x in receiver_rows}
        rows = [x for x in sender_rows if x["split"] == split and x["relative_path"] in rmap]
        if limit_samples is not None:
            allowed = sorted({x["sample_id"] for x in rows})[:limit_samples]
            rows = [x for x in rows if x["sample_id"] in set(allowed)]
        self.rows, self.root = rows, root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rel = self.rows[idx]["relative_path"]
        s = torch.load(self.root / "sender" / rel, map_location="cpu", weights_only=True)
        r = torch.load(self.root / "receiver" / rel, map_location="cpu", weights_only=True)
        for key in ("sample_id", "layer", "start", "end", "token_hash"):
            if s[key] != r[key]:
                raise RuntimeError(f"KV pairing mismatch for {rel}: {key}")
        if not torch.equal(s["position_ids"], r["position_ids"]):
            raise RuntimeError(f"position_ids mismatch for {rel}")
        if not torch.equal(s["attention_mask"], r["attention_mask"]):
            raise RuntimeError(f"padding mask mismatch for {rel}")
        return s, r


def cosine_sum(pred, target):
    return F.cosine_similarity(pred.float().reshape(-1, pred.shape[-1]), target.float().reshape(-1, target.shape[-1]), dim=-1)


@torch.no_grad()
def reconstruction_metrics(model, dataset, cfg, device):
    model.eval()
    l, g = cfg["num_layers"], cfg["num_kv_heads"]
    k_sse = torch.zeros(l, g, dtype=torch.float64)
    v_sse = torch.zeros(l, g, dtype=torch.float64)
    k_cos = torch.zeros(l, g, dtype=torch.float64)
    v_cos = torch.zeros(l, g, dtype=torch.float64)
    elements = torch.zeros(l, g, dtype=torch.float64)
    vectors = torch.zeros(l, g, dtype=torch.float64)
    for s, r in DataLoader(dataset, batch_size=None, shuffle=False):
        layer = int(s["layer"])
        sk, sv = s["k"].to(device), s["v"].to(device)
        rk, rv = r["k"].to(device), r["v"].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            pk, pv = model.forward_layer(sk, sv, layer)
        diffk = (pk.float() - rk.float()).square().sum((0, 2)).cpu()
        diffv = (pv.float() - rv.float()).square().sum((0, 2)).cpu()
        k_sse[layer] += diffk
        v_sse[layer] += diffv
        k_cos[layer] += F.cosine_similarity(pk.float(), rk.float(), dim=-1).sum(0).cpu()
        v_cos[layer] += F.cosine_similarity(pv.float(), rv.float(), dim=-1).sum(0).cpu()
        elements[layer] += sk.shape[0] * sk.shape[-1]
        vectors[layer] += sk.shape[0]
    km = k_sse / elements.clamp_min(1)
    vm = v_sse / elements.clamp_min(1)
    kc = k_cos / vectors.clamp_min(1)
    vc = v_cos / vectors.clamp_min(1)
    return {
        "global_k_mse": k_sse.sum().item() / elements.sum().item(),
        "global_v_mse": v_sse.sum().item() / elements.sum().item(),
        "global_k_cosine": k_cos.sum().item() / vectors.sum().item(),
        "global_v_cosine": v_cos.sum().item() / vectors.sum().item(),
        "layer_group_k_mse": km.tolist(),
        "layer_group_v_mse": vm.tolist(),
        "layer_group_k_cosine": kc.tolist(),
        "layer_group_v_cosine": vc.tolist(),
        "per_layer_k_cosine": kc.mean(1).tolist(),
        "per_layer_v_cosine": vc.mean(1).tolist(),
        "per_group_k_cosine": kc.mean(0).tolist(),
        "per_group_v_cosine": vc.mean(0).tolist(),
        "gate": torch.sigmoid(model.alpha.detach()).cpu().tolist(),
    }


def save_heatmaps(metrics, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in ("layer_group_k_cosine", "layer_group_v_cosine", "layer_group_k_mse", "layer_group_v_mse"):
        fig, ax = plt.subplots(figsize=(10, 12))
        image = ax.imshow(metrics[name], aspect="auto", interpolation="nearest")
        ax.set_xlabel("KV group")
        ax.set_ylabel("layer")
        ax.set_title(name)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(str(out_prefix) + f"_{name}.png", dpi=160)
        plt.close(fig)


def save_checkpoint(path, model, extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "extra": extra}, path)


def load_translator(cfg, checkpoint=None, device=None):
    model = DenseKVTranslator(cfg)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
    # Keep FP32 master parameters. CUDA autocast performs the large MLP
    # operations in FP16, while GradScaler safely unscales FP32 gradients.
    return model.to(device or require_cuda())


def phase1_run(cfg, lr, tag, smoke=False):
    device = require_cuda()
    seed_all(cfg["seed"])
    model = load_translator(cfg, device=device)
    train = PairDataset(cfg, "train", cfg["smoke_train_size"] if smoke else cfg["train_size"])
    val = PairDataset(cfg, "validation")
    out = Path(cfg["work_dir"]) / "phase1" / tag
    out.mkdir(parents=True, exist_ok=True)
    progress(f"Phase I {tag}: initial validation pass started")
    initial = reconstruction_metrics(model, val, cfg, device)
    progress(f"Phase I {tag}: initial validation pass completed")
    history = [{"epoch": 0, "validation": initial}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["phase1_weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    epochs = cfg["phase1_smoke_epochs"] if smoke else cfg["phase1_epochs"]
    best_loss = initial["global_k_mse"] + initial["global_v_mse"]
    best_path = out / "best.pt"
    save_checkpoint(best_path, model, {"epoch": 0, "validation_loss": best_loss})
    for epoch in range(1, epochs + 1):
        progress(f"Phase I {tag}: epoch {epoch}/{epochs} training started")
        model.train()
        loader = DataLoader(train, batch_size=None, shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
        optimizer.zero_grad(set_to_none=True)
        total_loss, count = 0.0, 0
        for step, (s, r) in enumerate(loader, 1):
            layer = int(s["layer"])
            sk, sv = s["k"].to(device), s["v"].to(device)
            rk, rv = r["k"].to(device), r["v"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                pk, pv = model.forward_layer(sk, sv, layer)
                loss = F.mse_loss(pk, rk) + F.mse_loss(pv, rv)
                scaled = loss / cfg["phase1_grad_accumulation"]
            scaler.scale(scaled).backward()
            total_loss += loss.detach().float().item()
            count += 1
            if step % cfg["phase1_grad_accumulation"] == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["phase1_gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0 or step == len(loader):
                progress(f"Phase I {tag}: epoch {epoch}/{epochs}, chunk {step}/{len(loader)}")
        progress(f"Phase I {tag}: epoch {epoch}/{epochs} train/validation audit started")
        train_metrics = reconstruction_metrics(model, train, cfg, device)
        val_metrics = reconstruction_metrics(model, val, cfg, device)
        row = {"epoch": epoch, "mean_chunk_train_loss": total_loss / max(1, count), "train": train_metrics, "validation": val_metrics}
        history.append(row)
        save_json(out / "history.json", history)
        save_heatmaps(val_metrics, out / f"epoch_{epoch:02d}")
        score = val_metrics["global_k_mse"] + val_metrics["global_v_mse"]
        if score < best_loss:
            best_loss = score
            save_checkpoint(best_path, model, {"epoch": epoch, "validation_loss": best_loss})
        progress(f"Phase I {tag}: epoch {epoch}/{epochs} completed")
    final_train = history[-1]["train"]["global_k_mse"] + history[-1]["train"]["global_v_mse"]
    first_train = history[-1].get("mean_chunk_train_loss", float("inf"))
    initial_val = initial["global_k_mse"] + initial["global_v_mse"]
    summary = {
        "learning_rate": lr,
        "initial_validation_loss": initial_val,
        "best_validation_loss": best_loss,
        "validation_relative_improvement": (initial_val - best_loss) / max(initial_val, 1e-12),
        "final_train_reconstruction": final_train,
        "mean_chunk_train_loss": first_train,
        "best_checkpoint": str(best_path),
    }
    save_json(out / "summary.json", summary)
    del model, optimizer
    empty_cuda()
    progress(f"Phase I {tag}: completed")
    return summary


def translate_sample(cfg, translator, sample_id, split, receiver):
    sender_root = Path(cfg["work_dir"]) / "kv" / "sender" / split / sample_id
    items = []
    for layer in range(cfg["num_layers"]):
        files = sorted((sender_root / f"layer_{layer:02d}").glob("tokens_*.pt"))
        if not files:
            raise RuntimeError(f"Missing sender KV for {split}/{sample_id}/layer {layer}")
        ks, vs = [], []
        for path in files:
            x = torch.load(path, map_location="cpu", weights_only=True)
            ks.append(x["k"])
            vs.append(x["v"])
        k = torch.cat(ks).to(require_cuda())
        v = torch.cat(vs).to(require_cuda())
        with torch.autocast("cuda", dtype=torch.float16):
            pk, pv = translator.forward_layer(k, v, layer)
        pk = pk.transpose(0, 1).unsqueeze(0)
        pv = pv.transpose(0, 1).unsqueeze(0)
        pos = torch.arange(pk.shape[2], device=pk.device).unsqueeze(0)
        cos, sin = receiver.model.rotary_emb(pk, pos)
        receiver_dtype = receiver.model.layers[layer].self_attn.k_proj.weight.dtype
        # FP32 master parameters make the reliability gate promote the final
        # translator output to FP32. Native attention requires external K/V to
        # exactly match the frozen receiver projection dtype.
        items.append(
            (
                apply_rope(pk, cos, sin).to(dtype=receiver_dtype),
                pv.to(dtype=receiver_dtype),
            )
        )
    return items


def prompt_and_labels(tok, question, trace, aware, max_target):
    q = tok(format_question(question), add_special_tokens=False).input_ids if aware else []
    t = tok(trace, add_special_tokens=False, truncation=True, max_length=max_target).input_ids
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    current = [bos] + q + t[:-1]
    # Logit at the last question token (or BOS when unaware) predicts trace token 0.
    labels = [-100] * len(q) + t
    return torch.tensor([current], dtype=torch.long), torch.tensor([labels], dtype=torch.long)


def generate_traces(cfg):
    device = require_cuda()
    seed_all(cfg["seed"])
    splits = make_splits(cfg)
    tok = load_tokenizer(cfg["receiver_model"])
    model = load_model(cfg["receiver_model"])
    out = Path(cfg["work_dir"]) / "traces" / "receiver_self_traces.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if out.exists():
        completed = {x["id"]: x for x in read_jsonl(out)}
    with open(out, "a", encoding="utf-8") as f:
        progress(f"Receiver-self trace generation: started for {len(splits['train'])} samples")
        for sample_index, sample in enumerate(splits["train"], 1):
            if sample["id"] in completed:
                continue
            gold = parse_gold(sample["answer"])
            accepted = None
            attempts = []
            for attempt in range(cfg["trace_retries"]):
                prompt = (
                    "Solve the math problem in your own reasoning style. The verified numeric answer is "
                    f"{gold}. Produce a correct derivation and end with exactly '#### {gold}'.\n\n"
                    f"Question: {sample['question']}\n\nSolution:"
                )
                enc = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg["max_source_tokens"]).to(device)
                generation_args = {
                    "max_new_tokens": cfg["trace_max_new_tokens"],
                    "do_sample": attempt > 0,
                    "pad_token_id": tok.pad_token_id,
                }
                if attempt > 0:
                    generation_args.update(temperature=0.7, top_p=0.9)
                gen = model.generate(**enc, **generation_args)
                text = tok.decode(gen[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
                attempts.append(text)
                if parse_generated(text) == gold:
                    accepted = text
                    break
            row = {
                "id": sample["id"],
                "question": sample["question"],
                "gold_answer": gold,
                "receiver_self_trace": accepted,
                "accepted": accepted is not None,
                "attempt_count": len(attempts),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if sample_index % 8 == 0 or sample_index == len(splits["train"]):
                progress(f"Receiver-self trace generation: {sample_index}/{len(splits['train'])} samples")
    del model
    empty_cuda()
    progress("Receiver-self trace generation: completed")


def phase2(cfg, phase1_checkpoint):
    device = require_cuda()
    seed_all(cfg["seed"])
    splits = make_splits(cfg)
    sample_map = {x["id"]: x for x in splits["train"]}
    traces = [x for x in read_jsonl(Path(cfg["work_dir"]) / "traces" / "receiver_self_traces.jsonl") if x["accepted"]]
    if not traces:
        raise RuntimeError("No valid receiver-self trace; Phase II cannot start")
    translator = load_translator(cfg, phase1_checkpoint, device)
    receiver = load_model(cfg["receiver_model"])
    # Eval mode keeps the frozen receiver deterministic; gradients still flow to its cache inputs.
    receiver.eval()
    tok = load_tokenizer(cfg["receiver_model"])
    optimizer = torch.optim.AdamW(translator.parameters(), lr=cfg["phase2_learning_rate"], weight_decay=cfg["phase2_weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best = float("inf")
    out = Path(cfg["work_dir"]) / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])
    progress(f"Phase II: started for {cfg['phase2_steps']} optimization steps")
    for step in range(1, cfg["phase2_steps"] + 1):
        row = traces[(step - 1) % len(traces)]
        sample = sample_map[row["id"]]
        aware = rng.random() < cfg["phase2_context_aware_probability"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            cache_items = translate_sample(cfg, translator, row["id"], "train", receiver)
            cache = DynamicCache(ddp_cache_data=cache_items, config=receiver.config)
            current, labels = prompt_and_labels(
                tok, sample["question"], row["receiver_self_trace"], aware, cfg["phase2_max_target_tokens"]
            )
            current, labels = current.to(device), labels.to(device)
            prefix_len = cache.get_seq_length()
            mask = torch.ones((1, prefix_len + current.shape[1]), device=device, dtype=torch.long)
            output = receiver(
                input_ids=current,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=False,
            )
            logits = output.logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(translator.parameters(), cfg["phase2_gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        value = loss.detach().float().item()
        history.append({"step": step, "generation_ce": value, "context_aware": aware})
        if value < best:
            best = value
            save_checkpoint(out / "best.pt", translator, {"step": step, "generation_ce": value})
        if step % 10 == 0:
            save_json(out / "history.json", history)
            progress(f"Phase II: step {step}/{cfg['phase2_steps']}")
    save_json(out / "history.json", history)
    save_checkpoint(out / "final.pt", translator, {"step": cfg["phase2_steps"], "generation_ce": history[-1]["generation_ce"]})
    del receiver, translator, optimizer
    empty_cuda()
    progress("Phase II: completed")


@torch.no_grad()
def generate_with_cache(cfg, receiver, tok, question, cache_items=None):
    device = require_cuda()
    enc = tok(format_question(question), return_tensors="pt", truncation=True, max_length=cfg["max_source_tokens"]).to(device)
    if cache_items is None:
        output = receiver.generate(
            **enc, max_new_tokens=cfg["eval_max_new_tokens"], do_sample=False, pad_token_id=tok.pad_token_id
        )
        return tok.decode(output[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
    cache = DynamicCache(ddp_cache_data=cache_items, config=receiver.config)
    prefix_len = cache.get_seq_length()
    mask = torch.ones((1, prefix_len + enc.input_ids.shape[1]), device=device, dtype=torch.long)
    output = receiver.generate(
        input_ids=enc.input_ids,
        attention_mask=mask,
        past_key_values=cache,
        max_new_tokens=cfg["eval_max_new_tokens"],
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(output[0, enc.input_ids.shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def answer_nll_with_cache(cfg, receiver, tok, question, answer, cache_items=None):
    device = require_cuda()
    prompt = tok(
        format_question(question),
        add_special_tokens=False,
        truncation=True,
        max_length=cfg["max_source_tokens"],
    ).input_ids
    target = tok(answer, add_special_tokens=False).input_ids
    if not prompt or not target:
        return float("nan")
    sequence = torch.tensor([prompt + target[:-1]], dtype=torch.long, device=device)
    if cache_items is None:
        logits = receiver(sequence, use_cache=False).logits
    else:
        cache = DynamicCache(ddp_cache_data=cache_items, config=receiver.config)
        mask = torch.ones((1, cache.get_seq_length() + sequence.shape[1]), device=device, dtype=torch.long)
        logits = receiver(
            input_ids=sequence,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=False,
        ).logits
    selected = logits[:, len(prompt) - 1 : len(prompt) - 1 + len(target)].float()
    gold = torch.tensor(target, dtype=torch.long, device=device)
    return F.cross_entropy(selected.reshape(-1, selected.shape[-1]), gold).item()


@torch.no_grad()
def generation_sanity(cfg, phase1_checkpoint):
    device = require_cuda()
    splits = make_splits(cfg)
    tok = load_tokenizer(cfg["receiver_model"])
    receiver = load_model(cfg["receiver_model"])
    trained = load_translator(cfg, phase1_checkpoint, device)
    random_translator = load_translator(cfg, device=device)
    rows = []
    test = splits["test"]
    progress(f"Phase-I generation sanity: started for {len(test)} samples")
    for i, sample in enumerate(test):
        variants = {
            "receiver_only": None,
            "random_translator": translate_sample(cfg, random_translator, sample["id"], "test", receiver),
            "phase1_translator": translate_sample(cfg, trained, sample["id"], "test", receiver),
        }
        shuffled = translate_sample(cfg, trained, test[(i + 1) % len(test)]["id"], "test", receiver)
        variants["phase1_shuffled"] = shuffled
        outputs = {name: generate_with_cache(cfg, receiver, tok, sample["question"], cache) for name, cache in variants.items()}
        outputs["native_cache_oracle"] = outputs["receiver_only"]
        rows.append(
            {
                "id": sample["id"],
                "gold": parse_gold(sample["answer"]),
                "outputs": outputs,
                "parsed": {k: parse_generated(v) for k, v in outputs.items()},
            }
        )
        if (i + 1) % 4 == 0 or i + 1 == len(test):
            progress(f"Phase-I generation sanity: {i + 1}/{len(test)} samples")
    save_json(Path(cfg["work_dir"]) / "metrics" / "phase1_generation_sanity.json", rows)
    del receiver, trained, random_translator
    empty_cuda()
    progress("Phase-I generation sanity: completed")


@torch.no_grad()
def final_evaluation(cfg, phase1_checkpoint, phase2_checkpoint):
    device = require_cuda()
    splits = make_splits(cfg)
    tok = load_tokenizer(cfg["receiver_model"])
    receiver = load_model(cfg["receiver_model"])
    random_t = load_translator(cfg, device=device)
    phase1_t = load_translator(cfg, phase1_checkpoint, device)
    phase2_t = load_translator(cfg, phase2_checkpoint, device)
    test = splits["test"]
    rows = []
    correct = {k: 0 for k in ("receiver_only", "random", "phase1", "phase2", "phase2_shuffled")}
    nll_sum = {k: 0.0 for k in correct}
    progress(f"Final evaluation: started for {len(test)} samples")
    for i, sample in enumerate(test):
        caches = {
            "receiver_only": None,
            "random": translate_sample(cfg, random_t, sample["id"], "test", receiver),
            "phase1": translate_sample(cfg, phase1_t, sample["id"], "test", receiver),
            "phase2": translate_sample(cfg, phase2_t, sample["id"], "test", receiver),
            "phase2_shuffled": translate_sample(cfg, phase2_t, test[(i + 1) % len(test)]["id"], "test", receiver),
        }
        gold = parse_gold(sample["answer"])
        outputs, parsed = {}, {}
        for name, cache in caches.items():
            outputs[name] = generate_with_cache(cfg, receiver, tok, sample["question"], cache)
            parsed[name] = parse_generated(outputs[name])
            correct[name] += int(parsed[name] == gold)
            nll_sum[name] += answer_nll_with_cache(
                cfg, receiver, tok, sample["question"], sample["answer"], cache
            )
        # Native-cache oracle is audited exactly in Gate 0 and is receiver-only semantics here.
        outputs["native_cache_oracle"] = outputs["receiver_only"]
        parsed["native_cache_oracle"] = parsed["receiver_only"]
        rows.append({"id": sample["id"], "gold": gold, "outputs": outputs, "parsed": parsed})
        if (i + 1) % 4 == 0 or i + 1 == len(test):
            progress(f"Final evaluation: {i + 1}/{len(test)} samples")
    n = len(test)
    summary = {
        "accuracy": {k: v / n for k, v in correct.items()},
        "teacher_forcing_answer_nll": {k: v / n for k, v in nll_sum.items()},
        "native_cache_oracle_accuracy": correct["receiver_only"] / n,
        "native_cache_oracle_teacher_forcing_answer_nll": nll_sum["receiver_only"] / n,
        "phase2_minus_phase1_accuracy": (correct["phase2"] - correct["phase1"]) / n,
        "phase2_correct_minus_shuffled_accuracy": (correct["phase2"] - correct["phase2_shuffled"]) / n,
        "full_outputs": "metrics/final_outputs.json",
    }
    save_json(Path(cfg["work_dir"]) / "metrics" / "final_outputs.json", rows)
    save_json(Path(cfg["work_dir"]) / "metrics" / "final_summary.json", summary)
    del receiver, random_t, phase1_t, phase2_t
    empty_cuda()
    progress("Final evaluation: completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["split", "gate0", "extract_sender", "extract_receiver", "traces"])
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    if args.command == "split":
        make_splits(cfg)
    elif args.command == "gate0":
        gate0(cfg)
    elif args.command == "extract_sender":
        extract(cfg, "sender")
    elif args.command == "extract_receiver":
        extract(cfg, "receiver")
    elif args.command == "traces":
        generate_traces(cfg)


if __name__ == "__main__":
    main()
