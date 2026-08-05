"""数据协议与核心工具（跨模型 KV 规模因子实验）。

实现「统一数据协议」（实验方案 §三）：
- Sender 输入 = context-only 序列（系统 + Context，无 Question）
- Receiver 输入 = question 前缀（从 "Question: ..." 起，含 Answer 前缀）
- KV 注入 = Sender context-only KV（pre-RoPE K + V）经 Writer 翻译后，
  用 Receiver 的 rotary_emb 重新旋转，构造 DynamicCache。

复用已验证的 8B→4B 机制：
- NativeCapture：在 self_attn forward 前用 k_norm 提取 pre-RoPE K（k_norm 之后），v_proj 提取 V
- dynamic_cache：pre-RoPE K → 应用 Receiver RoPE → DynamicCache
- prepare_manifest：跨模型 tokenizer 一致性检查 + Bridge/Comparison 分层采样 + shuffled donor 分配
"""

from __future__ import annotations

import json
import random
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


# ---------------------------------------------------------------------------
# Prompt 协议（方案 §三）
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are a question answering system. Answer the question using only the provided context. "
    "Return only the shortest answer span. Do not explain your reasoning. "
    'If the answer is yes or no, return exactly "yes" or "no".'
)


def user_text(raw, context=True):
    """构造 user 消息。context=True 时含完整 10 段 Context；context=False 时只含 Question。"""
    parts = []
    if context:
        parts.append("Context:")
        for index, (title, sentences) in enumerate(raw["context"], 1):
            parts.append(f"[{index}] {title}\n{''.join(sentences)}")
    parts.append(f"Question: {raw['question']}\n\nAnswer:")
    return "\n\n".join(parts)


def render(tok, raw, context=True):
    """用官方 chat template + enable_thinking=False 渲染。

    返回：
      full_input_ids     完整序列（context + question，含 system）
      context_input_ids  context 部分 ids（Sender 的输入）
      question_suffix_ids 从 "Question: " 起的 ids（Receiver 的 prompt，注入 KV 后 append）
      context_end_index  context 边界
    用 offset mapping 精确定位 "Question: " 文本边界。
    """
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(raw, context)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    official = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    official_ids = official.input_ids if hasattr(official, "input_ids") else official
    ids = list(encoded.input_ids)
    if ids != list(official_ids):
        raise RuntimeError("chat-template tokenization mismatch")
    if not context:
        return {"question_only_ids": ids}
    marker = f"Question: {raw['question']}\n\nAnswer:"
    start = text.rfind(marker)
    if start < 0:
        raise RuntimeError("question boundary missing")
    char_index = start + len("Question: ")
    boundary = next(i for i, (_, end) in enumerate(encoded.offset_mapping) if end > char_index)
    return {
        "full_input_ids": ids,
        "context_input_ids": ids[:boundary],
        "question_suffix_ids": ids[boundary:],
        "context_end_index": boundary,
        "boundary_token_offset": list(encoded.offset_mapping[boundary]),
    }


# ---------------------------------------------------------------------------
# 数据加载 / manifest（方案 §三 数据规模）
# ---------------------------------------------------------------------------
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError("fast tokenizer required")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path, cfg):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg["receiver_dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation=cfg.get("attention_implementation", "eager"),
    ).to(cuda()).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def stratified(rows, count, rng):
    """Bridge/Comparison 各取一半（方案 §三：Bridge 和 Comparison 各占一半）。"""
    groups = {"bridge": [], "comparison": []}
    for row in rows:
        if row.get("type") in groups:
            groups[row["type"]].append(row)
    for values in groups.values():
        rng.shuffle(values)
    half = count // 2
    chosen = groups["bridge"][:half] + groups["comparison"][:count - half]
    rng.shuffle(chosen)
    return chosen


def normalize_answer(text):
    value = text.lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def token_f1(prediction, answer):
    p, a = normalize_answer(prediction).split(), normalize_answer(answer).split()
    if not p or not a:
        return float(p == a)
    same = sum((Counter(p) & Counter(a)).values())
    if not same:
        return 0.0
    precision, recall = same / len(p), same / len(a)
    return 2 * precision * recall / (precision + recall)


def shuffle_derangement(samples):
    """为 shuffled 条件分配 donor：Bridge↔Bridge、Comparison↔Comparison、
    答案不同、context 长度尽量匹配（方案 §十一.2）。确定性（匈牙利算法）。
    """
    ordered = sorted(samples, key=lambda x: (len(x["context_input_ids"]), x["id"]))
    count = len(ordered)
    cost = np.empty((count, count), dtype=np.float64)
    forbidden = 1e12
    for i, sample in enumerate(ordered):
        for j, donor in enumerate(ordered):
            invalid = (
                sample["id"] == donor["id"]
                or sample["type"] != donor["type"]
                or normalize_answer(sample["answer"]) == normalize_answer(donor["answer"])
            )
            cost[i, j] = forbidden if invalid else abs(len(sample["context_input_ids"]) - len(donor["context_input_ids"]))
    sources, donors = linear_sum_assignment(cost)
    if len(sources) != count or any(cost[i, j] >= forbidden for i, j in zip(sources, donors)):
        raise RuntimeError("cannot construct answer-different type-matched length-aware derangement")
    return {ordered[i]["id"]: ordered[j]["id"] for i, j in zip(sources, donors)}


def prepare_manifest(cfg):
    """构建 manifest（方案 §三）。sender_path / receiver_path 两个模型必须对同一文本
    tokenize 出完全一致的 ids（方案 §0 跨模型 tokenizer 一致性），否则 raise。
    """
    sender_tok = tokenizer(cfg["sender_path"])
    receiver_tok = tokenizer(cfg["receiver_path"])
    train_raw = load_json(cfg["hotpot_train"])
    dev_raw = load_json(cfg["hotpot_dev"])
    dev_map = {row["_id"]: row for row in dev_raw}
    rng = random.Random(cfg["seed"])
    train = stratified(train_raw, cfg["sizes"]["train"], rng)
    # test 从 dev 中取（方案 §三：test 与 train 不相交）
    dev_pool = [row for row in dev_raw]
    rng.shuffle(dev_pool)
    test = stratified(dev_pool, cfg["sizes"]["test"], rng)
    test_ids = {row["_id"] for row in test}
    remaining = [row for row in dev_pool if row["_id"] not in test_ids]
    validation = stratified(remaining, cfg["sizes"]["validation"], rng)

    output = {}
    for split, raws in (("train", train), ("validation", validation), ("test", test)):
        rows = []
        for raw in raws:
            sender = render(sender_tok, raw, True)
            receiver = render(receiver_tok, raw, True)
            if sender["full_input_ids"] != receiver["full_input_ids"]:
                raise RuntimeError(f"full_input_ids mismatch: {raw['_id']}")
            if sender["context_input_ids"] != receiver["context_input_ids"]:
                raise RuntimeError(f"context_input_ids mismatch: {raw['_id']}")
            qonly = render(receiver_tok, raw, False)
            target = receiver_tok(raw["answer"], add_special_tokens=False).input_ids[: cfg["max_answer_tokens"]]
            target = (target or [receiver_tok.eos_token_id]) + [receiver_tok.eos_token_id]
            rows.append({
                "id": raw["_id"],
                "type": raw.get("type", "unknown"),
                "answer": raw["answer"],
                "question": raw["question"],
                **sender,
                **qonly,
                "answer_token_ids": target,
                "context_length": len(sender["context_input_ids"]),
            })
        mapping = shuffle_derangement(rows)
        for row in rows:
            row["shuffle_id"] = mapping[row["id"]]
        output[split] = rows
    save_json(Path(cfg["work_dir"]) / "artifacts" / "manifest.json", output)
    save_json(Path(cfg["work_dir"]) / "artifacts" / "protocol.json", {
        "seed": cfg["seed"],
        "sizes": cfg["sizes"],
        "full_context": True,
        "sender": cfg["sender_path"],
        "receiver": cfg["receiver_path"],
        "sender_question_aware": False,
        "thinking": False,
        "max_new_tokens": cfg["max_new_tokens"],
        "prompt": SYSTEM,
    })
    progress("manifest frozen")


def rows_for(cfg, mode):
    manifest = load_json(Path(cfg["work_dir"]) / "artifacts" / "manifest.json")
    sizes = cfg["smoke_sizes"] if mode == "smoke" else cfg["sizes"]
    return {split: rows[: sizes[split]] for split, rows in manifest.items()}


# ---------------------------------------------------------------------------
# pre-RoPE KV 提取（方案 §二：pre-RoPE K，位置在 k_norm 之后）
# ---------------------------------------------------------------------------
class NativeCapture:
    def __init__(self, model):
        self.pre, self.value, self.handles = {}, {}, []
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._hook(index), with_kwargs=True))

    def _hook(self, index):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:2], -1, module.head_dim)
            key = module.k_norm(module.k_proj(hidden).view(shape))
            value = module.v_proj(hidden).view(shape)
            self.pre[index] = key[0].detach().cpu().half()
            self.value[index] = value[0].detach().cpu().half()
        return hook

    def get(self, layers):
        if len(self.pre) != layers:
            raise RuntimeError(f"captured {len(self.pre)}/{layers} layers")
        return torch.stack([self.pre[i] for i in range(layers)]), torch.stack([self.value[i] for i in range(layers)])

    def close(self):
        for handle in self.handles:
            handle.remove()


def capture_native(model, ids, layers):
    """提取 context-only 输入的 pre-RoPE K 和 native V，形状 [L, T, 8, 128]，FP16。"""
    capture = NativeCapture(model)
    tensor = torch.tensor([ids], dtype=torch.long, device=cuda())
    positions = torch.arange(len(ids), device=cuda()).unsqueeze(0)
    with torch.no_grad():
        model(tensor, attention_mask=torch.ones_like(tensor), position_ids=positions, use_cache=False)
    key, value = capture.get(layers)
    capture.close()
    return key, value


# ---------------------------------------------------------------------------
# KV 注入（方案 §七：翻译后恢复 Receiver 尺度，再应用 Receiver 的 RoPE）
# ---------------------------------------------------------------------------
def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), -1)


def apply_rope(model, tensor, positions):
    position_ids = torch.tensor([positions], dtype=torch.long, device=tensor.device)
    dummy = torch.empty(1, len(positions), model.config.hidden_size, dtype=tensor.dtype, device=tensor.device)
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    return tensor * cos[0][None, :, None, :] + rotate_half(tensor) * sin[0][None, :, None, :]


def dynamic_cache(model, pre_key, value, positions=None):
    """把 pre-RoPE K + V 注入 Receiver：K 先经 Receiver 自己的 RoPE 旋转（方案 §七 恢复后）。"""
    device = next(model.parameters()).device
    pre_key = pre_key.to(device)
    value = value.to(device)
    positions = list(range(pre_key.shape[1])) if positions is None else positions
    post_key = apply_rope(model, pre_key, positions)
    items = [
        (post_key[layer].permute(1, 0, 2).unsqueeze(0), value[layer].permute(1, 0, 2).unsqueeze(0))
        for layer in range(pre_key.shape[0])
    ]
    return DynamicCache(ddp_cache_data=items, config=model.config)


def _resolve_prompt(sample, prompt_kind, prompt_ids):
    if prompt_ids is not None:
        return prompt_ids
    if prompt_kind == "suffix":
        return sample["question_suffix_ids"]
    if prompt_kind == "qonly":
        return sample["question_only_ids"]
    if prompt_kind == "full":
        return sample["full_input_ids"]
    raise ValueError(prompt_kind)


def answer_logits(model, sample, key=None, value=None, prompt_kind="suffix", prompt_ids=None):
    prompt = _resolve_prompt(sample, prompt_kind, prompt_ids)
    target = sample["answer_token_ids"]
    current = prompt + target[:-1]
    prefix = 0 if key is None else key.shape[1]
    ids = torch.tensor([current], dtype=torch.long, device=cuda())
    positions = torch.arange(prefix, prefix + len(current), device=cuda()).unsqueeze(0)
    mask = torch.ones(1, prefix + len(current), dtype=torch.long, device=cuda())
    kwargs = {}
    if key is not None:
        kwargs["past_key_values"] = dynamic_cache(model, key, value)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, **kwargs)
    start = len(prompt) - 1
    logits = output.logits[0, start:start + len(target)].float()
    gold = torch.tensor(target, dtype=torch.long, device=cuda())
    return logits, gold


def ce_loss(model, sample, key, value):
    logits, gold = answer_logits(model, sample, key, value)
    return F.cross_entropy(logits, gold), logits, gold


@torch.no_grad()
def generate(model, tok, sample, cfg, key=None, value=None, prompt_kind="suffix", prompt_ids=None):
    """自由生成（方案 §十一/评估统一用 max_new_tokens=32、do_sample=False）。"""
    prompt = _resolve_prompt(sample, prompt_kind, prompt_ids)
    prefix = 0 if key is None else key.shape[1]
    ids = torch.tensor([prompt], dtype=torch.long, device=cuda())
    positions = torch.arange(prefix, prefix + len(prompt), device=cuda()).unsqueeze(0)
    mask = torch.ones(1, prefix + len(prompt), dtype=torch.long, device=cuda())
    kwargs = {}
    if key is not None:
        kwargs["past_key_values"] = dynamic_cache(model, key, value)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True, **kwargs)
    past, next_token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True)
    generated = []
    next_position = prefix + len(prompt)
    for _ in range(cfg["max_new_tokens"]):
        token = int(next_token.item())
        if token == tok.eos_token_id:
            break
        generated.append(token)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=cuda())], 1)
        output = model(
            input_ids=next_token,
            attention_mask=mask,
            position_ids=torch.tensor([[next_position]], device=cuda()),
            past_key_values=past,
            use_cache=True,
        )
        past, next_token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    return tok.decode(generated, skip_special_tokens=True).strip(), len(generated)


# ---------------------------------------------------------------------------
# 表示对齐损失（方案 §十 路径 F2 Stage A：L_A = L_K,NMSE + L_V,NMSE + 0.25(L_K,cos + L_V,cos)）
# ---------------------------------------------------------------------------
def representation_loss(pred_k, pred_v, target_k, target_v, cosine_weight):
    rows = []
    for p, g, name in ((pred_k, target_k, "k"), (pred_v, target_v, "v")):
        pf, gf = p.float(), g.float()
        nmse = (pf - gf).square().mean(dim=(1, 2, 3)) / gf.square().mean(dim=(1, 2, 3)).clamp_min(1e-8)
        cosine = F.cosine_similarity(pf.flatten(1), gf.flatten(1), 1)
        rows.append((name, nmse, cosine))
    loss = rows[0][1].mean() + rows[1][1].mean() + cosine_weight * ((1 - rows[0][2]).mean() + (1 - rows[1][2]).mean())
    metrics = [{
        "layer": layer,
        "k_nmse": rows[0][1][layer].item(),
        "v_nmse": rows[1][1][layer].item(),
        "k_cosine": rows[0][2][layer].item(),
        "v_cosine": rows[1][2][layer].item(),
    } for layer in range(pred_k.shape[0])]
    return loss, metrics


def sampled_positions(length, count):
    """Stage A 每样本固定采样 128 个 token 位置（方案 §十）。"""
    if length < count:
        raise RuntimeError(f"context length {length} < sampled tokens {count}")
    values = [(index * (length - 1)) // (count - 1) for index in range(count)]
    if len(set(values)) != count:
        raise RuntimeError("uniform token sampling produced duplicates")
    return values


# ---------------------------------------------------------------------------
# KV 资产 Store（方案 §十六 cache/ 布局）
# ---------------------------------------------------------------------------
class Store:
    """读取缓存的 KV 资产。source = Sender context KV，target = Receiver context KV。"""

    def __init__(self, cfg, mode, rows):
        self.cfg, self.mode, self.rows = cfg, mode, rows
        self.maps = {split: {x["id"]: x for x in values} for split, values in rows.items()}
        self.cache = {}

    def _load(self, key, path):
        if key not in self.cache:
            if len(self.cache) > 3:
                self.cache.clear()
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.cache[key]

    def source(self, split, sample_id):
        path = Path(self.cfg["work_dir"]) / "cache" / self.mode / self.cfg["source_dir"] / split / f"{sample_id}.pt"
        return self._load(("source", split, sample_id), path)

    def target(self, split, sample_id):
        path = Path(self.cfg["work_dir"]) / "cache" / self.mode / self.cfg["target_dir"] / split / f"{sample_id}.pt"
        return self._load(("target", split, sample_id), path)


def scales(cfg, mode, rows, direction="main"):
    """统计 Sender/Receiver 逐层、逐 feature 的 RMS scale（方案 §二）：s ∈ R^{L×1024}。

    按 direction 分别存放（self_06 与 06_to_17 的 source/target 组合不同）。
    """
    path = Path(cfg["work_dir"]) / "artifacts" / mode / direction / "scales.pt"
    if path.exists():
        return
    store = Store(cfg, mode, rows)
    sums = {
        name: torch.zeros(cfg["num_layers"], cfg["feature_dim"], dtype=torch.float64)
        for name in ("source_k", "source_v", "target_k", "target_v")
    }
    count = 0
    for index, sample in enumerate(rows["train"], 1):
        source = store.source("train", sample["id"])
        target = store.target("train", sample["id"])
        positions = sampled_positions(target["pre_key"].shape[1], cfg["sampled_tokens"])
        values = {
            "source_k": source["pre_key"][:, positions].float().flatten(2),
            "source_v": source["value"][:, positions].float().flatten(2),
            "target_k": target["pre_key"][:, positions].float().flatten(2),
            "target_v": target["value"][:, positions].float().flatten(2),
        }
        for name, value in values.items():
            sums[name] += value.double().square().sum(1)
        count += len(positions)
        if index % 32 == 0 or index == len(rows["train"]):
            progress(f"{mode}: RMS {index}/{len(rows['train'])}")
    output = {name: (value / count + 1e-6).sqrt().float() for name, value in sums.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    save_json(path.with_suffix(".json"), {
        "count_tokens": count,
        "shape": [cfg["num_layers"], cfg["feature_dim"]],
        "shared_by_all_writers": True,
    })
