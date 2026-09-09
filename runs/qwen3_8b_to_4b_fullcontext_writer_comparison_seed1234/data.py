from __future__ import annotations

import json
import random
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


SYSTEM = (
    "You are a question answering system. Answer the question using only the provided context. "
    "Return only the shortest answer span. Do not explain your reasoning. "
    'If the answer is yes or no, return exactly "yes" or "no".'
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def cuda():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if not tok.is_fast: raise RuntimeError("fast tokenizer required")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    return tok


def load_model(path, cfg):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg["receiver_dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype,
        attn_implementation=cfg["attention_implementation"],
    ).to(cuda()).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def user_text(raw, context=True):
    parts = []
    if context:
        parts.append("Context:")
        for index, (title, sentences) in enumerate(raw["context"], 1):
            parts.append(f"[{index}] {title}\n{''.join(sentences)}")
    parts.append(f"Question: {raw['question']}\n\nAnswer:")
    return "\n\n".join(parts)


def render(tok, raw, context=True):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(raw, context)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    official = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    official_ids = official.input_ids if hasattr(official, "input_ids") else official
    ids = list(encoded.input_ids)
    if ids != list(official_ids): raise RuntimeError("chat-template tokenization mismatch")
    if not context: return {"question_only_ids": ids}
    marker = f"Question: {raw['question']}\n\nAnswer:"
    start = text.rfind(marker)
    if start < 0: raise RuntimeError("question boundary missing")
    char_index = start + len("Question: ")
    boundary = next(i for i, (_, end) in enumerate(encoded.offset_mapping) if end > char_index)
    return {
        "full_input_ids": ids, "context_input_ids": ids[:boundary],
        "question_suffix_ids": ids[boundary:], "context_end_index": boundary,
        "boundary_token_offset": list(encoded.offset_mapping[boundary]),
    }


def stratified(rows, count, rng):
    groups = {"bridge": [], "comparison": []}
    for row in rows:
        if row.get("type") in groups: groups[row["type"]].append(row)
    for values in groups.values(): rng.shuffle(values)
    half = count // 2
    chosen = groups["bridge"][:half] + groups["comparison"][:count-half]
    rng.shuffle(chosen)
    return chosen


def shuffle_derangement(samples):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    ordered = sorted(samples, key=lambda x: (len(x["context_input_ids"]), x["id"]))
    count = len(ordered)
    cost = np.empty((count, count), dtype=np.float64)
    forbidden = 1e12
    for i, sample in enumerate(ordered):
        for j, donor in enumerate(ordered):
            invalid = sample["id"] == donor["id"] or normalize_answer(sample["answer"]) == normalize_answer(donor["answer"])
            cost[i, j] = forbidden if invalid else abs(len(sample["context_input_ids"]) - len(donor["context_input_ids"]))
    sources, donors = linear_sum_assignment(cost)
    if len(sources) != count or any(cost[i, j] >= forbidden for i, j in zip(sources, donors)):
        raise RuntimeError("cannot construct answer-different length-aware derangement")
    return {ordered[i]["id"]: ordered[j]["id"] for i, j in zip(sources, donors)}


def prepare_manifest(cfg):
    tok4, tok8 = tokenizer(cfg["model_4b"]), tokenizer(cfg["model_8b"])
    train_raw, dev_raw = load_json(cfg["hotpot_train"]), load_json(cfg["hotpot_dev"])
    dev_map = {row["_id"]: row for row in dev_raw}
    frozen_test = read_jsonl(Path(cfg["audit_4b_dir"]) / "artifacts" / "rendered_samples.jsonl")
    test_ids = [row["id"] for row in frozen_test]
    rng = random.Random(cfg["seed"])
    train = stratified(train_raw, cfg["sizes"]["train"], rng)
    remaining = [row for row in dev_raw if row["_id"] not in set(test_ids)]
    validation = stratified(remaining, cfg["sizes"]["validation"], rng)
    test = [dev_map[sample_id] for sample_id in test_ids]
    output = {}
    for split, raws in (("train", train), ("validation", validation), ("test", test)):
        rows = []
        for raw in raws:
            four, eight = render(tok4, raw, True), render(tok8, raw, True)
            if four["full_input_ids"] != eight["full_input_ids"] or four["context_input_ids"] != eight["context_input_ids"]:
                raise RuntimeError(f"4B/8B token mismatch: {raw['_id']}")
            qonly = render(tok4, raw, False)
            target = tok4(raw["answer"], add_special_tokens=False).input_ids[:cfg["max_answer_tokens"]]
            target = (target or [tok4.eos_token_id]) + [tok4.eos_token_id]
            rows.append({
                "id": raw["_id"], "type": raw.get("type", "unknown"),
                "answer": raw["answer"], "question": raw["question"],
                **four, **qonly, "answer_token_ids": target,
                "context_length": len(four["context_input_ids"]),
            })
        mapping = shuffle_derangement(rows)
        for row in rows: row["shuffle_id"] = mapping[row["id"]]
        output[split] = rows
    save_json(Path(cfg["work_dir"]) / "artifacts" / "manifest.json", output)
    save_json(Path(cfg["work_dir"]) / "artifacts" / "protocol.json", {
        "seed": cfg["seed"], "sizes": cfg["sizes"], "full_context": True,
        "supporting_fact_selection": False, "same_4b_8b_tokens": True,
        "test_ids_equal_frozen_cache_audits": [x["id"] for x in output["test"]] == test_ids,
        "reader": "Qwen3-4B Identity", "writer_question_aware": False,
        "prompt": SYSTEM, "thinking": False, "max_new_tokens": cfg["max_new_tokens"],
    })
    progress("full-context manifest frozen")


def rows_for(cfg, mode):
    manifest = load_json(Path(cfg["work_dir"]) / "artifacts" / "manifest.json")
    sizes = cfg["smoke_sizes"] if mode == "smoke" else cfg["sizes"]
    return {split: rows[:sizes[split]] for split, rows in manifest.items()}


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
        if len(self.pre) != layers: raise RuntimeError(f"captured {len(self.pre)}/{layers} layers")
        return torch.stack([self.pre[i] for i in range(layers)]), torch.stack([self.value[i] for i in range(layers)])

    def close(self):
        for handle in self.handles: handle.remove()


def capture_native(model, ids, layers):
    capture = NativeCapture(model)
    tensor = torch.tensor([ids], dtype=torch.long, device=cuda())
    positions = torch.arange(len(ids), device=cuda()).unsqueeze(0)
    with torch.no_grad(): model(tensor, attention_mask=torch.ones_like(tensor), position_ids=positions, use_cache=False)
    key, value = capture.get(layers); capture.close()
    return key, value


class Store:
    def __init__(self, cfg, mode, rows):
        self.cfg, self.mode, self.rows = cfg, mode, rows
        self.maps = {split: {x["id"]: x for x in values} for split, values in rows.items()}
        self.cache = {}

    def _load(self, key, path):
        if key not in self.cache:
            if len(self.cache) > 3: self.cache.clear()
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.cache[key]

    def source(self, split, sample_id):
        path = Path(self.cfg["work_dir"]) / "cache" / self.mode / "source8" / split / f"{sample_id}.pt"
        return self._load(("source", split, sample_id), path)

    def target(self, split, sample_id):
        path = Path(self.cfg["work_dir"]) / "cache" / self.mode / "target4" / split / f"{sample_id}.pt"
        return self._load(("target", split, sample_id), path)

    def teacher(self, split, sample_id):
        path = Path(self.cfg["work_dir"]) / "cache" / self.mode / "teacher4" / split / f"{sample_id}.pt"
        return self._load(("teacher", split, sample_id), path)


def normalize_answer(text):
    value = text.lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def token_f1(prediction, answer):
    p, a = normalize_answer(prediction).split(), normalize_answer(answer).split()
    if not p or not a: return float(p == a)
    same = sum((Counter(p) & Counter(a)).values())
    if not same: return 0.0
    precision, recall = same/len(p), same/len(a)
    return 2*precision*recall/(precision+recall)


def representation_loss(pred_k, pred_v, target_k, target_v, cosine_weight):
    rows = []
    for p, g, name in ((pred_k, target_k, "k"), (pred_v, target_v, "v")):
        pf, gf = p.float(), g.float()
        nmse = (pf-gf).square().mean(dim=(1,2,3)) / gf.square().mean(dim=(1,2,3)).clamp_min(1e-8)
        cosine = F.cosine_similarity(pf.flatten(1), gf.flatten(1), 1)
        rows.append((name, nmse, cosine))
    loss = rows[0][1].mean()+rows[1][1].mean()+cosine_weight*((1-rows[0][2]).mean()+(1-rows[1][2]).mean())
    metrics = [{
        "layer": layer, "k_nmse": rows[0][1][layer].item(), "v_nmse": rows[1][1][layer].item(),
        "k_cosine": rows[0][2][layer].item(), "v_cosine": rows[1][2][layer].item(),
    } for layer in range(pred_k.shape[0])]
    return loss, metrics
