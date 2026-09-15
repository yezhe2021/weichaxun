from __future__ import annotations

import csv
import importlib
import json
import math
import random
import re
import string
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from receiver_anchor_injection import memory_dict


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def source_mode(mode):
    return "smoke" if mode == "smoke" else "formal"


def query_mode(mode):
    return "smoke" if mode == "smoke" else "development"


def r1_module(cfg):
    path = cfg["r1_dir"]
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("r1_common")


def rows_for(cfg, mode):
    rows = load_json(Path(cfg["r1_dir"]) / "artifacts" / source_mode(mode) / "manifest.json")
    sizes = cfg["smoke_sizes"] if mode == "smoke" else cfg["sizes"]
    output = {split: [dict(x) for x in rows[split][:sizes[split]]] for split in sizes}
    for split, samples in output.items():
        for sample in samples:
            sample["selected_position_ids"] = sample["selected_position_ids"][:cfg["max_evidence_tokens"]]
            sample["selected_token_count"] = len(sample["selected_position_ids"])
        for index, sample in enumerate(samples):
            candidates = [
                x for x in samples
                if x["id"] != sample["id"]
                and x["answer"].strip().lower() != sample["answer"].strip().lower()
            ]
            sample["shuffle_id"] = candidates[(17 * index + 7) % len(candidates)]["id"]
    return output


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
            shard_size = self.cfg["q35_shard_size"]
            path = (
                Path(self.cfg["v2_dir"]) / "cache" / self.mode / split / "q35"
                / f"shard_{index // shard_size:05d}.pt"
            )
        else:
            shard_size = self.cfg["native4_shard_size"]
            path = (
                Path(self.cfg["r1_dir"]) / "cache" / source_mode(self.mode) / split / "4b"
                / f"shard_{index // shard_size:05d}.pt"
            )
        record = self._load((sender, split, index // shard_size), path)[index % shard_size]
        if record["id"] != sample_id:
            raise RuntimeError(f"cache shard mismatch: {sender}/{split}/{sample_id}")
        return record

    def memory(self, split, sender, sample, kind="correct"):
        source_id = sample["shuffle_id"] if kind == "shuffled" else sample["id"]
        record = self.native(split, sender, source_id)
        target = sample["selected_token_count"]
        valid = min(target, record["native_k"].shape[1])
        key = record["native_k"][:, :target].clone()
        value = record["native_v"][:, :target].clone()
        if valid < target:
            key = F.pad(key, (0, 0, 0, 0, 0, target - valid))
            value = F.pad(value, (0, 0, 0, 0, 0, target - valid))
        mask = torch.zeros((1, target), dtype=torch.long)
        mask[:, :valid] = 1
        return key, value, mask

    def query(self, split, sample_id):
        index = self.positions[split][sample_id]
        size = self.cfg["query_shard_size"]
        path = (
            Path(self.cfg["a0_dir"]) / "cache" / query_mode(self.mode) / split / "query_4b"
            / f"shard_{index // size:05d}.pt"
        )
        return self._load(("query", split, index // size), path)[index % size]


def load_reader(cfg, mode, checkpoint=None, trainable=False):
    r1 = r1_module(cfg)
    model = r1.inject_lora(r1.load_model(cfg["model_4b"]), cfg)
    checkpoint = checkpoint or (
        Path(cfg["r1_dir"]) / "artifacts" / source_mode(mode) / "sparse_reader" / "best.pt"
    )
    r1.load_lora(model, checkpoint)
    r1.set_lora(model, True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable:
        for parameter in r1.lora_parameters(model):
            parameter.requires_grad_(True)
        model.train()
    else:
        model.eval()
    return r1, model, r1.tokenizer(cfg["model_4b"]), Path(checkpoint)


def save_lora(r1, model, path, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": r1.lora_state(model), **metadata}, path)


def set_external(controller, model, layers, pre_key, value, mask, positions):
    dtype = next(model.parameters()).dtype
    pre_key = pre_key.to(device=device(), dtype=dtype)
    value = value.to(device=device(), dtype=dtype)
    controller.set_memory(memory_dict(model, layers, pre_key, value, positions), mask)


def answer_target(tok, answer, maximum):
    ids = tok(" " + answer, add_special_tokens=False).input_ids[:maximum - 1]
    ids.append(tok.eos_token_id)
    return ids


def answer_loss(cfg, model, tok, controller, sample, memory=None, compact=False):
    question = sample["question_token_ids"]
    target = answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
    current = torch.tensor([question + target[:-1]], dtype=torch.long, device=device())
    question_positions = list(range(len(question))) if compact else sample["question_position_ids"]
    positions = question_positions + list(
        range(question_positions[-1] + 1, question_positions[-1] + len(target))
    )
    if memory is not None:
        layers, key, value, mask = memory
        set_external(controller, model, layers, key, value, mask, sample["selected_position_ids"])
    else:
        controller.clear()
    logits = model(
        current,
        attention_mask=torch.ones_like(current),
        position_ids=torch.tensor([positions], dtype=torch.long, device=device()),
        use_cache=False,
    ).logits
    if memory is not None:
        controller.assert_usage(layers)
    controller.clear()
    selected = logits[:, len(question) - 1: len(question) - 1 + len(target)].float()
    gold = torch.tensor(target, dtype=torch.long, device=device())
    return F.cross_entropy(selected.reshape(-1, selected.shape[-1]), gold)


@torch.no_grad()
def generate(cfg, model, tok, controller, sample, memory=None, compact=False):
    question = sample["question_token_ids"]
    positions = list(range(len(question))) if compact else sample["question_position_ids"]
    if memory is not None:
        layers, key, value, mask = memory
        set_external(controller, model, layers, key, value, mask, sample["selected_position_ids"])
    else:
        controller.clear()
    input_ids = torch.tensor([question], dtype=torch.long, device=device())
    attention_mask = torch.ones_like(input_ids)
    output = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=torch.tensor([positions], dtype=torch.long, device=device()),
        cache_position=torch.arange(input_ids.shape[1], device=device()),
        past_key_values=DynamicCache(config=model.config),
        use_cache=True,
    )
    past = output.past_key_values
    next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    generated, next_position = [], positions[-1] + 1
    for _ in range(cfg["max_new_tokens"]):
        token = int(next_token.item())
        if token == tok.eos_token_id:
            break
        generated.append(token)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=torch.long, device=device())), 1
        )
        output = model(
            next_token,
            attention_mask=attention_mask,
            position_ids=torch.tensor([[next_position]], dtype=torch.long, device=device()),
            cache_position=torch.tensor([past.get_seq_length()], device=device()),
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    if memory is not None:
        controller.assert_usage(layers)
    controller.clear()
    return tok.decode(generated, skip_special_tokens=True).strip()


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em(prediction, answer):
    return float(normalize_answer(prediction) == normalize_answer(answer))


def token_f1(prediction, answer):
    pred, gold = normalize_answer(prediction).split(), normalize_answer(answer).split()
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if not pred or not gold:
        return float(pred == gold)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def summarize(records):
    output = {}
    for condition in sorted({x["condition"] for x in records}):
        rows = [x for x in records if x["condition"] == condition]
        row = {
            "count": len(rows),
            "em": sum(x["em"] for x in rows) / len(rows),
            "token_f1": sum(x["token_f1"] for x in rows) / len(rows),
            "nll": sum(x["nll"] for x in rows) / len(rows),
        }
        for kind in ("bridge", "comparison"):
            chosen = [x for x in rows if str(x.get("type", "")).lower() == kind]
            row[f"{kind}_f1"] = (
                sum(x["token_f1"] for x in chosen) / len(chosen) if chosen else None
            )
        output[condition] = row
    return output


def write_records(root, records, summary):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "per_sample.json", records)
    save_json(root / "summary.json", summary)
    with (root / "manual_c_p_w.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(records[0]) + ["manual_correct", "manual_partial", "manual_wrong"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "manual_correct": "", "manual_partial": "", "manual_wrong": ""})


def writer_memory(cfg, writer, store, split, sample, kind="correct"):
    key, value, mask = store.memory(split, "q35", sample, kind)
    key, value = writer(key.to(device()), value.to(device()))
    return cfg["anchor_layers"], key, value, mask


def native_memory(cfg, store, split, sample, anchor=True, kind="correct"):
    key, value, mask = store.memory(split, "4b", sample, kind)
    if anchor:
        layers = cfg["anchor_layers"]
        key, value = key[layers], value[layers]
    else:
        layers = list(range(36))
    return layers, key, value, mask


def evaluate_conditions(cfg, rows, store, r1, model, tok, controller, conditions, split="test"):
    records = []
    for condition in conditions:
        progress(f"evaluate {condition['key']}")
        r1.set_lora(model, condition.get("lora", True))
        writer = condition.get("writer")
        if writer is not None:
            writer.eval()
        for sample in rows[split]:
            kind = condition.get("kind", "correct")
            if condition.get("source") == "native4":
                memory = native_memory(cfg, store, split, sample, condition.get("anchor", True), kind)
            elif condition.get("source") == "q35":
                memory = writer_memory(cfg, writer, store, split, sample, kind)
            else:
                memory = None
            loss = answer_loss(
                cfg, model, tok, controller, sample, memory,
                compact=condition.get("compact", False),
            ).item()
            prediction = generate(
                cfg, model, tok, controller, sample, memory,
                compact=condition.get("compact", False),
            )
            records.append({
                "id": sample["id"], "type": sample.get("type", ""),
                "condition": condition["key"], "answer": sample["answer"],
                "prediction": prediction, "em": em(prediction, sample["answer"]),
                "token_f1": token_f1(prediction, sample["answer"]), "nll": loss,
                "reader_protocol": condition.get("reader_protocol", "full36_reader"),
            })
    r1.set_lora(model, True)
    return records
