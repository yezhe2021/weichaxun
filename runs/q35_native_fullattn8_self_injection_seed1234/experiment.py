from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


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


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(
        cfg["model_q35"], local_files_only=True, use_fast=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg):
    return AutoModelForCausalLM.from_pretrained(
        cfg["model_q35"], local_files_only=True, dtype=torch.float16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )


def encode(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def format_sample(raw, tok):
    gold = {(str(title), int(index)) for title, index in raw["supporting_facts"]}
    context, selected, evidence_sentences = [], [], []
    for title, sentences in raw["context"]:
        context.extend(encode(tok, f"Document: {title}\n"))
        for index, sentence in enumerate(sentences):
            context.extend(encode(tok, f"Sentence {index}: "))
            start = len(context)
            content = encode(tok, sentence)
            context.extend(content)
            end = len(context)
            context.extend(encode(tok, "\n"))
            if (str(title), index) in gold:
                selected.extend(range(start, end))
                evidence_sentences.append(f"Document: {title}\nSentence {index}: {sentence}")
    question = encode(tok, f"\nQuestion: {raw['question'].strip()}\n\nAnswer:")
    evidence_prompt = encode(
        tok,
        "Evidence:\n" + "\n".join(evidence_sentences)
        + f"\n\nQuestion: {raw['question'].strip()}\n\nAnswer:",
    )
    if not selected:
        raise RuntimeError(f"no supporting tokens: {raw['_id']}")
    return {
        "id": raw["_id"], "type": raw.get("type"),
        "question": raw["question"].strip(), "answer": str(raw["answer"]).strip(),
        "context_token_ids": context, "question_token_ids": question,
        "full_prompt_ids": context + question,
        "selected_position_ids": selected,
        "question_position_ids": list(range(len(context), len(context) + len(question))),
        "supporting_text_prompt_ids": evidence_prompt,
        "selected_token_count": len(selected),
    }


def rows_for(cfg, mode):
    root = Path(cfg["work_dir"]) / "artifacts" / mode / "manifest.json"
    return load_json(root)


def build_manifest(cfg, mode):
    source = load_json(
        Path(cfg["r1_dir"]) / "artifacts" / "formal" / "manifest.json"
    )
    train_raw = {x["_id"]: x for x in load_json(cfg["hotpot_train"])}
    dev_raw = {x["_id"]: x for x in load_json(cfg["hotpot_dev"])}
    tok = tokenizer(cfg)
    sizes = cfg["smoke_sizes"] if mode == "smoke" else cfg["sizes"]
    output = {}
    for split in sizes:
        raw_map = train_raw if split == "train" else dev_raw
        output[split] = [
            format_sample(raw_map[x["id"]], tok)
            for x in source[split][:sizes[split]]
        ]
        for index, sample in enumerate(output[split]):
            candidates = [
                x for x in output[split]
                if x["id"] != sample["id"]
                and x["answer"].lower() != sample["answer"].lower()
            ]
            sample["shuffle_id"] = candidates[(17 * index + 7) % len(candidates)]["id"]
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "manifest.json", output)
    save_json(Path(cfg["work_dir"]) / "artifacts" / mode / "protocol_audit.json", {
        "sender_input": "complete context only; question absent",
        "sender_tokenizer": "Qwen3.5 native tokenizer",
        "selected_state": "all eight full-attention layers, pre-RoPE K and native V",
        "full_attention_layers": cfg["full_attention_layers"],
        "writer": "identity",
        "receiver": "same frozen Qwen3.5-4B",
        "delta_context_states": "absent/uninitialized",
        "reader_lora": "q_proj and o_proj only at eight full-attention layers",
        "hard_gate": None,
        "counts": {k: len(v) for k, v in output.items()},
    })
    progress(f"{mode}: native-tokenizer manifest built")


class NativeCapture:
    def __init__(self, model, cfg):
        self.cfg, self.values, self.indices = cfg, {}, None
        self.handles = [
            model.model.layers[index].self_attn.register_forward_pre_hook(
                self._hook(index), with_kwargs=True
            )
            for index in cfg["full_attention_layers"]
        ]

    def _hook(self, index):
        def capture(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            batch, length, _ = hidden.shape
            key = module.k_norm(module.k_proj(hidden).view(batch, length, 4, 256))
            value = module.v_proj(hidden).view(batch, length, 4, 256)
            self.values[index] = (
                key[0, self.indices].detach().cpu().half().contiguous(),
                value[0, self.indices].detach().cpu().half().contiguous(),
            )
        return capture

    @torch.no_grad()
    def run(self, model, sample):
        self.values.clear()
        self.indices = torch.tensor(sample["selected_position_ids"], device=cuda())
        ids = torch.tensor([sample["context_token_ids"]], device=cuda())
        positions = torch.arange(ids.shape[1], device=cuda()).unsqueeze(0)
        model(
            ids, attention_mask=torch.ones_like(ids), position_ids=positions,
            use_cache=False,
        )
        layers = self.cfg["full_attention_layers"]
        if set(self.values) != set(layers):
            raise RuntimeError(f"captured layers: {sorted(self.values)}")
        return (
            torch.stack([self.values[x][0] for x in layers]),
            torch.stack([self.values[x][1] for x in layers]),
        )

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def extract(cfg, mode):
    rows = rows_for(cfg, mode)
    model = load_model(cfg).eval()
    capture = NativeCapture(model, cfg)
    root = Path(cfg["work_dir"]) / "cache" / mode
    for split, samples in rows.items():
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(samples), cfg["shard_size"]):
            shard = start // cfg["shard_size"]
            destination = directory / f"shard_{shard:05d}.pt"
            if destination.exists():
                continue
            records = []
            for sample in samples[start:start + cfg["shard_size"]]:
                key, value = capture.run(model, sample)
                records.append({"id": sample["id"], "pre_key": key, "value": value})
            temporary = destination.with_suffix(".tmp")
            torch.save(records, temporary); temporary.replace(destination)
            progress(
                f"{mode}: extract {split} shard {shard + 1}/"
                f"{math.ceil(len(samples)/cfg['shard_size'])}"
            )
    capture.close(); del model; torch.cuda.empty_cache()


class Store:
    def __init__(self, cfg, mode, rows):
        self.cfg, self.mode = cfg, mode
        self.positions = {
            split: {x["id"]: i for i, x in enumerate(samples)}
            for split, samples in rows.items()
        }
        self.cache = {}

    def get(self, split, sample_id):
        index = self.positions[split][sample_id]
        shard = index // self.cfg["shard_size"]
        key = (split, shard)
        if key not in self.cache:
            if len(self.cache) > 4:
                self.cache.clear()
            self.cache[key] = torch.load(
                Path(self.cfg["work_dir"]) / "cache" / self.mode / split
                / f"shard_{shard:05d}.pt", map_location="cpu", weights_only=False,
            )
        record = self.cache[key][index % self.cfg["shard_size"]]
        if record["id"] != sample_id:
            raise RuntimeError("cache shard mismatch")
        return record


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.a = nn.Linear(base.in_features, rank, bias=False, **factory_kwargs)
        self.b = nn.Linear(rank, base.out_features, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        self.enabled = True
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        output = self.base(x)
        if self.enabled:
            output = output + self.b(self.a(self.dropout(x))) * self.scale
        return output


def inject_lora(model, cfg):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for index in cfg["full_attention_layers"]:
        attention = model.model.layers[index].self_attn
        attention.q_proj = LoRALinear(
            attention.q_proj, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"]
        )
        attention.o_proj = LoRALinear(
            attention.o_proj, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"]
        )
    return model


def lora_modules(model):
    return [module for module in model.modules() if isinstance(module, LoRALinear)]


def set_lora(model, enabled):
    for module in lora_modules(model):
        module.enabled = enabled


def lora_parameters(model):
    return [p for module in lora_modules(model) for p in (module.a.weight, module.b.weight)]


def lora_state(model):
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".a.weight" in name or ".b.weight" in name
    }


def load_lora(model, path):
    state = torch.load(path, map_location="cpu", weights_only=False)["lora"]
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameters[name].data.copy_(value.to(parameters[name].device))


def post_rope(model, pre_key, positions):
    length = len(positions)
    position_ids = torch.tensor([positions], device=cuda())
    position_ids = position_ids[None].expand(4, 1, length)[1:]
    dummy = torch.empty(
        1, length, model.config.hidden_size, dtype=pre_key.dtype, device=cuda()
    )
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    output = []
    for layer in range(pre_key.shape[0]):
        key = pre_key[layer].permute(1, 0, 2).unsqueeze(0)
        query = torch.zeros(1, 16, length, 256, dtype=key.dtype, device=cuda())
        _, rotated = apply_rotary_pos_emb(query, key, cos, sin)
        output.append(rotated)
    return output


def sparse_cache(model, cfg, record, positions, zero=False):
    receiver_dtype = model.model.layers[cfg["full_attention_layers"][0]].self_attn.k_proj.weight.dtype
    pre_key = record["pre_key"].to(device=cuda(), dtype=receiver_dtype)
    value = record["value"].to(device=cuda(), dtype=receiver_dtype)
    if zero:
        pre_key = torch.zeros_like(pre_key); value = torch.zeros_like(value)
    rotated = post_rope(model, pre_key, positions)
    cache = DynamicCache(config=model.config)
    for slot, layer in enumerate(cfg["full_attention_layers"]):
        cache.update(
            rotated[slot],
            value[slot].permute(1, 0, 2).unsqueeze(0),
            layer,
        )
    # Engineering invariant: no Context state may exist in any DeltaNet layer.
    for index, kind in enumerate(model.config.layer_types):
        if kind == "linear_attention" and cache.has_previous_state(index):
            raise RuntimeError(f"DeltaNet layer {index} unexpectedly has previous state")
    return cache


def answer_target(tok, answer, maximum):
    ids = encode(tok, " " + answer)[:maximum - 1]
    ids.append(tok.eos_token_id)
    return ids


def answer_loss(cfg, model, tok, sample, record):
    question, target = sample["question_token_ids"], answer_target(
        tok, sample["answer"], cfg["max_answer_tokens"]
    )
    current = torch.tensor([question + target[:-1]], device=cuda())
    positions = sample["question_position_ids"] + list(
        range(sample["question_position_ids"][-1] + 1,
              sample["question_position_ids"][-1] + len(target))
    )
    prefix = len(sample["selected_position_ids"])
    logits = model(
        current,
        attention_mask=torch.ones(1, prefix + current.shape[1], dtype=torch.long, device=cuda()),
        position_ids=torch.tensor([positions], device=cuda()),
        past_key_values=sparse_cache(
            model, cfg, record, sample["selected_position_ids"]
        ),
        cache_position=torch.arange(prefix, prefix + current.shape[1], device=cuda()),
        use_cache=False,
    ).logits
    selected = logits[:, len(question) - 1:len(question) - 1 + len(target)].float()
    return F.cross_entropy(
        selected.reshape(-1, selected.shape[-1]), torch.tensor(target, device=cuda())
    )


@torch.no_grad()
def validate_nll(cfg, model, tok, store, samples):
    model.eval(); values = []
    for sample in samples:
        values.append(answer_loss(cfg, model, tok, sample, store.get("validation", sample["id"])).item())
    return sum(values) / len(values)


def train(cfg, mode):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode); store = Store(cfg, mode, rows); tok = tokenizer(cfg)
    model = inject_lora(load_model(cfg), cfg)
    set_lora(model, True); model.train()
    parameters = lora_parameters(model)
    optimizer = AdamW(parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    grad_acc = cfg["smoke_gradient_accumulation"] if mode == "smoke" else cfg["gradient_accumulation"]
    maximum = cfg["smoke_updates"] if mode == "smoke" else math.ceil(
        len(rows["train"]) * cfg["reader_epochs"] / grad_acc
    )
    out = Path(cfg["work_dir"]) / "artifacts" / mode / "reader"
    out.mkdir(parents=True, exist_ok=True)
    history, evaluations, best, cursor, epoch = [], [], float("inf"), 0, 0
    samples = list(rows["train"]); optimizer.zero_grad(set_to_none=True)
    interval = 1 if mode == "smoke" else 16
    for update in range(1, maximum + 1):
        losses = []
        for _ in range(grad_acc):
            if cursor == 0:
                random.Random(cfg["seed"] + epoch).shuffle(samples); epoch += 1
            sample = samples[cursor]; cursor = (cursor + 1) % len(samples)
            model.train(); set_lora(model, True)
            loss = answer_loss(cfg, model, tok, sample, store.get("train", sample["id"]))
            (loss / grad_acc).backward(); losses.append(loss.detach().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"]).item()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        history.append({"update": update, "answer_nll": sum(losses)/len(losses), "grad_norm": grad_norm})
        if update % interval == 0 or update == maximum:
            validation = validate_nll(cfg, model, tok, store, rows["validation"])
            selected = validation < best
            if selected:
                best = validation
                torch.save({"lora": lora_state(model), "update": update, "validation_nll": validation}, out / "best.pt")
            evaluations.append({"update": update, "validation_nll": validation, "selected": selected})
            save_json(out / "history.json", history); save_json(out / "evaluations.json", evaluations)
            progress(f"{mode}: Reader LoRA {update}/{maximum}")
    save_json(out / "summary.json", {
        "completed": True, "best_validation_nll": best,
        "trained_modules": "q_proj/o_proj at 8 full-attention layers only",
    })
    del model; torch.cuda.empty_cache()


@torch.no_grad()
def generate(cfg, model, tok, sample, record=None, compact=False, text=False, zero=False):
    if text:
        prompt = sample["supporting_text_prompt_ids"]
        positions = list(range(len(prompt)))
    else:
        prompt = sample["question_token_ids"]
        positions = list(range(len(prompt))) if compact else sample["question_position_ids"]
    ids = torch.tensor([prompt], device=cuda())
    mask = torch.ones_like(ids); kwargs = {}
    if record is not None:
        prefix = record["pre_key"].shape[1]
        mask = torch.ones(1, prefix + ids.shape[1], dtype=torch.long, device=cuda())
        kwargs = {
            "past_key_values": sparse_cache(
                model, cfg, record, sample["selected_position_ids"], zero=zero
            ),
            "cache_position": torch.arange(prefix, prefix + ids.shape[1], device=cuda()),
        }
    output = model(
        ids, attention_mask=mask, position_ids=torch.tensor([positions], device=cuda()),
        use_cache=True, **kwargs,
    )
    past, next_token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True)
    generated, next_position = [], positions[-1] + 1
    for _ in range(cfg["max_new_tokens"]):
        token = int(next_token.item())
        if token == tok.eos_token_id:
            break
        generated.append(token)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=cuda())], 1)
        output = model(
            next_token, attention_mask=mask,
            position_ids=torch.tensor([[next_position]], device=cuda()),
            cache_position=torch.tensor([past.get_seq_length()], device=cuda()),
            past_key_values=past, use_cache=True,
        )
        past, next_token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    return tok.decode(generated, skip_special_tokens=True).strip()


def normalize(text):
    import re, string
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, answer):
    from collections import Counter
    pred, gold = normalize(prediction).split(), normalize(answer).split()
    common = sum((Counter(pred) & Counter(gold)).values())
    if not pred or not gold:
        return float(pred == gold)
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def evaluate(cfg, mode):
    seed_all(cfg["seed"])
    rows = rows_for(cfg, mode); store = Store(cfg, mode, rows); tok = tokenizer(cfg)
    model = inject_lora(load_model(cfg), cfg).eval()
    load_lora(model, Path(cfg["work_dir"]) / "artifacts" / mode / "reader" / "best.pt")
    conditions = [
        ("question_only", False, None, "correct", True, False),
        ("gold_supporting_text", False, None, "correct", False, True),
        ("self_sparse_kv_lora_off", False, "memory", "correct", False, False),
        ("self_sparse_kv_lora_on", True, "memory", "correct", False, False),
        ("shuffled_self_kv_lora_on", True, "memory", "shuffled", False, False),
        ("zero_self_kv_lora_on", True, "memory", "zero", False, False),
    ]
    output = []
    by_id = {x["id"]: x for x in rows["test"]}
    for condition, lora, family, kind, compact, text in conditions:
        set_lora(model, lora); model.eval(); progress(f"{mode}: evaluate {condition}")
        for sample in rows["test"]:
            record = None
            if family == "memory":
                source_id = sample["shuffle_id"] if kind == "shuffled" else sample["id"]
                record = store.get("test", source_id)
                positions_sample = by_id[source_id] if kind == "shuffled" else sample
                eval_sample = dict(sample)
                eval_sample["selected_position_ids"] = positions_sample["selected_position_ids"]
            else:
                eval_sample = sample
            prediction = generate(
                cfg, model, tok, eval_sample, record, compact=compact,
                text=text, zero=(kind == "zero"),
            )
            output.append({
                "sample_id": sample["id"], "type": sample.get("type", "unknown"),
                "condition": condition, "answer": sample["answer"],
                "prediction": prediction,
                "em": float(normalize(prediction) == normalize(sample["answer"])),
                "token_f1": token_f1(prediction, sample["answer"]),
                "manual_semantic_correct": "",
            })
    summary = {}
    for condition in sorted({x["condition"] for x in output}):
        selected = [x for x in output if x["condition"] == condition]
        record = {
            "em": sum(x["em"] for x in selected)/len(selected),
            "token_f1": sum(x["token_f1"] for x in selected)/len(selected),
            "count": len(selected),
        }
        for typ in ("bridge", "comparison"):
            subset = [x for x in selected if x["type"] == typ]
            record[f"{typ}_f1"] = sum(x["token_f1"] for x in subset)/len(subset) if subset else None
        summary[condition] = record
    correct = summary["self_sparse_kv_lora_on"]
    summary["dependence_diagnostics"] = {
        "correct_shuffled_em_gap": correct["em"] - summary["shuffled_self_kv_lora_on"]["em"],
        "correct_zero_em_gap": correct["em"] - summary["zero_self_kv_lora_on"]["em"],
        "correct_shuffled_f1_gap": correct["token_f1"] - summary["shuffled_self_kv_lora_on"]["token_f1"],
        "correct_zero_f1_gap": correct["token_f1"] - summary["zero_self_kv_lora_on"]["token_f1"],
    }
    root = Path(cfg["work_dir"]) / "artifacts" / mode / "evaluation"
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "summary.json", summary); save_json(root / "per_sample.json", output)
    with (root / "manual_semantic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys()); writer.writeheader(); writer.writerows(output)
    save_json(root / "completion.json", {
        "completed": True, "identity_writer": True,
        "delta_context_states_transmitted": False, "hard_gate": None,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("action", choices=("manifest", "extract", "train", "evaluate"))
    args = parser.parse_args(); cfg = load_json(args.config)
    if args.action == "manifest": build_manifest(cfg, args.mode)
    elif args.action == "extract": extract(cfg, args.mode)
    elif args.action == "train": train(cfg, args.mode)
    else: evaluate(cfg, args.mode)


if __name__ == "__main__":
    main()
