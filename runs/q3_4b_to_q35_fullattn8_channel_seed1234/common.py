from __future__ import annotations

import csv
import json
import math
import random
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from q35_anchor_injection import memory_dict


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def source_mode(mode):
    return "smoke" if mode == "smoke" else "formal"


def rows_for(cfg, mode):
    rows = load_json(Path(cfg["q35_self_dir"]) / "artifacts" / mode / "manifest.json")
    sizes = cfg["smoke_sizes"] if mode == "smoke" else cfg["sizes"]
    output = {split: [dict(x) for x in rows[split][:sizes[split]]] for split in sizes}
    for samples in output.values():
        for index, sample in enumerate(samples):
            candidates = [
                x for x in samples
                if x["id"] != sample["id"]
                and normalize(x["answer"]) != normalize(sample["answer"])
            ]
            if not candidates:
                raise RuntimeError("cannot build answer-different shuffled control")
            sample["shuffle_id"] = candidates[(17 * index + 7) % len(candidates)]["id"]
    return output


def q3_rows(cfg, mode):
    # The Qwen3.5 manifests (including smoke) are sampled from the R1 formal
    # manifest, so their Qwen3 native records must use that same index space.
    rows = load_json(Path(cfg["q3_r1_dir"]) / "artifacts" / "formal" / "manifest.json")
    sizes = cfg["smoke_sizes"] if mode == "smoke" else cfg["sizes"]
    return {split: [dict(x) for x in rows[split][:sizes[split]]] for split in sizes}


def tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def encode_context_with_offsets(raw, tok):
    ids, offsets, selected, cursor = [], [], [], 0
    gold = {(str(title), int(index)) for title, index in raw["supporting_facts"]}

    def add(text, support=False):
        nonlocal cursor
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        begin = len(ids)
        ids.extend(encoded.input_ids)
        offsets.extend([(cursor + a, cursor + b) for a, b in encoded.offset_mapping])
        if support:
            selected.extend(range(begin, len(ids)))
        cursor += len(text)

    for title, sentences in raw["context"]:
        add(f"Document: {title}\n")
        for index, sentence in enumerate(sentences):
            add(f"Sentence {index}: ")
            add(sentence, (str(title), index) in gold)
            add("\n")
    return ids, offsets, selected


class Stores:
    def __init__(self, cfg, mode, rows):
        self.cfg, self.mode = cfg, mode
        self.positions = {
            split: {sample["id"]: index for index, sample in enumerate(samples)}
            for split, samples in rows.items()
        }
        self.cache = {}

    def _load(self, key, path):
        if key not in self.cache:
            if len(self.cache) > 12:
                self.cache.clear()
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.cache[key]

    def q35(self, split, sample_id):
        index = self.positions[split][sample_id]; size = self.cfg["q35_shard_size"]
        path = Path(self.cfg["q35_self_dir"]) / "cache" / self.mode / split / f"shard_{index // size:05d}.pt"
        record = self._load(("q35", split, index // size), path)[index % size]
        if record["id"] != sample_id: raise RuntimeError("Qwen3.5 shard mismatch")
        return record

    def q3_native(self, split, sample_id):
        index = self.positions[split][sample_id]; size = self.cfg["q3_shard_size"]
        path = Path(self.cfg["q3_r1_dir"]) / "cache" / "formal" / split / "4b" / f"shard_{index // size:05d}.pt"
        record = self._load(("q3", split, index // size), path)[index % size]
        if record["id"] != sample_id: raise RuntimeError("Qwen3 shard mismatch")
        return record

    def aligned(self, split, sample_id):
        index = self.positions[split][sample_id]; size = self.cfg["aligned_shard_size"]
        path = Path(self.cfg["work_dir"]) / "cache" / self.mode / "q3_aligned" / split / f"shard_{index // size:05d}.pt"
        record = self._load(("aligned", split, index // size), path)[index % size]
        if record["id"] != sample_id: raise RuntimeError("aligned shard mismatch")
        return record

    def shuffled_id(self, sample):
        return sample["shuffle_id"]

    @staticmethod
    def fit(record, target, key_name, value_name):
        key, value = record[key_name].clone(), record[value_name].clone()
        valid = min(target, key.shape[1])
        key, value = key[:, :target], value[:, :target]
        if valid < target:
            key = F.pad(key, (0, 0, 0, 0, 0, target - valid))
            value = F.pad(value, (0, 0, 0, 0, 0, target - valid))
        mask = torch.zeros((1, target), dtype=torch.long); mask[:, :valid] = 1
        if "valid_mask" in record:
            original = record["valid_mask"][:target].bool()
            mask[:, :len(original)] &= original.long().unsqueeze(0)
        return key, value, mask

    def native_memory(self, split, sample, kind="correct"):
        source_id = self.shuffled_id(sample) if kind == "shuffled" else sample["id"]
        return self.fit(self.q35(split, source_id), sample["selected_token_count"], "pre_key", "value")

    def source_memory(self, split, sample, kind="correct"):
        source_id = self.shuffled_id(sample) if kind == "shuffled" else sample["id"]
        return self.fit(self.aligned(split, source_id), sample["selected_token_count"], "source_k", "source_v")


def build_alignment_and_cache(cfg, mode):
    rows, sources = rows_for(cfg, mode), q3_rows(cfg, mode)
    source_maps = {split: {x["id"]: x for x in values} for split, values in sources.items()}
    raw_train = {x["_id"]: x for x in load_json(cfg["hotpot_train"])}
    raw_dev = {x["_id"]: x for x in load_json(cfg["hotpot_dev"])}
    tok3, tok35 = tokenizer(cfg["model_q3"]), tokenizer(cfg["model_q35"])
    store = Stores(cfg, mode, rows)
    metadata, audit = [], []
    for split, samples in rows.items():
        raw_map = raw_train if split == "train" else raw_dev
        out = Path(cfg["work_dir"]) / "cache" / mode / "q3_aligned" / split
        out.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(samples), cfg["aligned_shard_size"]):
            shard = start // cfg["aligned_shard_size"]
            destination = out / f"shard_{shard:05d}.pt"
            if destination.exists():
                continue
            records = []
            for sample in samples[start:start + cfg["aligned_shard_size"]]:
                source = source_maps[split][sample["id"]]; raw = raw_map[sample["id"]]
                ids3, offsets3, selected3 = encode_context_with_offsets(raw, tok3)
                ids35, offsets35, selected35 = encode_context_with_offsets(raw, tok35)
                if ids3 != source["full_context_token_ids"] or ids35 != sample["context_token_ids"]:
                    raise RuntimeError(f"tokenization protocol mismatch: {sample['id']}")
                if selected3 != source["selected_position_ids"] or selected35 != sample["selected_position_ids"]:
                    raise RuntimeError(f"supporting-token protocol mismatch: {sample['id']}")
                source_spans = [offsets3[index] for index in selected3]
                target_spans = [offsets35[index] for index in selected35]
                matrix = torch.zeros(len(target_spans), len(source_spans), dtype=torch.float32)
                detail, covered_chars = [], 0
                for target_index, (ta, tb) in enumerate(target_spans):
                    overlaps = []
                    for source_index, (sa, sb) in enumerate(source_spans):
                        amount = max(0, min(tb, sb) - max(ta, sa))
                        if amount:
                            overlaps.append((source_index, amount, (sa, sb)))
                    total = sum(x[1] for x in overlaps)
                    if total:
                        for source_index, amount, _ in overlaps:
                            matrix[target_index, source_index] = amount / total
                        covered_chars += min(total, max(tb - ta, 0))
                    detail.append({
                        "target_index": target_index,
                        "target_token": tok35.convert_ids_to_tokens(ids35[selected35[target_index]]),
                        "target_span": [ta, tb],
                        "sources": [{
                            "source_index": i, "source_token": tok3.convert_ids_to_tokens(ids3[selected3[i]]),
                            "source_span": list(span), "weight": amount / total,
                        } for i, amount, span in overlaps] if total else [],
                    })
                native = store.q3_native(split, sample["id"])
                layers = cfg["q3_source_layers"]
                key = native["native_k"][layers].float().reshape(8, len(source_spans), 1024)
                value = native["native_v"][layers].float().reshape(8, len(source_spans), 1024)
                aligned_k = torch.einsum("ts,lsh->lth", matrix, key).reshape(8, len(target_spans), 8, 128)
                aligned_v = torch.einsum("ts,lsh->lth", matrix, value).reshape_as(aligned_k)
                valid = matrix.sum(-1) > 0
                records.append({
                    "id": sample["id"], "source_k": aligned_k.half(),
                    "source_v": aligned_v.half(), "valid_mask": valid,
                })
                coverage = valid.float().mean().item()
                audit.append({
                    "id": sample["id"], "split": split,
                    "q3_support_tokens": len(source_spans), "q35_support_tokens": len(target_spans),
                    "covered_target_tokens": int(valid.sum()), "target_token_coverage": coverage,
                    "target_character_coverage": covered_chars / max(sum(b-a for a,b in target_spans), 1),
                })
                metadata.append({"id": sample["id"], "split": split, "alignment": detail})
            temporary = destination.with_suffix(".tmp")
            torch.save(records, temporary); temporary.replace(destination)
            progress(f"{mode}: alignment/cache {split} shard {shard + 1}/{math.ceil(len(samples)/cfg['aligned_shard_size'])}")
    root = Path(cfg["work_dir"]) / "artifacts" / mode
    save_json(root / "alignment_audit.json", audit)
    save_json(root / "alignment_metadata.json", metadata)


def compute_scales(cfg, mode):
    rows = rows_for(cfg, mode)
    store = Stores(cfg, mode, rows)
    sums = {name: torch.zeros(8, 1024, dtype=torch.float64) for name in ("source_k", "source_v", "target_k", "target_v")}
    counts = {"source": 0, "target": 0}
    for index, sample in enumerate(rows["train"], 1):
        sk, sv, mask = store.source_memory("train", sample)
        target = store.q35("train", sample["id"])
        tk, tv = target["pre_key"].float(), target["value"].float()
        valid = mask[0].bool()
        for name, tensor in (("source_k", sk), ("source_v", sv), ("target_k", tk), ("target_v", tv)):
            x = tensor[:, valid].float().reshape(8, -1, 1024)
            sums[name] += x.double().square().sum(1)
        counts["source"] += int(valid.sum()); counts["target"] += int(valid.sum())
        if index % 64 == 0 or index == len(rows["train"]): progress(f"{mode}: scale stats {index}/{len(rows['train'])}")
    scales = {
        name: (value / max(counts[name.split('_')[0]], 1) + cfg["scale_epsilon"]).sqrt().clamp_min(cfg["scale_floor"]).float()
        for name, value in sums.items()
    }
    root = Path(cfg["work_dir"]) / "artifacts" / mode
    torch.save(scales, root / "scales.pt")
    save_json(root / "scale_audit.json", {
        "train_only": True, "counts": counts,
        "shapes": {name: list(value.shape) for name, value in scales.items()},
        "finite": {name: bool(torch.isfinite(value).all()) for name, value in scales.items()},
    })


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__(); self.base = base; self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout); self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in base.parameters(): parameter.requires_grad_(False)

    def forward(self, x):
        output = self.base(x)
        if self.enabled:
            hidden = F.linear(self.dropout(x), self.lora_A.to(x.dtype))
            output = output + F.linear(hidden, self.lora_B.to(x.dtype)) * self.scale
        return output


def load_model(cfg):
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_q35"], local_files_only=True, dtype=torch.float16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def inject_lora(model, cfg):
    for index in cfg["q35_target_layers"]:
        attention = model.model.layers[index].self_attn
        for name in ("q_proj", "o_proj"):
            setattr(attention, name, LoRALinear(
                getattr(attention, name), cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"]
            ).to(cuda()))
    return model


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def lora_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n}


def set_lora(model, enabled):
    for module in model.modules():
        if isinstance(module, LoRALinear): module.enabled = enabled


def load_lora(model, path):
    state = torch.load(path, map_location="cpu", weights_only=False)["lora"]
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        target = name
        if target not in parameters and target.endswith(".a.weight"):
            target = target[:-len(".a.weight")] + ".lora_A"
        if target not in parameters and target.endswith(".b.weight"):
            target = target[:-len(".b.weight")] + ".lora_B"
        if target not in parameters:
            raise KeyError(f"unsupported LoRA checkpoint key: {name}")
        parameters[target].data.copy_(value.to(parameters[target].device))


def save_lora(model, path, **metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": lora_state(model), **metadata}, path)


def load_reader(cfg, mode, checkpoint=None, trainable=False):
    model = inject_lora(load_model(cfg), cfg)
    checkpoint = checkpoint or Path(cfg["q35_self_dir"]) / "artifacts" / mode / "reader" / "best.pt"
    load_lora(model, checkpoint); set_lora(model, True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if trainable:
        for parameter in lora_parameters(model): parameter.requires_grad_(True)
        model.train()
    else: model.eval()
    return model, tokenizer(cfg["model_q35"]), Path(checkpoint)


def set_external(cfg, controller, model, key, value, mask, positions):
    dtype = model.model.layers[cfg["q35_target_layers"][0]].self_attn.k_proj.weight.dtype
    key, value = key.to(cuda(), dtype=dtype), value.to(cuda(), dtype=dtype)
    controller.set_memory(memory_dict(model, cfg["q35_target_layers"], key, value, positions), mask)


def answer_target(tok, answer, maximum):
    ids = tok(" " + answer, add_special_tokens=False).input_ids[:maximum - 1]
    ids.append(tok.eos_token_id); return ids


def answer_loss(cfg, model, tok, controller, sample, memory=None, mode="anchor"):
    target = answer_target(tok, sample["answer"], cfg["max_answer_tokens"])
    if mode == "full_text":
        prompt = sample["full_prompt_ids"] + target[:-1]
        positions = list(range(len(prompt)))
        current = torch.tensor([prompt], device=cuda())
        controller.clear()
        logits = model(current, attention_mask=torch.ones_like(current), position_ids=torch.tensor([positions], device=cuda()), use_cache=False).logits
        selected = logits[:, len(sample["full_prompt_ids"]) - 1:len(sample["full_prompt_ids"]) - 1 + len(target)]
    elif mode == "official":
        controller.clear()
        context = torch.tensor([sample["context_token_ids"]], device=cuda())
        prefix = model(context, attention_mask=torch.ones_like(context), position_ids=torch.arange(context.shape[1], device=cuda()).unsqueeze(0), use_cache=True)
        current_ids = sample["question_token_ids"] + target[:-1]
        current = torch.tensor([current_ids], device=cuda())
        positions = sample["question_position_ids"] + list(range(sample["question_position_ids"][-1] + 1, sample["question_position_ids"][-1] + len(target)))
        mask = torch.ones(1, context.shape[1] + current.shape[1], dtype=torch.long, device=cuda())
        logits = model(current, attention_mask=mask, position_ids=torch.tensor([positions], device=cuda()), cache_position=torch.arange(context.shape[1], context.shape[1] + current.shape[1], device=cuda()), past_key_values=prefix.past_key_values, use_cache=False).logits
        selected = logits[:, len(sample["question_token_ids"]) - 1:len(sample["question_token_ids"]) - 1 + len(target)]
    else:
        question = sample["question_token_ids"]
        current = torch.tensor([question + target[:-1]], device=cuda())
        positions = sample["question_position_ids"] + list(range(sample["question_position_ids"][-1] + 1, sample["question_position_ids"][-1] + len(target)))
        if memory is None: controller.clear()
        else: set_external(cfg, controller, model, *memory, sample["selected_position_ids"])
        logits = model(current, attention_mask=torch.ones_like(current), position_ids=torch.tensor([positions], device=cuda()), use_cache=False).logits
        if memory is not None: controller.assert_usage(cfg["q35_target_layers"])
        controller.clear()
        selected = logits[:, len(question) - 1:len(question) - 1 + len(target)]
    return F.cross_entropy(selected.float().reshape(-1, selected.shape[-1]), torch.tensor(target, device=cuda()))


@torch.no_grad()
def generate(cfg, model, tok, controller, sample, memory=None, mode="anchor"):
    controller.clear(); context_length = 0
    if mode == "full_text":
        prompt = sample["full_prompt_ids"]; positions = list(range(len(prompt)))
        ids = torch.tensor([prompt], device=cuda()); mask = torch.ones_like(ids)
        output = model(ids, attention_mask=mask, position_ids=torch.tensor([positions], device=cuda()), use_cache=True)
    elif mode == "official":
        context = torch.tensor([sample["context_token_ids"]], device=cuda()); context_length = context.shape[1]
        prefix = model(context, attention_mask=torch.ones_like(context), position_ids=torch.arange(context_length, device=cuda()).unsqueeze(0), use_cache=True)
        prompt = sample["question_token_ids"]; positions = sample["question_position_ids"]
        ids = torch.tensor([prompt], device=cuda()); mask = torch.ones(1, context_length + ids.shape[1], dtype=torch.long, device=cuda())
        output = model(ids, attention_mask=mask, position_ids=torch.tensor([positions], device=cuda()), cache_position=torch.arange(context_length, context_length + ids.shape[1], device=cuda()), past_key_values=prefix.past_key_values, use_cache=True)
    else:
        prompt = sample["question_token_ids"]; positions = sample["question_position_ids"]
        if memory is not None: set_external(cfg, controller, model, *memory, sample["selected_position_ids"])
        ids = torch.tensor([prompt], device=cuda()); mask = torch.ones_like(ids)
        output = model(ids, attention_mask=mask, position_ids=torch.tensor([positions], device=cuda()), cache_position=torch.arange(ids.shape[1], device=cuda()), past_key_values=DynamicCache(config=model.config), use_cache=True)
    past, next_token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True)
    generated, next_position = [], positions[-1] + 1
    for _ in range(cfg["max_new_tokens"]):
        token = int(next_token.item())
        if token == tok.eos_token_id: break
        generated.append(token)
        mask = torch.cat((mask, torch.ones((1,1), dtype=torch.long, device=cuda())), 1)
        output = model(next_token, attention_mask=mask, position_ids=torch.tensor([[next_position]], device=cuda()), cache_position=torch.tensor([past.get_seq_length()], device=cuda()), past_key_values=past, use_cache=True)
        past, next_token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True); next_position += 1
    if memory is not None and mode == "anchor": controller.assert_usage(cfg["q35_target_layers"])
    controller.clear(); return tok.decode(generated, skip_special_tokens=True).strip()


def normalize(text):
    text = str(text).lower(); text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def em(prediction, answer): return float(normalize(prediction) == normalize(answer))


def token_f1(prediction, answer):
    pred, gold = normalize(prediction).split(), normalize(answer).split()
    common = sum((Counter(pred) & Counter(gold)).values())
    if not pred or not gold: return float(pred == gold)
    if common == 0: return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def summarize(records):
    output = {}
    for condition in sorted({x["condition"] for x in records}):
        rows = [x for x in records if x["condition"] == condition]
        record = {"count": len(rows), "em": sum(x["em"] for x in rows)/len(rows), "token_f1": sum(x["token_f1"] for x in rows)/len(rows), "nll": sum(x["nll"] for x in rows)/len(rows)}
        for kind in ("bridge", "comparison"):
            selected = [x for x in rows if str(x.get("type", "")).lower() == kind]
            record[f"{kind}_em"] = sum(x["em"] for x in selected)/len(selected) if selected else None
            record[f"{kind}_f1"] = sum(x["token_f1"] for x in selected)/len(selected) if selected else None
        output[condition] = record
    return output


def write_results(root, records, summary):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    save_json(root / "per_sample.json", records); save_json(root / "summary.json", summary)
    with (root / "manual_c_p_w.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(records[0]) + ["manual_correct", "manual_partial", "manual_wrong"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in records: writer.writerow({**row, "manual_correct":"", "manual_partial":"", "manual_wrong":""})


def native_memory(store, split, sample, kind="correct"):
    return store.native_memory(split, sample, kind)


def translated_memory(cfg, writer, store, split, sample, kind="correct"):
    key, value, mask = store.source_memory(split, sample, kind)
    key, value = writer(key.to(cuda()), value.to(cuda()))
    return key, value, mask
