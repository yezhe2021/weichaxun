import hashlib
import json
import math
import random
import re
import string
from collections import Counter, OrderedDict
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


READER_LAYERS = [12, 20, 28, 34]
LORA_LAYERS = [12, 13, 20, 21, 28, 29, 34, 35]
VARIANTS = ("reader_only", "reader_lora", "lora_only")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_answer(value):
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction, answer):
    predicted, target = normalize_answer(prediction), normalize_answer(answer)
    exact = float(predicted == target)
    predicted_tokens, target_tokens = predicted.split(), target.split()
    if not predicted_tokens or not target_tokens:
        return exact, float(predicted_tokens == target_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(target_tokens)).values())
    if overlap == 0:
        return exact, 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(target_tokens)
    return exact, 2 * precision * recall / (precision + recall)


def extract_prediction(text):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.I | re.S)
    anchored = re.findall(r"(?:FINAL|ANSWER)\s*:\s*([^\n\r]+)", clean, flags=re.I)
    candidate = anchored[-1] if anchored else next((line for line in clean.splitlines() if line.strip()), "")
    candidate = re.sub(r"^[\s`*:\-]+|[\s`*]+$", "", candidate)
    candidate = re.split(r"\s+(?:because|since|based on)\s+", candidate, maxsplit=1, flags=re.I)[0]
    return candidate.strip(), "final_anchor" if anchored else ("first_line" if candidate else "not_found")


def apply_chat(tokenizer, system, user):
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def question_prompt(tokenizer, row):
    system = "Answer the question with a short answer. End with exactly FINAL: <answer>."
    return apply_chat(tokenizer, system, f"QUESTION\n{row['question']}") + "FINAL:"


def full_text_prompt(tokenizer, row):
    system = "Answer using the supplied gold evidence. Give a short answer. End with exactly FINAL: <answer>."
    user = f"QUESTION\n{row['question']}\n\nGOLD EVIDENCE\n{row['evidence']}"
    return apply_chat(tokenizer, system, user) + "FINAL:"


def load_receiver(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


class MemoryCache:
    def __init__(self, index_path, capacity=4):
        self.path = Path(index_path)
        self.root = self.path.parent
        self.index = read_json(index_path)
        self.entries = self.index["entries"]
        self.capacity = capacity
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index not in self.loaded:
            entry = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
            self.loaded[index] = entry
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]


def memory_to(payload, device):
    memory = payload["memory"].to(device)
    if memory.ndim != 2:
        raise RuntimeError(f"Expected memory [T,dm], got {tuple(memory.shape)}")
    valid = torch.as_tensor(payload["metadata"]["valid_mask"], dtype=torch.bool, device=device)
    support = torch.as_tensor(payload["metadata"]["support_token_mask"], dtype=torch.bool, device=device)
    if valid.numel() != memory.shape[0] or support.numel() != memory.shape[0]:
        raise RuntimeError("Memory mask length mismatch")
    if not valid.any():
        raise RuntimeError("External memory is empty")
    return {
        "states": memory.unsqueeze(0),
        "mask": valid.unsqueeze(0),
        "support_mask": support.unsqueeze(0),
    }


class ReaderRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden):
        value = hidden.float()
        value = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return value * self.weight


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, attention_dim, heads, gate_init):
        super().__init__()
        if attention_dim % heads:
            raise ValueError("attention_dim must be divisible by heads")
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.norm = ReaderRMSNorm(hidden_size)
        self.query = nn.Linear(hidden_size, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, hidden, keys, values, memory_mask):
        batch, sequence, _ = hidden.shape
        query = self.query(self.norm(hidden)).view(batch, sequence, self.heads, self.head_dim)
        scores = torch.einsum("bshd,bthd->bhst", query, keys) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~memory_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        readout = torch.einsum("bhst,bthd->bshd", attention, values).reshape(batch, sequence, self.attention_dim)
        delta = self.gate * self.output(readout)
        return delta.to(hidden.dtype), attention


class CrossAttentionReader(nn.Module):
    def __init__(
        self,
        model,
        memory_dim,
        layers=READER_LAYERS,
        attention_dim=512,
        heads=8,
        gate_init=0.01,
    ):
        super().__init__()
        self.layers = list(layers)
        self.memory_dim = int(memory_dim)
        self.attention_dim = int(attention_dim)
        self.heads = int(heads)
        self.head_dim = self.attention_dim // self.heads
        self.gate_init = float(gate_init)
        hidden_size = int(model.config.hidden_size)
        if max(self.layers) >= int(model.config.num_hidden_layers):
            raise ValueError("Reader layer index exceeds Receiver depth")
        self.key = nn.Linear(self.memory_dim, self.attention_dim, bias=False)
        self.value = nn.Linear(self.memory_dim, self.attention_dim, bias=False)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_size, self.attention_dim, self.heads, gate_init)
            for _ in self.layers
        ])
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        self._memory = None
        self._projected = None
        self._trace = None

    def gates(self):
        return torch.stack([block.gate for block in self.blocks])

    def _project_memory(self):
        if self._projected is None:
            states = self._memory["states"].float()
            batch, tokens, _ = states.shape
            keys = self.key(states).view(batch, tokens, self.heads, self.head_dim)
            values = self.value(states).view(batch, tokens, self.heads, self.head_dim)
            self._projected = (keys, values)
        return self._projected

    @contextmanager
    def inject(self, model, memory, trace=None, enabled=True):
        if not enabled:
            yield trace
            return
        states = memory["states"]
        if states.ndim != 3 or states.shape[-1] != self.memory_dim:
            raise RuntimeError(f"Reader/memory mismatch: {tuple(states.shape)}")
        self._memory = memory
        self._projected = None
        self._trace = trace
        handles = []
        for local_index, layer_index in enumerate(self.layers):
            norm = model.model.layers[layer_index].post_attention_layernorm

            def pre_hook(module, args, local_index=local_index, layer_index=layer_index):
                hidden = args[0]
                keys, values = self._project_memory()
                delta, attention = self.blocks[local_index](hidden, keys, values, self._memory["mask"])
                if self._trace is not None:
                    self._trace.setdefault(str(layer_index), []).append({
                        "attention": attention.detach().float().cpu(),
                        "sequence_length": int(hidden.shape[1]),
                    })
                return (hidden + delta,) + tuple(args[1:])

            handles.append(norm.register_forward_pre_hook(pre_hook))
        try:
            yield trace
        finally:
            for handle in handles:
                handle.remove()
            self._memory = None
            self._projected = None
            self._trace = None

    def metadata(self):
        return {
            "reader_layers": self.layers,
            "memory_dim": self.memory_dim,
            "attention_dim": self.attention_dim,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "gate_init": self.gate_init,
            "injection": "after_self_attention_residual_before_mlp",
            "shared_key_value_projection": True,
        }


class LinearLoRA(nn.Module):
    def __init__(self, input_dim, output_dim, rank=8, alpha=16.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.down = nn.Linear(self.input_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.output_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return self.scale * self.up(self.down(value.float()))


class ReceiverQOLoRA(nn.Module):
    def __init__(self, model, layers=LORA_LAYERS, rank=8, alpha=16.0):
        super().__init__()
        self.layers = list(layers)
        self.rank = int(rank)
        self.alpha = float(alpha)
        modules = {}
        for layer_index in self.layers:
            attention = model.model.layers[layer_index].self_attn
            modules[f"q_{layer_index}"] = LinearLoRA(
                attention.q_proj.in_features, attention.q_proj.out_features, rank, alpha
            )
            modules[f"o_{layer_index}"] = LinearLoRA(
                attention.o_proj.in_features, attention.o_proj.out_features, rank, alpha
            )
        self.adapters = nn.ModuleDict(modules)

    @contextmanager
    def inject(self, model, enabled=True):
        if not enabled:
            yield
            return
        handles = []
        for layer_index in self.layers:
            attention = model.model.layers[layer_index].self_attn

            def q_hook(module, args, output, layer_index=layer_index):
                update = self.adapters[f"q_{layer_index}"](args[0])
                return output + update.to(output.dtype)

            def o_hook(module, args, output, layer_index=layer_index):
                update = self.adapters[f"o_{layer_index}"](args[0])
                return output + update.to(output.dtype)

            handles.append(attention.q_proj.register_forward_hook(q_hook))
            handles.append(attention.o_proj.register_forward_hook(o_hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def metadata(self):
        return {
            "layers": self.layers,
            "targets": ["q_proj", "o_proj"],
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": 0.0,
        }


def build_adapters(model, memory_dim, variant, seed, gate_init=0.01):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    reader = None
    receiver_lora = None
    if variant in {"reader_only", "reader_lora"}:
        torch.manual_seed(seed + 101)
        reader = CrossAttentionReader(model, memory_dim, gate_init=gate_init)
    if variant in {"reader_lora", "lora_only"}:
        torch.manual_seed(seed + 202)
        receiver_lora = ReceiverQOLoRA(model)
    return reader, receiver_lora


def adapter_parameters(reader, receiver_lora):
    modules = [module for module in (reader, receiver_lora) if module is not None]
    return [parameter for module in modules for parameter in module.parameters() if parameter.requires_grad]


def audit_trainable_parameters(model, reader, receiver_lora, optimizer):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Receiver backbone is not fully frozen")
    expected = {id(parameter) for parameter in adapter_parameters(reader, receiver_lora)}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual:
        raise RuntimeError("Optimizer parameters do not exactly match the enabled adapters")
    if not expected:
        raise RuntimeError("No trainable adapter parameters")


def pack_answer(tokenizer, row, max_length, device):
    prompt_ids = tokenizer(question_prompt(tokenizer, row), add_special_tokens=False).input_ids
    suffix_ids = tokenizer(" " + row["answer"] + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    if len(prompt_ids) + len(suffix_ids) > max_length:
        raise RuntimeError("Receiver answer sequence exceeds max_length")
    ids = torch.tensor([prompt_ids + suffix_ids], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, :len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels


def answer_mean_nll(logits, labels):
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    selected = shifted_labels != -100
    if not selected.any():
        raise RuntimeError("No answer tokens in loss")
    return F.cross_entropy(shifted_logits[selected], shifted_labels[selected], reduction="mean")


def adapted_forward(model, input_ids, attention_mask, memory, reader, receiver_lora, reader_enabled, lora_enabled):
    with ExitStack() as stack:
        if reader is not None:
            stack.enter_context(reader.inject(model, memory, enabled=reader_enabled))
        if receiver_lora is not None:
            stack.enter_context(receiver_lora.inject(model, enabled=lora_enabled))
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)


def forward_answer(model, tokenizer, row, memory, reader, receiver_lora, max_length, device, reader_enabled, lora_enabled):
    ids, mask, labels = pack_answer(tokenizer, row, max_length, device)
    output = adapted_forward(
        model, ids, mask, memory, reader, receiver_lora, reader_enabled, lora_enabled
    )
    return answer_mean_nll(output.logits, labels)


@torch.inference_mode()
def generate_adapted(
    model,
    tokenizer,
    row,
    memory,
    reader,
    receiver_lora,
    max_new_tokens,
    reader_enabled,
    lora_enabled,
    trace=None,
):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    with ExitStack() as stack:
        if reader is not None:
            stack.enter_context(reader.inject(model, memory, trace=trace, enabled=reader_enabled))
        if receiver_lora is not None:
            stack.enter_context(receiver_lora.inject(model, enabled=lora_enabled))
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    prediction, method = extract_prediction(text)
    return {
        "text": text,
        "prediction": prediction,
        "parse_method": method,
        "token_ids": tokens,
        "eos_reached": tokenizer.eos_token_id in tokens,
    }


@torch.inference_mode()
def generate_plain(model, tokenizer, prompt, max_new_tokens):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    prediction, method = extract_prediction(text)
    return {
        "text": text,
        "prediction": prediction,
        "parse_method": method,
        "token_ids": tokens,
        "eos_reached": tokenizer.eos_token_id in tokens,
    }


def trace_support_mass(trace, support_mask):
    support = support_mask.detach().float().cpu()
    result = {}
    for layer, calls in trace.items():
        prompt_values, decode_values = [], []
        for call_index, call in enumerate(calls):
            attention = call["attention"]
            mass = (attention * support[:, None, None, :]).sum(dim=-1).mean(dim=(0, 2))
            if call_index == 0 and call["sequence_length"] > 1:
                prompt_values.append(mass)
            else:
                decode_values.append(mass)
        result[layer] = {
            "prompt": torch.stack(prompt_values).mean(0).tolist() if prompt_values else None,
            "decode": torch.stack(decode_values).mean(0).tolist() if decode_values else None,
        }
    return result


def summarize_condition(records, condition):
    selected = [row for row in records if row["condition"] == condition]
    if not selected:
        return {"n": 0, "em": None, "f1": None, "by_type": {}}
    result = {
        "n": len(selected),
        "em": sum(row["em"] for row in selected) / len(selected),
        "f1": sum(row["f1"] for row in selected) / len(selected),
        "eos_rate": sum(row["output"]["eos_reached"] for row in selected) / len(selected),
        "by_type": {},
    }
    for question_type in ("bridge", "comparison"):
        group = [row for row in selected if row["type"] == question_type]
        if group:
            result["by_type"][question_type] = {
                "n": len(group),
                "em": sum(row["em"] for row in group) / len(group),
                "f1": sum(row["f1"] for row in group) / len(group),
            }
    return result
