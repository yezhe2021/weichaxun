import json
import re
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def parse_dtype(name, device):
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU requires float32")
    return dtype


def get_layers(model):
    base = getattr(model, "model", model)
    return base.layers


def student_prompt(tokenizer, row):
    system = (
        "Answer the question using the external evidence available to you. Do not use unstated facts. "
        "If the evidence is insufficient, answer INSUFFICIENT. End with exactly: FINAL: <answer>."
    )
    user = f"QUESTION\n{row['question']}"
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def student_prefixed_prompt(tokenizer, row):
    return student_prompt(tokenizer, row) + "FINAL:"


def full_text_prompt(tokenizer, row):
    system = (
        "Use only the supplied evidence. Follow the relation from the person to the organization, "
        "then from that organization to its location. Ignore distractors. End with exactly: FINAL: <answer>."
    )
    user = (
        f"QUESTION\n{row['question']}\n\n"
        f"EVIDENCE A\n{row['evidence_a']}\n\n"
        f"EVIDENCE B\n{row['evidence_b']}"
    )
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def full_text_prefixed_prompt(tokenizer, row):
    return full_text_prompt(tokenizer, row) + "FINAL:"


def sender_text(row):
    text = (
        f"QUESTION\n{row['question']}\n\n"
        f"EVIDENCE A\n{row['evidence_a']}\n\n"
        f"EVIDENCE B\n{row['evidence_b']}"
    )
    a_start = text.index(row["evidence_a"])
    a_end = a_start + len(row["evidence_a"])
    b_start = text.index(row["evidence_b"], a_end)
    b_end = b_start + len(row["evidence_b"])
    return text, ((a_start, a_end), (b_start, b_end))


def pack_answer(tokenizer, prompt, answer, max_length, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(f"FINAL: {answer}" + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    suffix_ids = suffix_ids[: max(1, max_length - len(prompt_ids))]
    ids = torch.tensor([prompt_ids + suffix_ids], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels


def pack_prefixed_answer(tokenizer, prompt, answer, max_length, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(" " + answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    suffix_ids = suffix_ids[: max(1, max_length - len(prompt_ids))]
    ids = torch.tensor([prompt_ids + suffix_ids], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels


def normalize_answer(text):
    return re.sub(r"^[\s`*\"']+|[\s`*\"'.,;:!?]+$", "", str(text)).casefold()


def extract_answer(text, allowed):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.IGNORECASE | re.DOTALL)
    values = list(dict.fromkeys([*allowed, "INSUFFICIENT"]))
    mapping = {normalize_answer(value): value for value in values}
    pattern = re.compile(
        r"(?<![\w-])(" + "|".join(sorted(map(re.escape, values), key=len, reverse=True)) + r")(?![\w-])",
        re.IGNORECASE,
    )
    anchored = re.findall(r"(?:FINAL|ANSWER|答案)\s*[:：]\s*([^\n\r]+)", clean, flags=re.IGNORECASE)
    for region in reversed(anchored):
        found = pattern.findall(region)
        if found:
            return mapping[normalize_answer(found[-1])], "final_anchor"
    found = pattern.findall(clean)
    if found:
        return mapping[normalize_answer(found[-1])], "last_valid_answer"
    return "", "not_found"


def iter_cache(index_path):
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    root = Path(index_path).parent
    for shard in index["shards"]:
        for example in torch.load(root / shard, map_location="cpu", weights_only=False)["examples"]:
            yield example


def memory_to(memory, device, dtype=None):
    output = {
        "keys": [value.to(device=device, dtype=dtype or value.dtype) for value in memory["keys"]],
        "values": [value.to(device=device, dtype=dtype or value.dtype) for value in memory["values"]],
    }
    if "answer_token_mask" in memory:
        output["answer_token_mask"] = memory["answer_token_mask"].to(device=device, dtype=torch.bool)
    return output


def zero_memory(memory):
    output = {
        "keys": [torch.zeros_like(value) for value in memory["keys"]],
        "values": [torch.zeros_like(value) for value in memory["values"]],
    }
    if "answer_token_mask" in memory:
        output["answer_token_mask"] = memory["answer_token_mask"]
    return output


def mismatched_memory(key_source, value_source):
    keys = []
    values = []
    for key, value in zip(key_source["keys"], value_source["values"]):
        length = min(key.shape[1], value.shape[1])
        keys.append(key[:, :length])
        values.append(value[:, :length])
    output = {"keys": keys, "values": values}
    if "answer_token_mask" in value_source:
        output["answer_token_mask"] = value_source["answer_token_mask"][: values[0].shape[1]]
    return output


class NativeKVExternalReader(nn.Module):
    def __init__(self, model, max_gate=0.5, gate_init=0.0, query_rank=0, output_rank=0):
        super().__init__()
        self.layer_count = len(get_layers(model))
        self.max_gate = float(max_gate)
        init = max(-0.999, min(0.999, gate_init / max(max_gate, 1e-8)))
        self.gate_logits = nn.Parameter(torch.full((self.layer_count,), float(torch.atanh(torch.tensor(init)))))
        self.query_rank = int(query_rank)
        self.output_rank = int(output_rank)
        hidden_size = int(model.config.hidden_size)
        query_width = int(model.config.num_attention_heads) * int(model.config.head_dim)
        if self.query_rank > 0:
            self.query_down = nn.ModuleList(
                [nn.Linear(query_width, self.query_rank, bias=False) for _ in range(self.layer_count)]
            )
            self.query_up = nn.ModuleList(
                [nn.Linear(self.query_rank, query_width, bias=False) for _ in range(self.layer_count)]
            )
            for module in self.query_up:
                nn.init.zeros_(module.weight)
        else:
            self.query_down = nn.ModuleList()
            self.query_up = nn.ModuleList()
        if self.output_rank > 0:
            self.output_down = nn.ModuleList(
                [nn.Linear(hidden_size, self.output_rank, bias=False) for _ in range(self.layer_count)]
            )
            self.output_up = nn.ModuleList(
                [nn.Linear(self.output_rank, hidden_size, bias=False) for _ in range(self.layer_count)]
            )
            for module in self.output_up:
                nn.init.zeros_(module.weight)
        else:
            self.output_down = nn.ModuleList()
            self.output_up = nn.ModuleList()
        self._memory = None
        self._pending = {}
        self._diagnostics = None

    def gates(self):
        return self.max_gate * torch.tanh(self.gate_logits)

    def _external_output(self, attention, hidden_states, layer_index):
        key = self._memory["keys"][layer_index]
        value = self._memory["values"][layer_index]
        batch, query_length, _ = hidden_states.shape
        if key.shape[0] != attention.config.num_key_value_heads:
            raise ValueError(f"Layer {layer_index}: unexpected KV head count {key.shape[0]}")
        head_dim = attention.head_dim
        query = attention.q_norm(
            attention.q_proj(hidden_states).view(batch, query_length, -1, head_dim)
        ).transpose(1, 2)
        query_delta = None
        if self.query_rank > 0:
            query_flat = query.transpose(1, 2).reshape(batch, query_length, -1).contiguous()
            query_delta = self.query_up[layer_index](self.query_down[layer_index](query_flat.float()))
            query = (query_flat + query_delta.to(query_flat.dtype)).view(
                batch, query_length, attention.config.num_attention_heads, head_dim
            ).transpose(1, 2)
        groups = attention.config.num_attention_heads // attention.config.num_key_value_heads
        key = key.unsqueeze(0).expand(batch, -1, -1, -1).repeat_interleave(groups, dim=1)
        value = value.unsqueeze(0).expand(batch, -1, -1, -1).repeat_interleave(groups, dim=1)
        scores = torch.matmul(query.float(), key.transpose(-1, -2).float()) * attention.scaling
        probability = scores.softmax(dim=-1)
        readout = torch.matmul(probability, value.float()).to(hidden_states.dtype)
        readout = readout.transpose(1, 2).reshape(batch, query_length, -1).contiguous()
        projected = attention.o_proj(readout)
        if self.output_rank > 0:
            correction = self.output_up[layer_index](self.output_down[layer_index](projected.float()))
            projected = projected + correction.to(projected.dtype)
        gate = self.gates()[layer_index].to(projected.dtype)
        external = gate * projected
        if self._diagnostics is not None:
            slot = self._diagnostics.setdefault(str(layer_index), {"calls": 0, "readout_norm": 0.0, "delta_norm": 0.0})
            slot["calls"] += 1
            slot["readout_norm"] += float(projected.detach().float().norm(dim=-1).mean().cpu())
            slot["delta_norm"] += float(external.detach().float().norm(dim=-1).mean().cpu())
            slot["gate"] = float(gate.detach().float().cpu())
            slot["attention_entropy"] = float(
                (-(probability * probability.clamp_min(1e-8).log()).sum(dim=-1).mean()
                 / torch.log(torch.tensor(max(2, probability.shape[-1]), device=probability.device))).detach().cpu()
            )
            slot["query_delta_norm"] = (
                float(query_delta.detach().float().norm(dim=-1).mean().cpu())
                if query_delta is not None
                else 0.0
            )
            answer_mask = self._memory.get("answer_token_mask")
            if answer_mask is not None and answer_mask.numel() == probability.shape[-1] and answer_mask.any():
                slot["target_attention_mass"] = float(
                    probability[..., answer_mask].sum(dim=-1).mean().detach().cpu()
                )
            if self._diagnostics.get("_capture_vectors", False):
                slot["readout_vector"] = projected[:, -1, :].detach().float().mean(dim=0).cpu()
                slot["delta_vector"] = external[:, -1, :].detach().float().mean(dim=0).cpu()
        return external

    @contextmanager
    def inject(self, model, memory, diagnostics=None):
        self._memory = memory
        self._diagnostics = diagnostics
        handles = []
        for layer_index, layer in enumerate(get_layers(model)):
            attention = layer.self_attn

            def pre_hook(module, args, kwargs, layer_index=layer_index):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                self._pending[layer_index] = self._external_output(module, hidden, layer_index)

            def post_hook(module, args, kwargs, output, layer_index=layer_index):
                external = self._pending.pop(layer_index)
                if isinstance(output, tuple):
                    return (output[0] + external,) + output[1:]
                return output + external

            handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(attention.register_forward_hook(post_hook, with_kwargs=True))
        try:
            yield diagnostics
        finally:
            for handle in handles:
                handle.remove()
            self._pending.clear()
            self._memory = None
            self._diagnostics = None


def summarize_diagnostics(diagnostics):
    rows = []
    for layer, values in diagnostics.items():
        if not isinstance(values, dict) or "calls" not in values:
            continue
        calls = max(1, values["calls"])
        rows.append(
            {
                "layer": int(layer),
                "gate": values.get("gate", 0.0),
                "readout_norm": values["readout_norm"] / calls,
                "delta_norm": values["delta_norm"] / calls,
                "attention_entropy": values.get("attention_entropy", 0.0),
                "target_attention_mass": values.get("target_attention_mass", 0.0),
                "query_delta_norm": values.get("query_delta_norm", 0.0),
            }
        )
    return sorted(rows, key=lambda row: row["layer"])
