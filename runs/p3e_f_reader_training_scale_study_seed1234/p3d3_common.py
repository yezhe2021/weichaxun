import hashlib
import json
import math
import random
import re
import string
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SELECTED_LAYERS = [int(round(value)) for value in torch.linspace(0, 35, 16).tolist()]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_json(path):
    with open(path, encoding="utf-8") as handle: return json.load(handle)


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle: json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def normalize_answer(value):
    text = str(value).lower(); text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction, answer):
    predicted, target = normalize_answer(prediction), normalize_answer(answer)
    exact = float(predicted == target); p_tokens, t_tokens = predicted.split(), target.split()
    if not p_tokens or not t_tokens: return exact, float(p_tokens == t_tokens)
    overlap = sum((Counter(p_tokens) & Counter(t_tokens)).values())
    if not overlap: return exact, 0.0
    precision, recall = overlap / len(p_tokens), overlap / len(t_tokens)
    return exact, 2 * precision * recall / (precision + recall)


def extract_prediction(text):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.I | re.S)
    clean = re.sub(r"</?answer>", "", clean, flags=re.I).strip()
    anchored = re.findall(r"(?:FINAL|ANSWER)\s*:\s*([^\n\r]+)", clean, flags=re.I)
    candidate = anchored[-1] if anchored else next((line for line in clean.splitlines() if line.strip()), "")
    candidate = re.sub(r"^[\s`*:\-]+|[\s`*]+$", "", candidate)
    candidate = re.split(r"\s+(?:because|since|based on)\s+", candidate, maxsplit=1, flags=re.I)[0]
    return candidate.strip(), "final_anchor" if anchored else ("first_line" if candidate else "not_found")


def apply_chat(tokenizer, system, user):
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def question_prompt(tokenizer, row):
    system = "Answer the question with a short answer. End with exactly FINAL: <answer>."
    return apply_chat(tokenizer, system, f"QUESTION\n{row['question']}") + "FINAL:"


def evidence_block(row): return f"EVIDENCE A\n{row['evidence_a']}\n\nEVIDENCE B\n{row['evidence_b']}"


def load_receiver(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, tokenizer


class MemoryCache:
    def __init__(self, index_path, capacity=4):
        self.path = Path(index_path); self.root = self.path.parent; self.index = read_json(index_path); self.entries = self.index["entries"]
        self.capacity = capacity; self.loaded = OrderedDict()
    def __len__(self): return len(self.entries)
    def load(self, index):
        if index not in self.loaded:
            self.loaded[index] = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
            while len(self.loaded) > self.capacity: self.loaded.popitem(last=False)
        self.loaded.move_to_end(index); return self.loaded[index]


def memory_to(payload, device, oracle_support=False):
    keys = payload["keys"].float().to(device); values = payload["values"].float().to(device)
    tokens = keys.shape[1]; valid = torch.as_tensor(payload["metadata"]["valid_mask"], dtype=torch.bool, device=device)
    support = torch.as_tensor(payload["metadata"]["support_token_mask"], dtype=torch.bool, device=device)
    if valid.numel() != tokens or support.numel() != tokens: raise RuntimeError("Memory token-mask length mismatch")
    mask = valid & support if oracle_support else valid
    if oracle_support and not mask.any(): raise RuntimeError("Oracle-support mask is empty")
    return {"keys": keys, "values": values, "mask": mask, "support_mask": support}


def aliases_overlap(left, right):
    left, right = normalize_answer(left), normalize_answer(right)
    if not left or not right: return False
    return left == right or (min(len(left), len(right)) >= 4 and (left in right or right in left))


def hard_negative_mapping(cache):
    payloads = [cache.load(index) for index in range(len(cache))]; mapping = []
    for index, payload in enumerate(payloads):
        row = payload["row"]; current_titles = {normalize_answer(title) for title in row.get("supporting_titles", [])}; current_bridge = normalize_answer(row.get("bridge_entity", "")); current_answer = row["answer"]
        candidates = []
        for candidate, other_payload in enumerate(payloads):
            if candidate == index: continue
            other = other_payload["row"]
            if other.get("type") != row.get("type") or other.get("answer_type") != row.get("answer_type"): continue
            if aliases_overlap(current_answer, other["answer"]): continue
            if current_titles & {normalize_answer(title) for title in other.get("supporting_titles", [])}: continue
            other_bridge = normalize_answer(other.get("bridge_entity", ""))
            if current_bridge and other_bridge and current_bridge == other_bridge: continue
            if normalize_answer(current_answer) in normalize_answer(evidence_block(other)): continue
            length_gap = abs(int(payload["keys"].shape[1]) - int(other_payload["keys"].shape[1]))
            answer_gap = abs(len(str(current_answer).split()) - len(str(other["answer"]).split()))
            candidates.append((length_gap, answer_gap, candidate))
        if not candidates: raise RuntimeError(f"No leakage-safe hard negative for sample {row.get('id', index)}")
        mapping.append(min(candidates)[2])
    return mapping


class LowRankProjection(nn.Module):
    def __init__(self, input_dim, output_dim, rank, nonlinear=False, small_output=False):
        super().__init__(); self.down = nn.Linear(input_dim, rank, bias=False); self.up = nn.Linear(rank, output_dim, bias=False); self.nonlinear = nonlinear
        nn.init.orthogonal_(self.down.weight)
        if small_output: nn.init.normal_(self.up.weight, std=1e-3)
        else: nn.init.orthogonal_(self.up.weight)
    def forward(self, value):
        hidden = self.down(value)
        return self.up(F.silu(hidden) if self.nonlinear else hidden)


class LayerAlignedBranch(nn.Module):
    def __init__(self, query_heads, head_dim, memory_dim, hidden_size, rank, gate_init):
        super().__init__(); self.query_heads = query_heads; self.head_dim = head_dim; self.memory_dim = memory_dim
        self.query_adapter = LowRankProjection(query_heads * head_dim, memory_dim, rank, nonlinear=True)
        self.output_projection = LowRankProjection(memory_dim, hidden_size, rank, nonlinear=False, small_output=True)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
    def native_query(self, q_states):
        if q_states.ndim != 4: raise RuntimeError(f"Expected rank-4 q_norm output, got {tuple(q_states.shape)}")
        if q_states.shape[1] == self.query_heads: q_states = q_states.transpose(1, 2)
        elif q_states.shape[2] != self.query_heads: raise RuntimeError(f"Cannot locate {self.query_heads} Query heads in {tuple(q_states.shape)}")
        if q_states.shape[-1] != self.head_dim: raise RuntimeError("Query head dimension mismatch")
        return q_states.reshape(q_states.shape[0], q_states.shape[1], -1)
    def forward(self, q_states, keys, values, mask):
        query = F.layer_norm(self.query_adapter(self.native_query(q_states).float()), (self.memory_dim,))
        scores = torch.einsum("bsd,td->bst", query, keys.float()) / math.sqrt(self.memory_dim)
        scores = scores.masked_fill(~mask[None, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        readout = torch.einsum("bst,td->bsd", attention, values.float())
        output = self.output_projection(readout)
        return (self.gate * output).to(q_states.dtype), attention


class LayerAlignedNativeQueryReader(nn.Module):
    def __init__(self, model, memory_dim, selected_layers=SELECTED_LAYERS, rank=32, gate_init=0.01):
        super().__init__(); self.selected_layers = list(selected_layers); self.memory_dim = int(memory_dim); self.rank = int(rank); self.gate_init = float(gate_init)
        config = model.config; self.query_heads = int(config.num_attention_heads); self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)); self.hidden_size = int(config.hidden_size)
        self.branches = nn.ModuleList([LayerAlignedBranch(self.query_heads, self.head_dim, memory_dim, self.hidden_size, rank, gate_init) for _ in self.selected_layers])
        self._memory = None; self._queries = {}; self._trace = None
    def gates(self): return torch.stack([branch.gate for branch in self.branches])
    @contextmanager
    def inject(self, model, memory, trace=None):
        if memory["keys"].shape != memory["values"].shape or memory["keys"].shape[0] != len(self.selected_layers) or memory["keys"].shape[-1] != self.memory_dim: raise RuntimeError("Reader/memory interface mismatch")
        self._memory, self._trace = memory, trace; handles = []
        for local, layer_index in enumerate(self.selected_layers):
            attention = model.model.layers[layer_index].self_attn
            def q_hook(module, args, output, local=local): self._queries[local] = output
            def attention_hook(module, args, kwargs, output, local=local, layer_index=layer_index):
                if local not in self._queries: raise RuntimeError("q_norm hook did not capture Native Query")
                delta, weights = self.branches[local](self._queries.pop(local), self._memory["keys"][local], self._memory["values"][local], self._memory["mask"])
                if self._trace is not None:
                    slot = self._trace.setdefault(layer_index, [])
                    slot.append({"attention": weights, "delta": delta})
                if isinstance(output, tuple): return (output[0] + delta,) + output[1:]
                return output + delta
            handles.append(attention.q_norm.register_forward_hook(q_hook))
            handles.append(attention.register_forward_hook(attention_hook, with_kwargs=True))
        try: yield trace
        finally:
            for handle in handles: handle.remove()
            self._queries.clear(); self._memory = None; self._trace = None
    def metadata(self):
        return {"selected_layers": self.selected_layers, "memory_dim": self.memory_dim, "rank": self.rank, "gate_init": self.gate_init, "query_heads": self.query_heads, "head_dim": self.head_dim, "hidden_size": self.hidden_size, "injection": "self_attention_output_before_decoder_residual", "query": "native_q_proj_q_norm_pre_rope"}


def pack_answer(tokenizer, row, answer, max_length, device):
    prompt_ids = tokenizer(question_prompt(tokenizer, row), add_special_tokens=False).input_ids
    suffix = tokenizer(" " + answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    if len(prompt_ids) + len(suffix) > max_length: raise RuntimeError("Receiver sequence exceeds max length")
    ids = torch.tensor([prompt_ids + suffix], dtype=torch.long, device=device); labels = ids.clone(); labels[:, :len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels


def answer_mean_nll(logits, labels):
    shifted_logits, shifted_labels = logits[:, :-1].float(), labels[:, 1:]
    selected = shifted_labels != -100
    losses = F.cross_entropy(shifted_logits[selected], shifted_labels[selected], reduction="none")
    if losses.numel() == 0: raise RuntimeError("No answer tokens in loss")
    return losses.mean()


def forward_answer(model, tokenizer, reader, row, memory, max_length, device, enabled=True):
    ids, mask, labels = pack_answer(tokenizer, row, row["answer"], max_length, device)
    if enabled:
        with reader.inject(model, memory): output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    else: output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    return answer_mean_nll(output.logits, labels)


@torch.inference_mode()
def generate(model, tokenizer, reader, row, memory, max_new_tokens, enabled=True, trace=None):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False); encoded = {name: value.to(model.device) for name, value in encoded.items()}
    kwargs = dict(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if enabled:
        with reader.inject(model, memory, trace): output = model.generate(**kwargs)
    else: output = model.generate(**kwargs)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist(); text = tokenizer.decode(tokens, skip_special_tokens=True); prediction, method = extract_prediction(text)
    return {"text": text, "prediction": prediction, "parse_method": method, "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens}
