import hashlib
import json
import os
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


ROOT = Path(__file__).resolve().parent
P2IW_ROOT = Path(os.environ.get("P2IW_ROOT", ROOT.parent / "p2iw_token_preserving_canonical_writer_qwen3_8b_seed1234"))
if str(P2IW_ROOT) not in sys.path:
    sys.path.insert(0, str(P2IW_ROOT))
from p2iw_common import PairCache, TokenCanonicalWriter, file_sha256, projection


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU requires float32")
    return dtype


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8")); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def is_qwen35(model_path):
    with open(Path(model_path) / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    return "qwen3_5" in str(config.get("model_type", "")) or any("Qwen3_5" in value for value in config.get("architectures", []))


def load_receiver(model_path, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_class = AutoModelForImageTextToText if is_qwen35(model_path) else AutoModelForCausalLM
    model = model_class.from_pretrained(
        model_path, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def student_prompt(tokenizer, row):
    system = (
        "Answer the question using the external evidence available to you. Do not use unstated facts. "
        "If the evidence is insufficient, answer INSUFFICIENT. End with exactly: FINAL: <answer>."
    )
    user = f"QUESTION\n{row['question']}"
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def student_prefixed_prompt(tokenizer, row):
    return student_prompt(tokenizer, row) + "FINAL:"


def pack_answer(tokenizer, prompt, answer, max_length, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    suffix = tokenizer(" " + answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    suffix = suffix[:max(1, max_length - len(prompt_ids))]
    ids = torch.tensor([prompt_ids + suffix], dtype=torch.long, device=device)
    labels = ids.clone(); labels[:, :len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels


def sequence_nll(model, tokenizer, reader, row, memory, answer, max_length, device):
    ids, mask, labels = pack_answer(tokenizer, student_prefixed_prompt(tokenizer, row), answer, max_length, device)
    with reader.inject(model, memory):
        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True)
    return output.loss.float()


def canonical_to(memory, device):
    return {
        "keys": memory["keys"].to(device=device), "values": memory["values"].to(device=device),
        "mask": memory.get("mask", torch.ones(memory["keys"].shape[0], dtype=torch.bool)).to(device=device, dtype=torch.bool),
        "answer_token_mask": memory.get("answer_token_mask", torch.zeros(memory["keys"].shape[0], dtype=torch.bool)).to(device=device, dtype=torch.bool),
    }


def normalize_answer(text):
    return re.sub(r"^[\s`*\"']+|[\s`*\"'.,;:!?]+$", "", str(text)).casefold()


def extract_answer(text, allowed):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.I | re.S)
    values = list(dict.fromkeys([*allowed, "INSUFFICIENT"]))
    mapping = {normalize_answer(value): value for value in values}
    pattern = re.compile(r"(?<![\w-])(" + "|".join(sorted(map(re.escape, values), key=len, reverse=True)) + r")(?![\w-])", re.I)
    anchored = re.findall(r"(?:FINAL|ANSWER)\s*:\s*([^\n\r]+)", clean, flags=re.I)
    for region in reversed(anchored):
        found = pattern.findall(region)
        if found:
            return mapping[normalize_answer(found[-1])], "final_anchor"
    found = pattern.findall(clean)
    return (mapping[normalize_answer(found[-1])], "last_valid_answer") if found else ("", "not_found")


@torch.inference_mode()
def generate(model, tokenizer, reader, row, memory, max_new_tokens, device, enabled=True):
    current = tokenizer(student_prefixed_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    past, generated, diagnostics = None, [], {}
    eos = tokenizer.eos_token_id; eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_reached = False
    for _ in range(max_new_tokens):
        if enabled:
            with reader.inject(model, memory, diagnostics):
                output = model(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        else:
            output = model(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        token = int(output.logits[:, -1].argmax(-1).item()); generated.append(token); past = output.past_key_values
        if token in eos_ids:
            eos_reached = True; break
        current = torch.tensor([[token]], dtype=torch.long, device=device)
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True), "token_ids": generated,
        "eos_reached": eos_reached, "diagnostics": diagnostics,
    }


def fixed_negative(cache, index):
    own = {cache.entries[index]["base_answer"], cache.entries[index]["counterfactual_answer"]}
    for offset in range(1, len(cache)):
        candidate = (index + offset) % len(cache)
        other = {cache.entries[candidate]["base_answer"], cache.entries[candidate]["counterfactual_answer"]}
        if own.isdisjoint(other):
            return candidate
    raise RuntimeError("No answer-disjoint negative")

