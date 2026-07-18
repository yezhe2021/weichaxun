import hashlib
import json
import random
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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


def is_qwen35(model_path):
    with open(Path(model_path) / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    model_type = str(config.get("model_type", ""))
    architectures = " ".join(config.get("architectures", []))
    return "qwen3_5" in model_type or "Qwen3_5" in architectures


def load_receiver(model_path, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cls = AutoModelForImageTextToText if is_qwen35(model_path) else AutoModelForCausalLM
    model = cls.from_pretrained(
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
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def student_prefixed_prompt(tokenizer, row):
    return student_prompt(tokenizer, row) + "FINAL:"


def full_text_prompt(tokenizer, row):
    name = str(getattr(tokenizer, "name_or_path", ""))
    if "Qwen3___5" in name or "Qwen3.5" in name:
        system = (
            "Answer by joining the exact person in Evidence A to an employer, then joining that exact employer "
            "in Evidence B to a city. Ignore distractors. All required facts are present. Finish with FINAL: <city>."
        )
        example = (
            "EXAMPLE\nQUESTION\nIn which city is the employer of Mina Cole located?\n\n"
            "EVIDENCE A\nTheo Park works for Cedar Labs. Mina Cole works for Aurora Systems.\n\n"
            "EVIDENCE B\nCedar Labs is located in Rome. Aurora Systems is located in Oslo.\n\n"
            "REASONING\nMina Cole -> Aurora Systems -> Oslo\nFINAL: Oslo\n\nNOW SOLVE\n"
        )
        user = (
            example + f"QUESTION\n{row['question']}\n\nEVIDENCE A\n{row['evidence_a']}\n\n"
            f"EVIDENCE B\n{row['evidence_b']}"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    system = (
        "Use only the supplied evidence. Follow the relation from the person to the organization, "
        "then from that organization to its location. Ignore distractors. End with exactly: FINAL: <answer>."
    )
    user = (
        f"QUESTION\n{row['question']}\n\nEVIDENCE A\n{row['evidence_a']}\n\n"
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


def pack_answer(tokenizer, prompt, answer, max_length, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(" " + answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    suffix_ids = suffix_ids[: max(1, max_length - len(prompt_ids))]
    ids = torch.tensor([prompt_ids + suffix_ids], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels


def canonical_to(memory, device, dtype):
    output = {
        "keys": memory["keys"].to(device=device, dtype=dtype),
        "values": memory["values"].to(device=device, dtype=dtype),
    }
    if "answer_slot_mass" in memory:
        output["answer_slot_mass"] = memory["answer_slot_mass"].to(device=device, dtype=torch.float32)
    return output


def native_to(memory, device, dtype):
    output = {
        "keys": [tensor.to(device=device, dtype=dtype) for tensor in memory["keys"]],
        "values": [tensor.to(device=device, dtype=dtype) for tensor in memory["values"]],
    }
    if "answer_token_mask" in memory:
        output["answer_token_mask"] = memory["answer_token_mask"].to(device=device, dtype=torch.bool)
    return output


class LazyPairCache:
    def __init__(self, index_path, capacity=2):
        with open(index_path, encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.root = Path(index_path).parent
        self.entries = self.index.get("pair_files", self.index.get("pairs_index"))
        if not isinstance(self.entries, list):
            raise ValueError(f"Cache {index_path} has no pair file index")
        self.capacity = int(capacity)
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index in self.loaded:
            self.loaded.move_to_end(index)
            return self.loaded[index]
        entry = self.entries[index]
        payload = torch.load(self.root / entry["file"], map_location="cpu", weights_only=False)
        examples = payload.get("examples", payload.get("variants"))
        pair = {example["variant"]: example for example in examples}
        self.loaded[index] = pair
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
        return pair


def compatible_negative(cache, index, candidate):
    if index == candidate:
        return False
    left = cache.entries[index]
    right = cache.entries[candidate]
    left_answers = {left["base_answer"], left["counterfactual_answer"]}
    right_answers = {right["base_answer"], right["counterfactual_answer"]}
    return left_answers.isdisjoint(right_answers)


def negative_mapping(cache, seed):
    rng = random.Random(seed)
    output = []
    for index in range(len(cache)):
        candidates = list(range(len(cache)))
        rng.shuffle(candidates)
        candidate = next((value for value in candidates if compatible_negative(cache, index, value)), None)
        if candidate is None:
            raise RuntimeError(f"No compatible negative for pair {index}")
        output.append(candidate)
    return output


def assert_pair_alignment(left, right):
    if len(left) != len(right):
        raise ValueError("Pair caches differ in length")
    left_ids = [entry["pair_id"] for entry in left.entries]
    right_ids = [entry["pair_id"] for entry in right.entries]
    if left_ids != right_ids:
        raise ValueError("Pair caches are not identically ordered")


def sequence_nll(model, tokenizer, reader, row, memory, answer, max_length, device, diagnostics=None):
    ids, mask, labels = pack_answer(
        tokenizer, student_prefixed_prompt(tokenizer, row), answer, max_length, device
    )
    with reader.inject(model, memory, diagnostics):
        output = model(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
    return output.loss.float(), output.logits


def answer_swap_loss(target_nll, alternative_nll, margin):
    return F.relu(float(margin) + target_nll - alternative_nll)


def normalize_answer(text):
    return re.sub(r"^[\s`*\"']+|[\s`*\"'.,;:!?]+$", "", str(text)).casefold()


def extract_answer(text, allowed):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.I | re.S)
    values = list(dict.fromkeys([*allowed, "INSUFFICIENT"]))
    mapping = {normalize_answer(value): value for value in values}
    pattern = re.compile(
        r"(?<![\w-])(" + "|".join(sorted(map(re.escape, values), key=len, reverse=True)) + r")(?![\w-])",
        re.I,
    )
    anchored = re.findall(r"(?:FINAL|ANSWER)\s*:\s*([^\n\r]+)", clean, flags=re.I)
    for region in reversed(anchored):
        found = pattern.findall(region)
        if found:
            return mapping[normalize_answer(found[-1])], "final_anchor"
    found = pattern.findall(clean)
    if found:
        return mapping[normalize_answer(found[-1])], "last_valid_answer"
    return "", "not_found"


@torch.inference_mode()
def generate(model, tokenizer, reader, prompt, memory, max_new_tokens, device, enabled=True):
    current = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    past = None
    generated = []
    diagnostics = {}
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_reached = False
    for _ in range(max_new_tokens):
        if enabled:
            with reader.inject(model, memory, diagnostics):
                output = model(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        else:
            output = model(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        token = int(output.logits[:, -1].argmax(dim=-1).item())
        generated.append(token)
        past = output.past_key_values
        if token in eos_ids:
            eos_reached = True
            break
        current = torch.tensor([[token]], device=device)
    return {
        "token_ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "eos_reached": eos_reached,
        "diagnostics": summarize_reader_diagnostics(diagnostics),
    }


def summarize_reader_diagnostics(diagnostics):
    output = []
    for key, values in diagnostics.items():
        if not str(key).isdigit() or not isinstance(values, dict):
            continue
        calls = max(1, values.get("calls", 1))
        output.append(
            {
                "layer": int(key),
                "gate": values.get("gate", 0.0),
                "attention_entropy": values.get("attention_entropy", 0.0) / calls,
                "readout_norm": values.get("readout_norm", 0.0) / calls,
                "delta_norm": values.get("delta_norm", 0.0) / calls,
                "target_attention_mass": values.get("target_attention_mass", 0.0) / calls,
            }
        )
    return sorted(output, key=lambda row: row["layer"])


def grouped_pairs(rows, limit=0):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["variant"]] = row
    pairs = [value for value in grouped.values() if {"base", "counterfactual"}.issubset(value)]
    return pairs[:limit] if limit > 0 else pairs


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
