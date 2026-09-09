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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


SYSTEM = (
    "You are a question answering system. Answer the question using only the provided context. "
    "Return only the shortest answer span. Do not explain your reasoning. "
    'If the answer is yes or no, return exactly "yes" or "no".'
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError("Fast tokenizer with offset mappings is required")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg):
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], local_files_only=True, torch_dtype=dtype,
        attn_implementation=cfg["attention_implementation"],
    ).to(device()).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def paragraph_text(sentences):
    return "".join(sentences)


def user_text(raw, include_context=True):
    blocks = []
    if include_context:
        blocks.append("Context:")
        for index, (title, sentences) in enumerate(raw["context"], 1):
            blocks.append(f"[{index}] {title}\n{paragraph_text(sentences)}")
    blocks.append(f"Question: {raw['question']}\n\nAnswer:")
    return "\n\n".join(blocks)


def render(tok, raw, include_context=True):
    user = user_text(raw, include_context)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded.input_ids)
    template_output = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    template_ids = template_output.input_ids if hasattr(template_output, "input_ids") else template_output
    if ids != list(template_ids):
        raise RuntimeError("rendered-string tokenization differs from apply_chat_template")
    result = {"rendered_prompt": rendered, "input_ids": ids}
    if include_context:
        marker = f"Question: {raw['question']}\n\nAnswer:"
        marker_start = rendered.rfind(marker)
        if marker_start < 0:
            raise RuntimeError("Question boundary not found in rendered prompt")
        question_char = marker_start + len("Question: ")
        context_end = next(
            (i for i, (_, end) in enumerate(encoded.offset_mapping) if end > question_char),
            len(ids),
        )
        if context_end <= 0 or context_end >= len(ids):
            raise RuntimeError("invalid context/question token boundary")
        result.update({
            "context_end_index": context_end,
            "context_input_ids": ids[:context_end],
            "question_input_ids": ids[context_end:],
            "question_char_index": question_char,
            "boundary_token_offset": list(encoded.offset_mapping[context_end]),
            "split_equal": ids[:context_end] + ids[context_end:] == ids,
        })
    return result


def fixed_samples(cfg, tok):
    raw = load_json(cfg["hotpot_dev"])
    rng = random.Random(cfg["seed"])
    groups = {"bridge": [], "comparison": []}
    for row in raw:
        kind = row.get("type", "unknown")
        if kind in groups:
            groups[kind].append(row)
    for values in groups.values():
        rng.shuffle(values)
    half = cfg["formal_samples"] // 2
    chosen = groups["bridge"][:half] + groups["comparison"][:cfg["formal_samples"] - half]
    rng.shuffle(chosen)
    output = []
    for row in chosen:
        full = render(tok, row, True)
        question_only = render(tok, row, False)
        output.append({
            "id": row["_id"], "type": row.get("type", "unknown"),
            "answer": row["answer"], "question": row["question"],
            "titles": [title for title, _ in row["context"]],
            **full, "question_only_input_ids": question_only["input_ids"],
            "question_only_rendered_prompt": question_only["rendered_prompt"],
        })
    return output


def selected_samples(cfg, mode):
    rows = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "rendered_samples.jsonl")
    count = cfg["smoke_samples"] if mode == "smoke" else cfg["formal_samples"]
    if len(rows) < count:
        raise RuntimeError(f"rendered manifest has {len(rows)} rows, need {count}")
    return rows[:count]


def normalize_answer(text):
    def remove_articles(value): return re.sub(r"\b(a|an|the)\b", " ", value)
    def white_space_fix(value): return " ".join(value.split())
    def remove_punc(value): return "".join(ch for ch in value if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(text.lower())))


def answer_f1(prediction, gold):
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def dynamic_cache(model, keys, values):
    items = [(keys[i].unsqueeze(0), values[i].unsqueeze(0)) for i in range(len(keys))]
    return DynamicCache(ddp_cache_data=items, config=model.config)


def cache_tensors(cache):
    return (
        [layer.keys[0].detach().clone() for layer in cache.layers],
        [layer.values[0].detach().clone() for layer in cache.layers],
    )


def target_ids(tok, answer, maximum):
    ids = tok(answer, add_special_tokens=False).input_ids[:maximum]
    if not ids:
        ids = [tok.eos_token_id]
    return ids + [tok.eos_token_id]


def nmse(reference, other):
    return ((reference.float() - other.float()).square().mean() /
            reference.float().square().mean().clamp_min(1e-12)).item()


def vector_metrics(reference, other):
    a, b = reference.float(), other.float()
    difference = b - a
    return {
        "cosine": F.cosine_similarity(a.flatten(), b.flatten(), 0).item(),
        "nmse": nmse(a, b),
        "max_absolute_error": difference.abs().max().item(),
        "mean_absolute_error": difference.abs().mean().item(),
    }
