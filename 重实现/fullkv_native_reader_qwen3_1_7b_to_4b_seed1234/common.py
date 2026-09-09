from __future__ import annotations

import hashlib
import json
import random
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = (
    "You are a question answering system. Answer the question using only the provided context. "
    "Return only the shortest answer span. Do not explain your reasoning. "
    'If the answer is yes or no, return exactly "yes" or "no".'
)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def progress(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_for(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(path: str, cfg: dict[str, Any], frozen: bool = True):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype_for(cfg["dtype"]),
        attn_implementation=cfg["attention_implementation"],
    ).to(cuda()).eval()
    if frozen:
        model.requires_grad_(False)
    return model


def sha256_ints(values: Iterable[int]) -> str:
    payload = ",".join(str(int(value)) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(character for character in value if character not in set(string.punctuation))

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def answer_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    target = normalize_answer(gold).split()
    if not predicted or not target:
        return float(predicted == target)
    overlap = sum((Counter(predicted) & Counter(target)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def work_path(cfg: dict[str, Any], *parts: str) -> Path:
    return Path(cfg["work_dir"]).joinpath(*parts)

