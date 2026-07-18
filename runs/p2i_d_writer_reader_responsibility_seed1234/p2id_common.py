import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_P2I_ROOT = Path(
    os.environ.get(
        "P2I_ROOT",
        ROOT.parent / "p2i_cached_canonical_evidence_kv_qwen3_8b_seed1234",
    )
)


def add_p2i_path(path=None):
    root = Path(path or DEFAULT_P2I_ROOT).resolve()
    if not (root / "canonical_modules.py").is_file():
        raise FileNotFoundError(f"P2-I dependency is missing: {root}")
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)
    return root


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def finite_gradients(named_parameters):
    return [
        name for name, parameter in named_parameters
        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]


def condition_metrics(records):
    conditions = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        conditions.append(
            {
                "condition": condition,
                "n": len(selected),
                "original_target_accuracy": float(np.mean([row["original_target_correct"] for row in selected])),
                "source_memory_accuracy": float(np.mean([row["source_memory_correct"] for row in selected])),
                "mean_confidence": float(np.mean([row["confidence"] for row in selected])),
            }
        )
    return conditions


def paired_consistency(records, condition="correct"):
    grouped = {}
    for row in records:
        if row["condition"] == condition:
            grouped.setdefault(row["pair_id"], {})[row["variant"]] = row
    complete = [pair for pair in grouped.values() if {"base", "counterfactual"}.issubset(pair)]
    if not complete:
        return 0.0
    return float(np.mean([
        pair["base"]["original_target_correct"] * pair["counterfactual"]["original_target_correct"]
        for pair in complete
    ]))


def deterministic_negative(entries, index):
    answers = {entries[index]["base_answer"], entries[index]["counterfactual_answer"]}
    for offset in range(1, len(entries)):
        candidate = (index + offset) % len(entries)
        other = {entries[candidate]["base_answer"], entries[candidate]["counterfactual_answer"]}
        if answers.isdisjoint(other):
            return candidate
    raise RuntimeError(f"No answer-disjoint negative for pair {index}")


def checkpoint_reader(checkpoint, receiver_name):
    if receiver_name not in checkpoint.get("readers", {}):
        raise ValueError(f"Checkpoint has no Reader for {receiver_name}")
    return checkpoint["readers"][receiver_name], checkpoint["reader_metadata"][receiver_name]
