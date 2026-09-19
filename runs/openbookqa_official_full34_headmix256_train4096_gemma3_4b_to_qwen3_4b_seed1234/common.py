import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_tensor(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp); tmp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def configuration(mode):
    cfg = read_json(ROOT / "config.json")
    if mode == "smoke":
        cfg.update(train_samples=2, validation_samples=2, test_samples=2, audit_samples=2,
                   stage_a_epochs=1, stage_a_steps=1, stage_b_epochs=1, stage_b_steps=1,
                   batch_size=2, validation_interval_steps=1)
    cfg["mode"] = mode
    cfg["signature"] = digest(cfg)
    return cfg


def run_root(cfg):
    return ROOT / "runs" / cfg["mode"]


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def tokenizer(path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def text_config(model_or_config):
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "text_config", config)


def text_backbone(model):
    inner = model.model
    return getattr(inner, "language_model", inner)


def geometry(model_or_config):
    config = text_config(model_or_config)
    return config.num_hidden_layers, config.num_key_value_heads, config.head_dim


def load_model(cfg, family):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    dtype = torch.float32 if family == "gemma" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg["models"][family], local_files_only=True, dtype=dtype,
        attn_implementation=cfg["attention_implementation"])
    if family == "gemma":
        # The experiment is text-only. Drop the unused vision modules before the CUDA transfer.
        del model.model.vision_tower
        del model.model.multi_modal_projector
    model = model.to("cuda").eval().requires_grad_(False)
    expected = {"qwen": (36, 8, 128), "gemma": (34, 4, 256)}[family]
    observed = geometry(model)
    if observed != expected: raise RuntimeError(f"{family} architecture mismatch: {observed}")
    return model
