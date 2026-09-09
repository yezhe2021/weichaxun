from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from common import load_json, save_json


def file_hash(path):
    digest = hashlib.sha256(); digest.update(Path(path).read_bytes()); return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args(); cfg = load_json(args.config)
    model_path, model4_path = Path(cfg["model_path"]), Path(cfg["model_4b_path"])
    required = [model_path / "config.json", model_path / "tokenizer_config.json"]
    if not model_path.is_dir() or any(not path.is_file() for path in required):
        raise RuntimeError(f"Qwen3-1.7B model is unavailable or incomplete at {model_path}")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    actual = {"num_hidden_layers": config.num_hidden_layers, "num_key_value_heads": config.num_key_value_heads, "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)}
    expected = {"num_hidden_layers": 28, "num_key_value_heads": 8, "head_dim": 128}
    if actual != expected: raise RuntimeError(f"unexpected Qwen3-1.7B structure: {actual}")
    tok17 = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    tok4 = AutoTokenizer.from_pretrained(model4_path, local_files_only=True, use_fast=True)
    tokenizer_fields = ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id")
    comparison = {field: {"qwen3_1_7b": getattr(tok17, field), "qwen3_4b": getattr(tok4, field)} for field in tokenizer_fields}
    comparison["class"] = {"qwen3_1_7b": tok17.__class__.__name__, "qwen3_4b": tok4.__class__.__name__}
    comparison["chat_template_equal"] = tok17.chat_template == tok4.chat_template
    if any(value["qwen3_1_7b"] != value["qwen3_4b"] for key, value in comparison.items() if isinstance(value, dict)) or not comparison["chat_template_equal"]:
        raise RuntimeError(f"1.7B/4B tokenizer protocol mismatch: {comparison}")
    save_json(Path(cfg["work_dir"]) / "artifacts" / "model_tokenizer_preflight.json", {"passed": True, "structure": actual, "comparison": comparison, "model_config_sha256": file_hash(model_path / "config.json")})
    print("Qwen3-1.7B model/tokenizer preflight passed", flush=True)


if __name__ == "__main__": main()
