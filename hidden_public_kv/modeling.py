import gc

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from .core import decoder_layers, text_config


QWEN3_TAPS = (3, 8, 12, 17, 21, 26, 30, 35)
QWEN35_TAPS = (3, 7, 11, 15, 19, 23, 27, 31)


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_frozen_model(path, kind: str, device="cuda", dtype=torch.float16, gradient_checkpointing=False):
    model_class = AutoModelForImageTextToText if kind == "qwen35" else AutoModelForCausalLM
    model = model_class.from_pretrained(
        path, dtype=dtype, trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model


def validate_architecture(model, kind: str):
    config = text_config(model)
    expected_layers = 32 if kind == "qwen35" else 36
    if len(decoder_layers(model)) != expected_layers or int(config.hidden_size) != 2560:
        raise RuntimeError(f"Unexpected {kind} architecture")
    if kind == "qwen35":
        kinds = list(config.layer_types)
        full = tuple(index for index, value in enumerate(kinds) if value == "full_attention")
        if full != QWEN35_TAPS:
            raise RuntimeError(f"Unexpected Qwen3.5 stage ends: {full}")


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
