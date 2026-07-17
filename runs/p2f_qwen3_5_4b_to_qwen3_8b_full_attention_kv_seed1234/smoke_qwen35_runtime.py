import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from cache_qwen35_native_kv import (
    full_attention_layer_indices,
    project_native_kv,
    text_backbone,
)
from p2a_common import parse_dtype, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    indices = full_attention_layer_indices(model)
    if indices != [3, 7, 11, 15, 19, 23, 27, 31]:
        raise RuntimeError(f"Unexpected Qwen3.5 full-attention layers: {indices}")
    encoded = tokenizer("Question: Where? Evidence: Alpha is in Rome.", return_tensors="pt")
    captured = {}

    def hook(module, hook_args, kwargs):
        hidden = kwargs.get("hidden_states", hook_args[0] if hook_args else None)
        key, value = project_native_kv(module, hidden)
        captured["key"] = key
        captured["value"] = value

    attention = text_backbone(model).layers[indices[0]].self_attn
    handle = attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=encoded.input_ids.to(device),
                attention_mask=encoded.attention_mask.to(device),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        handle.remove()
    if tuple(captured["key"].shape[:2]) != (1, 4):
        raise RuntimeError(f"Unexpected K shape: {tuple(captured['key'].shape)}")
    if captured["key"].shape[-1] != 256 or captured["value"].shape[-1] != 256:
        raise RuntimeError("Unexpected Qwen3.5 full-attention head dimension")
    if not output.hidden_states or output.hidden_states[-1].shape[-1] != 2560:
        raise RuntimeError("Qwen3.5 hidden-state capture failed")
    print("P2-F Qwen3.5 runtime smoke passed")


if __name__ == "__main__":
    main()
