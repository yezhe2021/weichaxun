import argparse

import torch

from canonical_modules import CanonicalEvidenceWriter, CanonicalExternalReader, full_attention_layers
from p2i_common import LazyPairCache, load_receiver, native_to, parse_dtype, resolve_device, student_prefixed_prompt


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="One-pair runtime smoke test for a P2-I Receiver")
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--sender-index", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    cache = LazyPairCache(args.sender_index, capacity=1)
    row = cache.load(0)["base"]
    writer = CanonicalEvidenceWriter().to(device).eval()
    memory = writer(native_to(row["memory"], device, dtype), output_dtype=dtype)
    model, tokenizer = load_receiver(args.receiver_model, device, dtype)
    reader = CanonicalExternalReader(
        model, canonical_dim=256, adapter_rank=32, active_layers=full_attention_layers(model)
    ).to(device).eval()
    encoded = tokenizer(
        student_prefixed_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False
    )
    diagnostics = {}
    with reader.inject(model, memory, diagnostics):
        first = model(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=True,
            return_dict=True,
        )
    token = first.logits[:, -1].argmax(-1, keepdim=True)
    with reader.inject(model, memory, diagnostics):
        second = model(
            input_ids=token,
            past_key_values=first.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    if not torch.isfinite(first.logits).all() or not torch.isfinite(second.logits).all():
        raise RuntimeError("Reader runtime produced non-finite logits")
    expected = set(full_attention_layers(model))
    observed = {int(key) for key in diagnostics if str(key).isdigit()}
    if observed != expected:
        raise RuntimeError(f"Reader hooks missed layers: expected={sorted(expected)} observed={sorted(observed)}")
    print(
        {
            "status": "complete",
            "receiver_model": args.receiver_model,
            "active_layers": sorted(observed),
            "canonical_shape": tuple(memory["keys"].shape),
            "decode_step_finite": True,
        }
    )


if __name__ == "__main__":
    main()
