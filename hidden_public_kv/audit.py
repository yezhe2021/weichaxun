import argparse

import torch

from .core import PublicReader, PublicWriter, capture_hidden_taps
from .modeling import QWEN35_TAPS, load_frozen_model, load_tokenizer, validate_architecture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.sender_model)
    model = load_frozen_model(args.sender_model, "qwen35", args.device, torch.float16)
    validate_architecture(model, "qwen35")
    encoded = tokenizer("Alpha is located in Rome.", return_tensors="pt")
    encoded = {name: value.to(args.device) for name, value in encoded.items()}
    hidden, output = capture_hidden_taps(model, encoded, QWEN35_TAPS, use_cache=True)
    writer = PublicWriter(2560).to(device=args.device, dtype=torch.float16)
    memory = writer(hidden, encoded["attention_mask"])
    memory.validate(8, 8, 128)
    if output.past_key_values is None:
        raise RuntimeError("Sender native cache was not produced alongside Public KV taps")
    if any(value.requires_grad or value.grad_fn is not None for value in hidden):
        raise RuntimeError("Sender hidden taps were not detached")
    print("Qwen3.5 native-cache + detached hidden-to-Public-KV audit passed")


if __name__ == "__main__":
    main()
