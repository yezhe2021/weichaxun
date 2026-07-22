import argparse

import torch

from p3d3_common import SELECTED_LAYERS, generate, load_receiver
from p3e_a_common import NativeHeadwiseReader


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); model, tokenizer = load_receiver(args.model, device)
    reader = NativeHeadwiseReader(model, SELECTED_LAYERS, rank=32, gate_init=0.01).to(device).eval(); tokens = 7
    memory = {"keys": torch.randn(16, tokens, 8, 128, device=device), "values": torch.randn(16, tokens, 8, 128, device=device),
              "mask": torch.ones(tokens, dtype=torch.bool, device=device), "support_mask": torch.ones(tokens, dtype=torch.bool, device=device)}
    trace = {}; output = generate(model, tokenizer, reader, {"question": "Which city is named?", "answer": "Paris"}, memory, 1, trace=trace)
    if len(trace) != 16: raise RuntimeError(f"Expected 16 injected layers, got {len(trace)}")
    for layer in SELECTED_LAYERS:
        attention = trace[layer][0]["attention"]
        if attention.shape[-3:] != (8, 4, tokens): raise RuntimeError(f"GQA attention shape mismatch: {tuple(attention.shape)}")
    print({"status": "complete", "reader_layers": len(trace), "metadata": reader.metadata(), "generated_tokens": len(output["token_ids"])})


if __name__ == "__main__": main()
