import argparse

import torch

from p3d3_common import LayerAlignedNativeQueryReader, SELECTED_LAYERS, generate, load_receiver


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); model, tokenizer = load_receiver(args.model, device)
    reader = LayerAlignedNativeQueryReader(model, 256, SELECTED_LAYERS, rank=32, gate_init=0.01).to(device).eval()
    tokens = 7
    memory = {"keys": torch.randn(16, tokens, 256, device=device), "values": torch.randn(16, tokens, 256, device=device),
              "mask": torch.ones(tokens, dtype=torch.bool, device=device), "support_mask": torch.ones(tokens, dtype=torch.bool, device=device)}
    row = {"question": "Which city is named in the evidence?", "answer": "Paris"}
    trace = {}; output = generate(model, tokenizer, reader, row, memory, max_new_tokens=1, enabled=True, trace=trace)
    if len(trace) != 16: raise RuntimeError(f"Expected 16 Reader traces, got {len(trace)}")
    for layer in SELECTED_LAYERS:
        if not trace.get(layer): raise RuntimeError(f"Missing trace for Receiver layer {layer}")
        attention = trace[layer][0]["attention"]
        if attention.shape[-1] != tokens: raise RuntimeError("External attention token dimension mismatch")
    print({"status": "complete", "reader_layers": len(trace), "metadata": reader.metadata(), "generated_tokens": len(output["token_ids"])})


if __name__ == "__main__": main()
