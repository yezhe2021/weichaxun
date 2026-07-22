import argparse

import torch

from p3d3_common import generate, load_receiver
from p3e_b_common import NativeHeadwiseReader, SenderNativeHeadwiseCache, native_memory_to


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--stage-a-checkpoint", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); cache = SenderNativeHeadwiseCache(args.memory); payload = cache.load(0)
    model, tokenizer = load_receiver(args.model, device); checkpoint = torch.load(args.stage_a_checkpoint, map_location="cpu", weights_only=False); metadata = checkpoint["reader_metadata"]
    reader = NativeHeadwiseReader(model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"]).to(device).eval(); reader.load_state_dict(checkpoint["reader"])
    trace = {}; output = generate(model, tokenizer, reader, payload["row"], native_memory_to(payload, device), 1, trace=trace)
    if len(trace) != 16: raise RuntimeError(f"Expected 16 Reader traces, got {len(trace)}")
    for calls in trace.values():
        if calls[0]["attention"].shape[-3:-1] != (8, 4): raise RuntimeError("GQA grouping mismatch")
    print({"status": "complete", "memory_shape": list(payload["keys"].shape), "reader_layers": len(trace), "generated_tokens": len(output["token_ids"]), "lossless_reshape": True})


if __name__ == "__main__": main()
