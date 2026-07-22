import argparse

import torch

from p3d3_common import load_receiver, pack_answer, seed_everything
from p3e_c0_common import DiagnosticHeadwiseReader
from p3e_c1_common import DuplicateHeadwiseCache, LearnableCanonicalHeadReader, duplicate_memory_to


@torch.inference_mode()
def answer_logits(model, tokenizer, reader, row, memory, max_length):
    ids, mask, labels = pack_answer(tokenizer, row, row["answer"], max_length, model.device)
    with reader.inject(model, memory): output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    positions = labels[:, 1:] != -100
    return output.logits[:, :-1][positions]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--native-reader", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); cache = DuplicateHeadwiseCache(args.memory); payload = cache.load(0)
    model, tokenizer = load_receiver(args.model, device); checkpoint = torch.load(args.native_reader, map_location="cpu", weights_only=False); metadata = checkpoint["reader_metadata"]
    fixed = DiagnosticHeadwiseReader(model, metadata["selected_layers"], True, metadata["rank"], metadata["gate_init"]).to(device).eval(); fixed.load_state_dict(checkpoint["reader"])
    learnable = LearnableCanonicalHeadReader(model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"], 2, 1.0).to(device).eval(); learnable.load_native_reader(checkpoint)
    memory = duplicate_memory_to(payload, device); fixed_logits = answer_logits(model, tokenizer, fixed, payload["row"], memory, args.max_length)
    learnable_logits = answer_logits(model, tokenizer, learnable, payload["row"], memory, args.max_length)
    error = float((fixed_logits.float() - learnable_logits.float()).abs().max().cpu()); routes = learnable.routes().detach().cpu()
    expected = torch.zeros_like(routes)
    for query_head in range(32):
        pair = 2 * (query_head // 4); expected[:, query_head, pair:pair + 2] = 0.5
    route_error = float((routes - expected).abs().max())
    if error >= 1e-4 or route_error != 0.0: raise RuntimeError(f"C0 warm-start equivalence failed: logits={error}, route={route_error}")
    print({"status": "passed", "answer_logit_max_error": error, "route_max_error": route_error})


if __name__ == "__main__": main()
