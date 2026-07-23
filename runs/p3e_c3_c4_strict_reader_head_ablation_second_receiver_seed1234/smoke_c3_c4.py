import argparse

import torch

from p3d3_common import answer_mean_nll, forward_answer, load_receiver, pack_answer, seed_everything
from p3e_c1_common import LearnableCanonicalHeadReader
from p3e_c2_common import SenderNativeHeadwiseCache, load_writer, writer_memory
from p3e_c3_common import initialize_fresh_reader
from p3e_c4_common import Qwen35CanonicalReader, load_qwen35


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--qwen4", required=True); parser.add_argument("--qwen35", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--writer", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(1234); device = torch.device(args.device); payload = SenderNativeHeadwiseCache(args.memory).load(0)
    writer, _ = load_writer(args.writer, device); writer.requires_grad_(False); writer.eval(); memory = writer_memory(writer, payload, device, no_grad=True)
    q4, tokenizer = load_receiver(args.qwen4, device); q4.requires_grad_(False); reader = LearnableCanonicalHeadReader(q4, payload["metadata"].get("selected_layers", [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 35]), 32, 0.01, 2, 1.0).to(device)
    initialize_fresh_reader(reader, "fully_random", 1234); loss = forward_answer(q4, tokenizer, reader, payload["row"], memory, 512, device); loss.backward()
    if not any(parameter.grad is not None for parameter in reader.parameters()) or any(parameter.grad is not None for parameter in writer.parameters()): raise RuntimeError("C3-A gradient smoke failed")
    del reader, q4, tokenizer; torch.cuda.empty_cache()
    q35, tokenizer35 = load_qwen35(args.qwen35, device); reader35 = Qwen35CanonicalReader(q35, seed=1234).to(device)
    ids, mask, labels = pack_answer(tokenizer35, payload["row"], payload["row"]["answer"], 512, device)
    with reader35.inject(q35, memory): output = q35(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    loss35 = answer_mean_nll(output.logits, labels); loss35.backward()
    if not any(parameter.grad is not None for parameter in reader35.parameters()) or any(parameter.grad is not None for parameter in writer.parameters()): raise RuntimeError("C4 gradient smoke failed")
    print({"status": "passed", "c3a_gradient": True, "c4_gradient": True})


if __name__ == "__main__": main()
