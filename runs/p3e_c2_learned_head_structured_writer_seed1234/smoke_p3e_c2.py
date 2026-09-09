import argparse

import torch

from p3d3_common import load_receiver, pack_answer, seed_everything
from p3e_c1_common import LearnableCanonicalHeadReader
from p3e_c2_common import HeadStructuredWriter, SenderNativeHeadwiseCache, writer_memory


@torch.inference_mode()
def logits(model, tokenizer, reader, row, memory, max_length=512):
    ids, mask, labels = pack_answer(tokenizer, row, row["answer"], max_length, model.device)
    with reader.inject(model, memory): output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    selected = labels[:, 1:] != -100; return output.logits[:, :-1][selected]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--c1-reader", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(1234); device = torch.device(args.device); payload = SenderNativeHeadwiseCache(args.memory).load(0)
    writer = HeadStructuredWriter().to(device).eval(); memory = writer_memory(writer, payload, device, no_grad=True)
    expected_keys = payload["keys"].float().to(device).repeat_interleave(2, dim=2); expected_values = payload["values"].float().to(device).repeat_interleave(2, dim=2)
    key_error = float((memory["keys"] - expected_keys).abs().max()); value_error = float((memory["values"] - expected_values).abs().max())
    route = writer.routing_weights().detach().cpu(); expected_route = torch.zeros_like(route)
    for head in range(16): expected_route[:, head, head // 2] = 1.0
    route_error = float((route - expected_route).abs().max())
    model, tokenizer = load_receiver(args.model, device); checkpoint = torch.load(args.c1_reader, map_location="cpu", weights_only=False); metadata = checkpoint["reader_metadata"]
    reader = LearnableCanonicalHeadReader(model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"], metadata["top_k"], 0.25).to(device).eval(); reader.load_state_dict(checkpoint["reader"])
    reference = {"keys": expected_keys, "values": expected_values, "mask": memory["mask"], "support_mask": memory["support_mask"]}
    logit_error = float((logits(model, tokenizer, reader, payload["row"], memory) - logits(model, tokenizer, reader, payload["row"], reference)).float().abs().max())
    if max(key_error, value_error, route_error, logit_error) >= 1e-6: raise RuntimeError(f"Writer duplicate initialization failed: K={key_error}, V={value_error}, route={route_error}, logits={logit_error}")
    print({"status": "passed", "key_error": key_error, "value_error": value_error, "route_error": route_error, "logit_error": logit_error})


if __name__ == "__main__": main()
