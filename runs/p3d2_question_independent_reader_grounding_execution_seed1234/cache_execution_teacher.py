import argparse
from contextlib import contextmanager
from pathlib import Path

import torch
from tqdm import tqdm

from p3d2_common import (
    build_cache, decoder_layers, execution_text_prompt, gold_logit_trace, load_receiver, pack_answer,
    prediction_position_mask, question_prompt, read_json, seed_everything, write_json,
)


@contextmanager
def capture_layer_outputs(model, layers):
    captured, handles = {}, []
    for layer_index in layers:
        def hook(module, args, kwargs, output, layer_index=layer_index):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = hidden.detach()
        handles.append(decoder_layers(model)[layer_index].register_forward_hook(hook, with_kwargs=True))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def trace(model, tokenizer, row, prompt, max_length, device, layers):
    ids, mask, labels = pack_answer(tokenizer, prompt, row["answer"], max_length, device)
    answer_mask = prediction_position_mask(labels)
    question_position = int(labels[0].ne(-100).nonzero()[0]) - 1
    with capture_layer_outputs(model, layers) as captured:
        output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    hidden = torch.stack([captured[layer][0, answer_mask].float().cpu() for layer in layers])
    question_state = torch.stack([captured[layer][0, question_position].float().cpu() for layer in layers])
    return {
        "hidden": hidden,
        "question_state": question_state,
        "gold_logits": gold_logit_trace(output.logits, labels).float().cpu(),
        "answer_token_ids": labels[0, 1:][labels[0, 1:].ne(-100)].cpu(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    protocol = read_json(args.protocol)
    cache = build_cache(protocol, args.split)
    model, tokenizer = load_receiver(args.model, device)
    layers = list(range(36))
    limit = min(len(cache), args.max_samples or len(cache))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in tqdm(range(limit), desc=f"p3d2_teacher_{args.split}"):
        payload = cache.load(index)
        row = payload["row"]
        question = trace(model, tokenizer, row, question_prompt(tokenizer, row), args.max_length, device, layers)
        full_text = trace(model, tokenizer, row, execution_text_prompt(tokenizer, row), args.max_length, device, layers)
        if not torch.equal(question["answer_token_ids"], full_text["answer_token_ids"]):
            raise RuntimeError(f"Teacher answer-token mismatch for {row['id']}")
        item = {
            "sample_id": row["id"], "answer_token_ids": question["answer_token_ids"],
            "question_hidden": question["hidden"].half(),
            "question_state": question["question_state"].half(),
            "text_delta": (full_text["hidden"] - question["hidden"]).half(),
            "question_gold_logits": question["gold_logits"].half(),
            "text_gold_logit_delta": (full_text["gold_logits"] - question["gold_logits"]).half(),
        }
        filename = f"sample_{index:05d}.pt"
        torch.save(item, output / filename)
        entries.append({"file": filename, "id": row["id"], "answer_tokens": len(item["answer_token_ids"])})
    write_json(output / "index.json", {
        "status": "complete", "split": args.split, "model": args.model, "layers": layers,
        "samples": limit, "entries": entries,
        "alignment": "answer-token positions only; no full-sequence hidden matching",
    })


if __name__ == "__main__":
    main()
