import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p2is_common import (
    PairCache, TokenCanonicalReader, canonical_to, full_attention_layers, load_receiver,
    pack_answer, parse_dtype, resolve_device, student_prefixed_prompt, write_json,
)


@torch.inference_mode()
def encode(model, tokenizer, reader, prompt_row, row, memory, max_length, device, topk):
    ids, mask, labels = pack_answer(tokenizer, student_prefixed_prompt(tokenizer, prompt_row), row["answer"], max_length, device)
    with reader.inject(model, memory):
        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True)
    shift_logits = output.logits[:, :-1].float(); shift_labels = labels[:, 1:]
    selected = shift_logits[shift_labels != -100]
    values, indices = selected.topk(min(topk, selected.shape[-1]), dim=-1)
    return {"answer_nll": float(output.loss.cpu()), "top_indices": indices.cpu(), "top_values": values.half().cpu(), "target_token_ids": shift_labels[shift_labels != -100].cpu()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--receiver-name", required=True); parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--reader-checkpoint", required=True); parser.add_argument("--old-index", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--topk", type=int, default=128); parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--dtype", default="float16")
    args = parser.parse_args(); device = resolve_device(args.device); dtype = parse_dtype(args.dtype, device); cache = PairCache(args.old_index, 2)
    checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["writer_checkpoint_sha256"] != cache.index["writer_checkpoint_sha256"]: raise RuntimeError("Reader and old Canonical cache hashes differ")
    model, tokenizer = load_receiver(args.receiver_model, device, dtype); metadata = checkpoint["reader_metadata"]
    reader = TokenCanonicalReader(model, canonical_dim=256, rank=metadata["rank"], max_gate=metadata["max_gate"], gate_init=0.0, active_layers=metadata["active_layers"]).to(device).eval()
    reader.load_state_dict(checkpoint["reader"])
    for parameter in reader.parameters(): parameter.requires_grad_(False)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); entries = []
    count = min(len(cache), args.max_pairs) if args.max_pairs > 0 else len(cache)
    for index in tqdm(range(count), desc=f"cache_teacher_{args.receiver_name}"):
        pair = cache.load(index); rows = []; prompt_row = pair["base"]
        for variant in ("base", "counterfactual"):
            row = pair[variant]; teacher = encode(model, tokenizer, reader, prompt_row, row, canonical_to(row["memory"], device), args.max_length, device, args.topk)
            rows.append({"pair_id": row["pair_id"], "variant": variant, "answer": row["answer"], "teacher": teacher})
        filename = f"pair_{index:05d}.pt"; torch.save({"pair_id": rows[0]["pair_id"], "variants": rows}, output / filename)
        entries.append({"pair_id": rows[0]["pair_id"], "file": filename, "base_answer": rows[0]["answer"], "counterfactual_answer": rows[1]["answer"]})
    write_json(output / "index.json", {"format_version": 1, "receiver": args.receiver_name, "pairs": len(entries), "topk": args.topk, "old_writer_hash": cache.index["writer_checkpoint_sha256"], "pair_files": entries})
    write_json(output / "CACHE_SUCCESS.json", {"status": "complete", "receiver": args.receiver_name, "pairs": len(entries)})


if __name__ == "__main__": main()
