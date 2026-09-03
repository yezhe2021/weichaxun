import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import load_receiver, seed_everything
from p3e_f_common import sha256_file, write_json, write_jsonl
from p3e_j_common import (
    PairedCanonicalTokenCache,
    load_decoder,
    nearest_token_metrics,
    reconstruction_metrics,
    verify_tokenizer_alignment,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--canonical-index", required=True)
    parser.add_argument("--native-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = PairedCanonicalTokenCache(
        args.canonical_index, args.native_index, args.data, capacity=2
    )
    count = min(args.max_samples, len(cache))
    model, tokenizer = load_receiver(args.model, device)
    decoder, checkpoint = load_decoder(args.decoder, device)
    decoder.requires_grad_(False)
    decoder.eval()
    rows, nearest_predicted, nearest_ids = [], [], []
    with torch.inference_mode():
        for index in tqdm(range(count), desc="p3e_j_stage_a_reconstruction_eval"):
            payload = cache.load(index)
            verify_tokenizer_alignment(tokenizer, payload)
            keys = payload["keys"].to(device)
            values = payload["values"].to(device)
            mask = payload["mask"].to(device)
            token_ids = payload["token_ids"].to(device)
            target = model.get_input_embeddings()(token_ids).float()
            predicted = decoder(keys, values, mask)
            rows.append({
                "id": payload["row"]["id"],
                **reconstruction_metrics(predicted, target, token_ids, mask),
            })
            remaining = 512 - sum(value.numel() for value in nearest_ids)
            if remaining > 0:
                positions = torch.nonzero(mask, as_tuple=False).flatten()[:remaining]
                nearest_predicted.append(predicted[positions])
                nearest_ids.append(token_ids[positions])
    write_jsonl(output / "per_sample_reconstruction.jsonl", rows)
    total_tokens = sum(row["tokens"] for row in rows)
    nearest_tokens = torch.cat(nearest_ids)
    nearest = nearest_token_metrics(
        torch.cat(nearest_predicted),
        nearest_tokens,
        torch.ones_like(nearest_tokens, dtype=torch.bool),
        model.get_input_embeddings().weight,
        max_positions=512,
    )
    write_json(output / "SUCCESS.json", {
        "status": "complete",
        "experiment": "P3-E-J Stage-A embedding reconstruction validation",
        "samples": count,
        "tokens": total_tokens,
        "embedding_cosine": sum(
            row["embedding_cosine"] * row["tokens"] for row in rows
        ) / total_tokens,
        "embedding_mse": sum(
            row["embedding_mse"] * row["tokens"] for row in rows
        ) / total_tokens,
        "nearest_neighbor": {
            **nearest,
            "scope": "first_512_valid_validation_positions_in_fixed_sample_order",
        },
        "decoder": args.decoder,
        "decoder_sha256": sha256_file(args.decoder),
        "decoder_stage": checkpoint["stage"],
    })


if __name__ == "__main__":
    main()
