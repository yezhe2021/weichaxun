import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3a_common import MemoryCache, memory_for, seed_everything, sequence_nll, write_json, write_jsonl, zero_memory

P2IR = Path("/home/yezhe/伪查询/runs/p2ir_token_preserving_canonical_reader_onboarding_seed1234")
sys.path.insert(0, str(P2IR))
from p2ir_common import load_receiver
from p2ir_reader import TokenCanonicalReader, full_attention_layers


@torch.inference_mode()
def validate(model, tokenizer, reader, cache, source, indices, max_length, device):
    reader.eval(); losses = []
    for index in indices:
        payload = cache.load(index); memory = memory_for(payload, source, device)
        losses.append(float(sequence_nll(model, tokenizer, reader, payload["row"], memory, payload["row"]["answer"], max_length, device).cpu()))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--cache", required=True)
    parser.add_argument("--source", choices=("hidden", "raw_kv", "pca_kv", "canonical"), required=True)
    parser.add_argument("--profile", choices=("probe", "hotpot"), required=True); parser.add_argument("--mode", choices=("small", "full"), required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--init-checkpoint"); parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--small-samples", type=int, default=32); parser.add_argument("--rank", type=int, default=32); parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--margin", type=float, default=0.5); parser.add_argument("--contrast-weight", type=float, default=0.2)
    parser.add_argument("--gradient-accumulation", type=int, default=4); parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); cache = MemoryCache(args.cache, 3)
    if len(cache) != 512 and args.mode == "full": raise RuntimeError(f"Full training requires 512 samples, found {len(cache)}")
    model, tokenizer = load_receiver(args.model, device, torch.float16)
    all_layers = full_attention_layers(model); active_layers = all_layers if args.profile == "hotpot" else all_layers[3::4][:8]
    reader = TokenCanonicalReader(model, canonical_dim=256, rank=args.rank, max_gate=1.0, gate_init=0.02, active_layers=active_layers).to(device)
    if args.init_checkpoint:
        previous = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False); reader.load_state_dict(previous["reader"])
    if any(parameter.requires_grad for parameter in model.parameters()): raise RuntimeError("Receiver backbone must remain frozen")
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in reader.parameters()}: raise RuntimeError("Only Reader may be optimized")
    train_indices = list(range(min(args.small_samples, len(cache)))) if args.mode == "small" else list(range(448))
    validation_indices = train_indices if args.mode == "small" else list(range(448, 512))
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); history, best, best_epoch = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        reader.train(); order = train_indices[:]; random.Random(args.seed + epoch).shuffle(order); optimizer.zero_grad(set_to_none=True)
        for position, index in enumerate(tqdm(order, desc=f"p3a_{args.profile}_{args.source}_{args.mode}_{epoch}")):
            payload = cache.load(index); row = payload["row"]; correct = memory_for(payload, args.source, device)
            target = sequence_nll(model, tokenizer, reader, row, correct, row["answer"], args.max_length, device)
            other = cache.load((index + 1) % len(cache)); negative = memory_for(other, args.source, device) if (index + epoch) % 2 == 0 else zero_memory(correct)
            negative_nll = sequence_nll(model, tokenizer, reader, row, negative, row["answer"], args.max_length, device)
            contrast = F.relu(args.margin + target - negative_nll); loss = target + args.contrast_weight * contrast
            if not torch.isfinite(loss): raise RuntimeError(f"Non-finite loss at {index}")
            (loss / args.gradient_accumulation).backward()
            if any(parameter.grad is not None for parameter in model.parameters()): raise RuntimeError("Frozen backbone received gradients")
            step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            if step:
                torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            history.append({"epoch": epoch, "index": index, "target_nll": float(target.detach().cpu()), "negative_nll": float(negative_nll.detach().cpu()), "loss": float(loss.detach().cpu())})
        validation_loss = validate(model, tokenizer, reader, cache, args.source, validation_indices, args.max_length, device)
        payload_out = {"format_version": 1, "source": args.source, "profile": args.profile, "mode": args.mode, "reader": {n: v.detach().cpu() for n, v in reader.state_dict().items()}, "reader_metadata": reader.metadata(), "epoch": epoch, "validation_loss": validation_loss, "cache_index": str(Path(args.cache).resolve()), "writer_checkpoint_sha256": cache.index["writer_checkpoint_sha256"], "writer_state_sha256": cache.index["writer_state_sha256"], "args": vars(args)}
        torch.save(payload_out, output / "checkpoint_latest.pt")
        if validation_loss < best: best, best_epoch = validation_loss, epoch; torch.save(payload_out, output / "checkpoint_best.pt")
        write_jsonl(output / "history.jsonl", history)
    write_json(output / "TRAIN_SUCCESS.json", {"status": "complete", "source": args.source, "profile": args.profile, "mode": args.mode, "epochs": args.epochs, "best_epoch": best_epoch, "validation_loss": best, "only_reader_trainable": True, "active_layers": active_layers})


if __name__ == "__main__": main()
