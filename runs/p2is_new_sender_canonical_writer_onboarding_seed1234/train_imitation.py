import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from p2is_common import (
    PairCache, Sender4CanonicalWriter, anchor_terms, load_aligned_pair, seed_everything,
    state_sha256, write_json, write_jsonl,
)


@torch.inference_mode()
def validate(writer, q4, old, indices, device):
    writer.eval(); rows = []
    for index in indices:
        qpair, opair = load_aligned_pair(q4, old, index)
        for variant in ("base", "counterfactual"):
            output = writer(qpair[variant]["key_flat"].to(device), qpair[variant]["value_flat"].to(device))
            target = {name: opair[variant]["memory"][name].to(device) for name in ("keys", "values")}
            losses = anchor_terms(output, target)
            rows.append({name: float(value.cpu()) for name, value in losses.items()})
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--q4-index", required=True); parser.add_argument("--old-index", required=True)
    parser.add_argument("--ridge", required=True); parser.add_argument("--out", required=True); parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rank", type=int, default=64); parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8); parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device)
    q4, old = PairCache(args.q4_index, 3), PairCache(args.old_index, 3)
    if len(q4) != 512 or len(old) != 512: raise ValueError("Imitation requires exact aligned 512-pair train caches")
    ridge = torch.load(args.ridge, map_location="cpu", weights_only=False)
    config = {"dim": 256, "rank": args.rank, "freeze_base": True}
    writer = Sender4CanonicalWriter(ridge, **config).to(device)
    trainable = [parameter for parameter in writer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    output_dir = Path(args.out); output_dir.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        writer.train(); order = list(range(448)); random.Random(args.seed + epoch).shuffle(order); optimizer.zero_grad(set_to_none=True)
        losses_epoch = []
        for position, index in enumerate(tqdm(order, desc=f"p2is_imitation_{epoch}")):
            qpair, opair = load_aligned_pair(q4, old, index); total = torch.zeros((), device=device); local = []
            for variant in ("base", "counterfactual"):
                prediction = writer(qpair[variant]["key_flat"].to(device), qpair[variant]["value_flat"].to(device))
                target = {name: opair[variant]["memory"][name].to(device) for name in ("keys", "values")}
                terms = anchor_terms(prediction, target); total = total + 0.5 * terms["total"]; local.append(terms)
            (total / args.gradient_accumulation).backward()
            if any(parameter.grad is not None for parameter in list(writer.key_base.parameters()) + list(writer.value_base.parameters())):
                raise RuntimeError("Frozen ridge base projection received gradients")
            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            if should_step:
                bad = [name for name, parameter in writer.named_parameters() if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
                if bad: raise RuntimeError(f"Non-finite imitation gradients: {bad[:8]}")
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses_epoch.append(float(total.detach().cpu()))
        validation = validate(writer, q4, old, range(448, 512), device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses_epoch)), **{f"validation_{name}": value for name, value in validation.items()}}
        history.append(row); write_jsonl(output_dir / "history.jsonl", history)
        payload = {
            "format_version": 1, "stage": "canonical_imitation", "writer": {name: value.detach().cpu() for name, value in writer.state_dict().items()},
            "writer_config": config, "writer_state_sha256": state_sha256(writer.state_dict()), "ridge_file": str(Path(args.ridge).resolve()),
            "epoch": epoch, "validation": validation, "args": vars(args),
        }
        torch.save(payload, output_dir / "checkpoint_latest.pt")
        if validation["total"] < best:
            best, best_epoch = validation["total"], epoch; torch.save(payload, output_dir / "checkpoint_best.pt")
    write_json(output_dir / "TRAIN_SUCCESS.json", {"status": "complete", "best_epoch": best_epoch, "validation_anchor": best, "base_projection_frozen": True})


if __name__ == "__main__": main()
