import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from p2ir_common import (
    PairCache, canonical_to, load_receiver, parse_dtype, resolve_device, seed_everything,
    sequence_nll, write_json, write_jsonl,
)
from p2ir_reader import TokenCanonicalReader, full_attention_layers


@torch.inference_mode()
def validate(model, tokenizer, reader, cache, indices, max_length, device, margin):
    reader.eval(); losses, base_ok, cf_ok = [], [], []
    for index in indices:
        pair = cache.load(index); local = {}
        prompt_row = pair["base"]
        for variant in ("base", "counterfactual"):
            row = pair[variant]; other = pair["counterfactual" if variant == "base" else "base"]
            memory = canonical_to(row["memory"], device)
            target = sequence_nll(model, tokenizer, reader, prompt_row, memory, row["answer"], max_length, device)
            alternative = sequence_nll(model, tokenizer, reader, prompt_row, memory, other["answer"], max_length, device)
            losses.append(float((target + F.relu(margin + target - alternative)).cpu()))
            local[variant] = float(target < alternative)
        base_ok.append(local["base"]); cf_ok.append(local["counterfactual"])
    return {
        "loss": float(np.mean(losses)), "base_answer_preference": float(np.mean(base_ok)),
        "counterfactual_answer_preference": float(np.mean(cf_ok)),
        "paired_answer_preference": float(np.mean(np.asarray(base_ok) * np.asarray(cf_ok))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-name", choices=("qwen3_4b", "qwen3_5_4b"), required=True)
    parser.add_argument("--receiver-model", required=True); parser.add_argument("--canonical-index", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--mode", choices=("small", "full"), required=True)
    parser.add_argument("--init-checkpoint"); parser.add_argument("--small-pairs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4); parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--rank", type=int, default=64); parser.add_argument("--max-gate", type=float, default=1.0)
    parser.add_argument("--gate-init", type=float, default=0.02); parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--switch-weight", type=float, default=0.5); parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4); parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    seed_everything(args.seed); device = resolve_device(args.device); dtype = parse_dtype(args.dtype, device)
    cache = PairCache(args.canonical_index, capacity=3)
    expected_pairs = 512
    if len(cache) != expected_pairs:
        raise ValueError(f"Reader training requires 512 canonical train pairs, found {len(cache)}")
    model, tokenizer = load_receiver(args.receiver_model, device, dtype)
    active_layers = full_attention_layers(model)
    reader = TokenCanonicalReader(
        model, canonical_dim=256, rank=args.rank, max_gate=args.max_gate,
        gate_init=args.gate_init, active_layers=active_layers,
    ).to(device)
    if args.init_checkpoint:
        previous = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if previous["receiver_name"] != args.receiver_name:
            raise ValueError("Reader initialization belongs to another receiver")
        reader.load_state_dict(previous["reader"])
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Receiver backbone is not fully frozen")
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in reader.parameters()}
    if optimized != expected:
        raise RuntimeError("Optimizer contains parameters other than the Reader")
    if args.mode == "small":
        train_indices = list(range(args.small_pairs)); validation_indices = train_indices
    else:
        train_indices = list(range(448)); validation_indices = list(range(448, 512))
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, global_step = [], float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        reader.train(); order = list(train_indices); random.Random(args.seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True); epoch_losses = []
        for position, index in enumerate(tqdm(order, desc=f"p2ir_{args.receiver_name}_{args.mode}_{epoch}")):
            pair = cache.load(index); prompt_row = pair["base"]
            pair_loss = torch.zeros((), device=device)
            row_log = {"epoch": epoch, "pair_index": index, "pair_id": prompt_row["pair_id"]}
            for variant in ("base", "counterfactual"):
                row = pair[variant]; alternative_row = pair["counterfactual" if variant == "base" else "base"]
                memory = canonical_to(row["memory"], device)
                target_nll = sequence_nll(model, tokenizer, reader, prompt_row, memory, row["answer"], args.max_length, device)
                alternative_nll = sequence_nll(model, tokenizer, reader, prompt_row, memory, alternative_row["answer"], args.max_length, device)
                switch = F.relu(args.margin + target_nll - alternative_nll)
                pair_loss = pair_loss + 0.5 * (target_nll + args.switch_weight * switch)
                row_log[f"{variant}_nll"] = float(target_nll.detach().cpu())
                row_log[f"{variant}_switch_margin"] = float((alternative_nll - target_nll).detach().cpu())
            if not torch.isfinite(pair_loss):
                raise RuntimeError(f"Non-finite Reader loss at pair {index}")
            (pair_loss / args.gradient_accumulation).backward()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Frozen Receiver received gradients")
            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            if should_step:
                bad = [name for name, parameter in reader.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
                if bad:
                    raise RuntimeError(f"Non-finite Reader gradients: {bad[:8]}")
                norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite Reader gradient norm")
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                row_log["gradient_norm"] = float(norm.detach().cpu())
            row_log["loss"] = float(pair_loss.detach().cpu()); epoch_losses.append(row_log["loss"])
            history.append(row_log); global_step += 1
        metrics = validate(model, tokenizer, reader, cache, validation_indices, args.max_length, device, args.margin)
        epoch_row = {"epoch": epoch, "mean_train_loss": float(np.mean(epoch_losses)), **{f"validation_{key}": value for key, value in metrics.items()}}
        write_json(output / "latest_epoch.json", epoch_row); write_jsonl(output / "train_history.jsonl", history)
        payload = {
            "format_version": 1, "receiver_name": args.receiver_name, "receiver_model": args.receiver_model,
            "reader": {name: value.detach().cpu() for name, value in reader.state_dict().items()},
            "reader_metadata": reader.metadata(), "canonical_index": str(Path(args.canonical_index).resolve()),
            "writer_checkpoint_sha256": cache.index["writer_checkpoint_sha256"],
            "writer_state_sha256": cache.index["writer_state_sha256"], "mode": args.mode,
            "epoch": epoch, "global_step": global_step, "args": vars(args),
        }
        torch.save(payload, output / "checkpoint_latest.pt")
        if metrics["loss"] < best_loss:
            best_loss, best_epoch = metrics["loss"], epoch
            torch.save(payload, output / "checkpoint_best.pt")
    final = validate(model, tokenizer, reader, cache, validation_indices, args.max_length, device, args.margin)
    write_json(output / "TRAIN_SUCCESS.json", {
        "status": "complete", "receiver": args.receiver_name, "mode": args.mode,
        "best_epoch": best_epoch, "epochs_completed": args.epochs, "validation": final,
        "writer_checkpoint_sha256": cache.index["writer_checkpoint_sha256"],
        "writer_state_sha256": cache.index["writer_state_sha256"], "active_layers": active_layers,
        "only_reader_trainable": True,
    })


if __name__ == "__main__":
    main()
