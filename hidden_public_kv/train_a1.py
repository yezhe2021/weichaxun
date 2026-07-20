import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .data import pack_answer, per_sequence_nll, write_jsonl
from .modeling import load_frozen_model, load_tokenizer, validate_architecture
from .train_utils import (
    HiddenCache, assert_only_modules_have_grad, build_reader, build_writer, cached_memory,
    different_answer_partner, make_optimizer, save_checkpoint,
)


def nll_for(model, reader, memory, packed):
    labels = packed["labels"]
    inputs = {name: value for name, value in packed.items() if name != "labels"}
    with reader.inject(model, memory):
        logits = model(**inputs, use_cache=False, return_dict=True).logits
        return per_sequence_nll(logits, labels).mean()


def backward_nll(model, reader, memory, packed, scale):
    labels = packed["labels"]
    inputs = {name: value for name, value in packed.items() if name != "labels"}
    with reader.inject(model, memory):
        logits = model(**inputs, use_cache=False, return_dict=True).logits
        nll = per_sequence_nll(logits, labels).mean()
        (scale * nll).backward()
    return nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adafactor", "adamw"), default="adafactor")
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cache = HiddenCache(args.cache)
    count = min(len(cache), args.limit) if args.limit else len(cache)
    rows = [cache.load(index)["row"] for index in range(count)]
    checkpoint = torch.load(args.a0_checkpoint, map_location="cpu", weights_only=False)
    tokenizer = load_tokenizer(args.receiver_model)
    model = load_frozen_model(args.receiver_model, "qwen3", args.device, torch.float16, gradient_checkpointing=True)
    validate_architecture(model, "qwen3")
    model.train()
    reader = build_reader(model, checkpoint["reader_metadata"]).to(device=args.device, dtype=torch.float16)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    writer = build_writer().to(device=args.device, dtype=torch.float16)
    writer.load_state_dict(checkpoint["writer"])
    writer.freeze_norms()
    optimizer = make_optimizer(writer.parameters(), args.optimizer, args.lr, args.weight_decay)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history, global_step = [], 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        order = list(range(count))
        random.Random(args.seed + epoch).shuffle(order)
        for position, index in enumerate(tqdm(order, desc=f"a1_epoch_{epoch}")):
            other_index = different_answer_partner(rows, index, args.seed + epoch)
            current, other = cache.load(index), cache.load(other_index)
            packed = pack_answer(tokenizer, current["row"], args.device, args.max_length)
            with torch.no_grad():
                correct_probe = nll_for(model, reader, cached_memory(writer, current["correct"], args.device), packed)
                wrong_probe = nll_for(model, reader, cached_memory(writer, other["correct"], args.device), packed)
                rank_active = bool((args.margin + correct_probe - wrong_probe).item() > 0)
            correct_weight = 1.0 + (args.rank_weight if rank_active else 0.0)
            correct_nll = backward_nll(
                model, reader, cached_memory(writer, current["correct"], args.device), packed,
                correct_weight / args.gradient_accumulation,
            )
            wrong_nll_value = float(wrong_probe.cpu())
            if rank_active and args.rank_weight:
                wrong_nll = backward_nll(
                    model, reader, cached_memory(writer, other["correct"], args.device), packed,
                    -args.rank_weight / args.gradient_accumulation,
                )
                wrong_nll_value = float(wrong_nll.detach().cpu())
            assert_only_modules_have_grad(model, writer)
            loss_value = float(correct_nll.detach().cpu()) + args.rank_weight * max(
                0.0, args.margin + float(correct_nll.detach().cpu()) - wrong_nll_value
            )
            history.append({
                "epoch": epoch, "index": index, "id": current["row"]["id"], "negative_id": other["row"]["id"],
                "correct_nll": float(correct_nll.detach().cpu()), "wrong_nll": wrong_nll_value,
                "rank_active": rank_active, "loss": loss_value,
            })
            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            if should_step:
                torch.nn.utils.clip_grad_norm_([p for p in writer.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if (position + 1) % 16 == 0:
                write_jsonl(output / "history.jsonl", history)
        save_checkpoint(output / "checkpoint_latest.pt", writer, reader, args, epoch, global_step)
        write_jsonl(output / "history.jsonl", history)


if __name__ == "__main__":
    main()
