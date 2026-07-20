import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .core import capture_hidden_taps
from .data import load_jsonl, pack_answer, per_sequence_nll, write_jsonl
from .modeling import QWEN3_TAPS, load_frozen_model, load_tokenizer, validate_architecture
from .train_utils import assert_only_modules_have_grad, build_reader, build_writer, make_optimizer, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adafactor", "adamw"), default="adafactor")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rows = load_jsonl(args.data, args.limit)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model)
    model = load_frozen_model(args.model, "qwen3", args.device, torch.float16, gradient_checkpointing=True)
    validate_architecture(model, "qwen3")
    model.train()
    writer = build_writer().to(device=args.device, dtype=torch.float16)
    reader = build_reader(model).to(device=args.device, dtype=torch.float16)
    optimizer = make_optimizer(list(writer.parameters()) + list(reader.parameters()), args.optimizer, args.lr, args.weight_decay)
    history, global_step = [], 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(order)
        for position, index in enumerate(tqdm(order, desc=f"a0_epoch_{epoch}")):
            row = rows[index]
            evidence = tokenizer(row["evidence_text"], return_tensors="pt", add_special_tokens=True)
            evidence = {name: value.to(args.device) for name, value in evidence.items()}
            hidden, _ = capture_hidden_taps(model, evidence, QWEN3_TAPS, use_cache=False)
            memory = writer(hidden, evidence["attention_mask"])
            packed = pack_answer(tokenizer, row, args.device, args.max_length)
            labels = packed.pop("labels")
            with reader.inject(model, memory):
                logits = model(**packed, use_cache=False, return_dict=True).logits
                nll = per_sequence_nll(logits, labels).mean()
                (nll / args.gradient_accumulation).backward()
            assert_only_modules_have_grad(model, writer, reader)
            history.append({"epoch": epoch, "index": index, "id": row["id"], "nll": float(nll.detach().cpu())})
            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            if should_step:
                torch.nn.utils.clip_grad_norm_(list(writer.parameters()) + list(reader.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if (position + 1) % 16 == 0:
                write_jsonl(output / "history.jsonl", history)
        save_checkpoint(output / "checkpoint_latest.pt", writer, reader, args, epoch, global_step)
        write_jsonl(output / "history.jsonl", history)


if __name__ == "__main__":
    main()
