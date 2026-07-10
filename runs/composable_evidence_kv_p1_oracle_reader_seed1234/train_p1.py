import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p1_common import (
    OracleCacheDataset,
    OracleEvidenceAdapter,
    parse_dtype,
    resolve_device,
    run_teacher_forced,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="Train the P1 oracle Evidence-KV reader")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reader-layers", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--shared-dim", type=int, default=256)
    parser.add_argument("--reader-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--contrast-weight", type=float, default=0.2)
    parser.add_argument("--contrast-margin", type=float, default=0.2)
    parser.add_argument("--contrast-every", type=int, default=4)
    parser.add_argument("--prompt-style", choices=("chat", "plain"), default="chat")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    dataset = OracleCacheDataset(args.train_cache)

    tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    total_layers = len(getattr(receiver, "model", receiver).layers)
    if any(layer < 0 or layer >= total_layers for layer in args.reader_layers):
        raise ValueError(f"reader layers {args.reader_layers} are invalid for {total_layers} receiver layers")

    adapter = OracleEvidenceAdapter(
        dataset.raw_dim,
        receiver.config.hidden_size,
        args.shared_dim,
        args.reader_heads,
        args.reader_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    history = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        order = list(range(len(dataset)))
        random.Random(args.seed + epoch).shuffle(order)
        progress = tqdm(order, desc=f"p1_epoch_{epoch}")
        for position, sample_index in enumerate(progress):
            example = dataset[sample_index]
            condition = random.choices(
                ["a_plus_b", "a_only", "b_only"],
                weights=[0.6, 0.2, 0.2],
                k=1,
            )[0]
            output_correct, diagnostics = run_teacher_forced(
                receiver,
                tokenizer,
                adapter,
                example,
                condition,
                device,
                args.max_length,
                args.prompt_style,
            )
            ce = output_correct.loss.float()
            contrast = torch.zeros((), device=device)
            mismatched_ce = None
            if args.contrast_weight > 0 and global_step % args.contrast_every == 0 and len(dataset) > 1:
                mismatch = dataset[(sample_index + 1) % len(dataset)]
                output_mismatch, _ = run_teacher_forced(
                    receiver,
                    tokenizer,
                    adapter,
                    example,
                    "mismatched_a_plus_b",
                    device,
                    args.max_length,
                    args.prompt_style,
                    mismatch_example=mismatch,
                )
                mismatched_ce = output_mismatch.loss.float()
                contrast = F.relu(args.contrast_margin + ce - mismatched_ce)
            loss = (ce + args.contrast_weight * contrast) / args.gradient_accumulation
            loss.backward()

            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            grad_norm = None
            if should_step:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0).detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            gate_values = [item["gate_mean"] for item in diagnostics.values()]
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "sample": sample_index,
                "id": example["id"],
                "condition": condition,
                "loss": float((ce + args.contrast_weight * contrast).detach().cpu()),
                "answer_ce": float(ce.detach().cpu()),
                "contrast_loss": float(contrast.detach().cpu()),
                "mismatched_ce": None if mismatched_ce is None else float(mismatched_ce.detach().cpu()),
                "gate_mean": float(np.mean(gate_values)) if gate_values else 0.0,
                "grad_norm": grad_norm,
            }
            history.append(row)
            progress.set_postfix(ce=round(row["answer_ce"], 3), gate=round(row["gate_mean"], 3))
            global_step += 1

        checkpoint = {
            "format_version": 1,
            "adapter": adapter.state_dict(),
            "args": vars(args),
            "raw_dim": dataset.raw_dim,
            "receiver_hidden_size": int(receiver.config.hidden_size),
            "epoch": epoch,
        }
        torch.save(checkpoint, output / f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(checkpoint, output / "checkpoint_latest.pt")
        write_jsonl(output / "train_history.jsonl", history)

    with open(output / "TRAIN_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "args": vars(args),
                "n": len(dataset),
                "steps": global_step,
                "checkpoint": str(output / "checkpoint_latest.pt"),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
