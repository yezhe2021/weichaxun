import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2a_common import (
    NativeKVExternalReader,
    iter_cache,
    memory_to,
    pack_answer,
    pack_prefixed_answer,
    parse_dtype,
    resolve_device,
    student_prompt,
    student_prefixed_prompt,
    write_jsonl,
)


def load_pairs(index_path, max_pairs):
    grouped = defaultdict(dict)
    for example in iter_cache(index_path):
        grouped[example["pair_id"]][example["variant"]] = example
    pairs = [pair for pair in grouped.values() if {"base", "counterfactual"}.issubset(pair)]
    return pairs[:max_pairs] if max_pairs > 0 else pairs


def sequence_nll(receiver, tokenizer, adapter, row, memory, answer, max_length, device, prefill_final=False):
    if prefill_final:
        prompt = student_prefixed_prompt(tokenizer, row)
        ids, mask, labels = pack_prefixed_answer(tokenizer, prompt, answer, max_length, device)
    else:
        prompt = student_prompt(tokenizer, row)
        ids, mask, labels = pack_answer(tokenizer, prompt, answer, max_length, device)
    with adapter.inject(receiver, memory):
        output = receiver(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
    return output.loss.float()


def main(default_reader_rank=0, default_gate_init=0.0, default_epochs=1, default_prefill_final=False):
    parser = argparse.ArgumentParser(description="Train the P2-A full native-KV external Reader upper bound")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-gate", type=float, default=0.5)
    parser.add_argument("--gate-init", type=float, default=default_gate_init)
    parser.add_argument("--reader-rank", type=int, default=default_reader_rank)
    parser.add_argument("--prefill-final", action="store_true", default=default_prefill_final)
    parser.add_argument("--cf-weight", type=float, default=0.5)
    parser.add_argument("--cf-margin", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    pairs = load_pairs(args.train_index, args.max_pairs)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    adapter = NativeKVExternalReader(
        receiver,
        max_gate=args.max_gate,
        gate_init=args.gate_init,
        reader_rank=args.reader_rank,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(pairs)
        progress = tqdm(pairs, desc=f"p2a_epoch_{epoch}")
        for position, pair in enumerate(progress):
            base = pair["base"]
            counterfactual = pair["counterfactual"]
            base_memory = memory_to(base["memory"], device, dtype)
            cf_memory = memory_to(counterfactual["memory"], device, dtype)

            nll_b_y = sequence_nll(
                receiver, tokenizer, adapter, base, base_memory, base["answer"], args.max_length, device, args.prefill_final
            )
            nll_b_cf = sequence_nll(
                receiver, tokenizer, adapter, base, base_memory, counterfactual["answer"], args.max_length, device, args.prefill_final
            )
            nll_cf_cf = sequence_nll(
                receiver, tokenizer, adapter, base, cf_memory, counterfactual["answer"], args.max_length, device, args.prefill_final
            )
            nll_cf_y = sequence_nll(
                receiver, tokenizer, adapter, base, cf_memory, base["answer"], args.max_length, device, args.prefill_final
            )
            generation_loss = 0.5 * (nll_b_y + nll_cf_cf)
            margin_b = F.relu(args.cf_margin + nll_b_y - nll_b_cf)
            margin_cf = F.relu(args.cf_margin + nll_cf_cf - nll_cf_y)
            counterfactual_loss = margin_b + margin_cf
            total = generation_loss + args.cf_weight * counterfactual_loss
            (total / args.gradient_accumulation).backward()

            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(pairs)
            grad_norm = None
            if should_step:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0).detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            gates = adapter.gates().detach().float().cpu()
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "pair_id": base["pair_id"],
                "nll_b_y": float(nll_b_y.detach().cpu()),
                "nll_b_counterfactual": float(nll_b_cf.detach().cpu()),
                "nll_counterfactual_y_counterfactual": float(nll_cf_cf.detach().cpu()),
                "nll_counterfactual_y": float(nll_cf_y.detach().cpu()),
                "base_margin": float((nll_b_cf - nll_b_y).detach().cpu()),
                "counterfactual_margin": float((nll_cf_y - nll_cf_cf).detach().cpu()),
                "generation_loss": float(generation_loss.detach().cpu()),
                "counterfactual_loss": float(counterfactual_loss.detach().cpu()),
                "gate_abs_mean": float(gates.abs().mean()),
                "gate_abs_max": float(gates.abs().max()),
                "grad_norm": grad_norm,
            }
            history.append(row)
            progress.set_postfix(
                gen=round(row["generation_loss"], 2),
                mb=round(row["base_margin"], 2),
                mcf=round(row["counterfactual_margin"], 2),
            )
            global_step += 1

        checkpoint = {
            "format_version": 1,
            "adapter": adapter.state_dict(),
            "args": vars(args),
            "model_config": {
                "layers": int(receiver.config.num_hidden_layers),
                "query_heads": int(receiver.config.num_attention_heads),
                "kv_heads": int(receiver.config.num_key_value_heads),
                "head_dim": int(receiver.config.head_dim),
            },
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
                "pairs": len(pairs),
                "steps": global_step,
                "checkpoint": str(output / "checkpoint_latest.pt"),
                "final_gates": adapter.gates().detach().float().cpu().tolist(),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
