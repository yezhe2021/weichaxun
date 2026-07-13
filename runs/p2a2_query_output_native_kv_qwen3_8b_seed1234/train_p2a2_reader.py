import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2a_common import (
    NativeKVExternalReader,
    memory_to,
    mismatched_memory,
    pack_prefixed_answer,
    parse_dtype,
    resolve_device,
    student_prefixed_prompt,
    write_jsonl,
)


class LazyPairCache:
    def __init__(self, index_path, capacity=2):
        with open(index_path, encoding="utf-8") as handle:
            self.index = json.load(handle)
        if self.index.get("format_version") != 3:
            raise ValueError("P2-A2 requires format_version=3 pair cache")
        self.root = Path(index_path).parent
        self.entries = self.index["pair_files"]
        self.capacity = capacity
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index in self.loaded:
            self.loaded.move_to_end(index)
            return self.loaded[index]
        payload = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
        pair = {example["variant"]: example for example in payload["examples"]}
        self.loaded[index] = pair
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
        return pair


def sequence_nll(receiver, tokenizer, adapter, row, memory, answer, max_length, device):
    prompt = student_prefixed_prompt(tokenizer, row)
    ids, mask, labels = pack_prefixed_answer(tokenizer, prompt, answer, max_length, device)
    with adapter.inject(receiver, memory):
        output = receiver(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
    return output.loss.float()


def compatible_negative(cache, index, candidate):
    if index == candidate:
        return False
    left = cache.entries[index]
    right = cache.entries[candidate]
    left_answers = {left["base_answer"], left["counterfactual_answer"]}
    right_answers = {right["base_answer"], right["counterfactual_answer"]}
    return left_answers.isdisjoint(right_answers)


def negative_mapping(cache, seed):
    rng = random.Random(seed)
    mapping = []
    for index in range(len(cache)):
        candidates = list(range(len(cache)))
        rng.shuffle(candidates)
        selected = next((candidate for candidate in candidates if compatible_negative(cache, index, candidate)), None)
        if selected is None:
            raise RuntimeError(f"No compatible negative pair for index {index}")
        mapping.append(selected)
    return mapping


def backward_loss(loss, gradient_accumulation):
    (loss / gradient_accumulation).backward()
    return float(loss.detach().cpu())


def main():
    parser = argparse.ArgumentParser(description="Train one P2-A2 Query/Output Reader configuration")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config-name", choices=("output_only", "query_only", "query_output"), required=True)
    parser.add_argument("--query-rank", type=int, required=True)
    parser.add_argument("--output-rank", type=int, required=True)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-gate", type=float, default=0.5)
    parser.add_argument("--gate-init", type=float, default=0.01)
    parser.add_argument("--cf-weight", type=float, default=0.5)
    parser.add_argument("--cf-margin", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument("--negative-margin", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    expected = {
        "output_only": (0, 32),
        "query_only": (32, 0),
        "query_output": (32, 32),
    }[args.config_name]
    if (args.query_rank, args.output_rank) != expected:
        raise ValueError(f"{args.config_name} requires query/output ranks {expected}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    cache = LazyPairCache(args.train_index)
    pair_count = min(len(cache), args.max_pairs) if args.max_pairs > 0 else len(cache)

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
        query_rank=args.query_rank,
        output_rank=args.output_rank,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        order = list(range(pair_count))
        random.Random(args.seed + epoch).shuffle(order)
        negatives = negative_mapping(cache, args.seed + 1000 + epoch)
        progress = tqdm(order, desc=f"p2a2_{args.config_name}_epoch_{epoch}")
        for position, pair_index in enumerate(progress):
            pair = cache.load(pair_index)
            negative_pair = cache.load(negatives[pair_index])
            base = pair["base"]
            counterfactual = pair["counterfactual"]
            negative_base = negative_pair["base"]
            negative_cf = negative_pair["counterfactual"]
            base_memory = memory_to(base["memory"], device, dtype)
            cf_memory = memory_to(counterfactual["memory"], device, dtype)
            negative_base_memory = memory_to(negative_base["memory"], device, dtype)
            negative_cf_memory = memory_to(negative_cf["memory"], device, dtype)

            nll_b_y = sequence_nll(
                receiver, tokenizer, adapter, base, base_memory, base["answer"], args.max_length, device
            )
            nll_b_cf = sequence_nll(
                receiver, tokenizer, adapter, base, base_memory, counterfactual["answer"], args.max_length, device
            )
            margin_b = F.relu(args.cf_margin + nll_b_y - nll_b_cf)
            base_block = 0.5 * nll_b_y + args.cf_weight * margin_b
            base_loss_value = backward_loss(base_block, args.gradient_accumulation)

            nll_cf_cf = sequence_nll(
                receiver, tokenizer, adapter, base, cf_memory, counterfactual["answer"], args.max_length, device
            )
            nll_cf_y = sequence_nll(
                receiver, tokenizer, adapter, base, cf_memory, base["answer"], args.max_length, device
            )
            margin_cf = F.relu(args.cf_margin + nll_cf_cf - nll_cf_y)
            cf_block = 0.5 * nll_cf_cf + args.cf_weight * margin_cf
            cf_loss_value = backward_loss(cf_block, args.gradient_accumulation)

            # Across two epochs every pair receives both negative-memory constraints.
            negative_kind = "shuffled" if (pair_index + epoch) % 2 == 0 else "mismatched"
            if negative_kind == "shuffled":
                base_negative = negative_base_memory
                cf_negative = negative_cf_memory
            else:
                base_negative = mismatched_memory(base_memory, negative_base_memory)
                cf_negative = mismatched_memory(cf_memory, negative_cf_memory)

            nll_b_y_rank = sequence_nll(
                receiver, tokenizer, adapter, base, base_memory, base["answer"], args.max_length, device
            )
            nll_b_negative = sequence_nll(
                receiver, tokenizer, adapter, base, base_negative, base["answer"], args.max_length, device
            )
            negative_margin_b = F.relu(args.negative_margin + nll_b_y_rank - nll_b_negative)
            negative_base_block = 0.5 * args.negative_weight * negative_margin_b
            negative_base_value = backward_loss(negative_base_block, args.gradient_accumulation)

            nll_cf_cf_rank = sequence_nll(
                receiver, tokenizer, adapter, base, cf_memory, counterfactual["answer"], args.max_length, device
            )
            nll_cf_negative = sequence_nll(
                receiver, tokenizer, adapter, base, cf_negative, counterfactual["answer"], args.max_length, device
            )
            negative_margin_cf = F.relu(args.negative_margin + nll_cf_cf_rank - nll_cf_negative)
            negative_cf_block = 0.5 * args.negative_weight * negative_margin_cf
            negative_cf_value = backward_loss(negative_cf_block, args.gradient_accumulation)

            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            grad_norm = None
            if should_step:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0).detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            row = {
                "epoch": epoch,
                "global_step": global_step,
                "pair_id": base["pair_id"],
                "negative_pair_id": negative_base["pair_id"],
                "negative_kind": negative_kind,
                "nll_b_y": float(nll_b_y.detach().cpu()),
                "nll_b_counterfactual": float(nll_b_cf.detach().cpu()),
                "nll_counterfactual_y_counterfactual": float(nll_cf_cf.detach().cpu()),
                "nll_counterfactual_y": float(nll_cf_y.detach().cpu()),
                "base_counterfactual_margin": float((nll_b_cf - nll_b_y).detach().cpu()),
                "counterfactual_margin": float((nll_cf_y - nll_cf_cf).detach().cpu()),
                "base_negative_probability_margin": float((nll_b_negative - nll_b_y_rank).detach().cpu()),
                "counterfactual_negative_probability_margin": float(
                    (nll_cf_negative - nll_cf_cf_rank).detach().cpu()
                ),
                "base_block_loss": base_loss_value,
                "counterfactual_block_loss": cf_loss_value,
                "negative_base_loss": negative_base_value,
                "negative_counterfactual_loss": negative_cf_value,
                "gate_abs_mean": float(adapter.gates().detach().float().abs().mean().cpu()),
                "grad_norm": grad_norm,
            }
            history.append(row)
            progress.set_postfix(
                cf=round(0.5 * (row["base_counterfactual_margin"] + row["counterfactual_margin"]), 2),
                neg=round(
                    0.5
                    * (row["base_negative_probability_margin"] + row["counterfactual_negative_probability_margin"]),
                    2,
                ),
            )
            global_step += 1

        checkpoint = {
            "format_version": 2,
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
                "pairs": pair_count,
                "steps": global_step,
                "checkpoint": str(output / "checkpoint_latest.pt"),
                "gates": adapter.gates().detach().float().cpu().tolist(),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
