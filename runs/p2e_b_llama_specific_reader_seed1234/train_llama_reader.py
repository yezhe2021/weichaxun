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

from llama_specific_reader import LlamaSpecificExternalReader
from p2a_common import (
    memory_to,
    mismatched_memory,
    pack_prefixed_answer,
    parse_dtype,
    resolve_device,
    student_prefixed_prompt,
    write_jsonl,
)


VARIANTS = ("minimal_reader", "routed_reader")


class LazyPairCache:
    def __init__(self, index_path, capacity=3):
        with open(index_path, encoding="utf-8") as handle:
            self.index = json.load(handle)
        if self.index.get("format_version") not in (3, 4):
            raise ValueError("Expected a format_version=3 or 4 Native-KV pair cache")
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
        payload = torch.load(
            self.root / self.entries[index]["file"], map_location="cpu", weights_only=False
        )
        pair = {example["variant"]: example for example in payload["examples"]}
        self.loaded[index] = pair
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
        return pair


def compatible_negative(cache, left, right):
    if left == right:
        return False
    left_answers = {
        cache.entries[left]["base_answer"], cache.entries[left]["counterfactual_answer"]
    }
    right_answers = {
        cache.entries[right]["base_answer"], cache.entries[right]["counterfactual_answer"]
    }
    return left_answers.isdisjoint(right_answers)


def negative_mapping(cache, seed):
    rng = random.Random(seed)
    mapping = []
    for index in range(len(cache)):
        candidates = list(range(len(cache)))
        rng.shuffle(candidates)
        mapping.append(next(candidate for candidate in candidates if compatible_negative(cache, index, candidate)))
    return mapping


def sequence_nll(receiver, tokenizer, reader, row, memory, answer, max_length, device):
    prompt = student_prefixed_prompt(tokenizer, row)
    ids, mask, labels = pack_prefixed_answer(tokenizer, prompt, answer, max_length, device)
    with reader.inject(receiver, memory):
        output = receiver(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
    return output.loss.float()


def assert_only_reader_trainable(receiver, reader, optimizer=None):
    if any(parameter.requires_grad for parameter in receiver.parameters()):
        raise RuntimeError("Qwen receiver backbone is not fully frozen")
    if any(parameter.grad is not None for parameter in receiver.parameters()):
        raise RuntimeError("Frozen Qwen receiver received gradients")
    if not any(parameter.requires_grad for parameter in reader.parameters()):
        raise RuntimeError("Llama-specific Reader has no trainable parameters")
    if optimizer is not None:
        expected = {id(parameter) for parameter in reader.parameters() if parameter.requires_grad}
        actual = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if expected != actual:
            raise RuntimeError("Optimizer must contain all and only Reader parameters")


def save_checkpoint(output, reader, args, cache_geometry, epoch):
    checkpoint = {
        "format_version": 1,
        "experiment": "P2-E-B Llama-specific Reader",
        "reader": reader.state_dict(),
        "args": vars(args),
        "sender_geometry": cache_geometry,
        "epoch": epoch,
    }
    torch.save(checkpoint, output / f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, output / "checkpoint_latest.pt")
    write_jsonl(output / f"routing_epoch_{epoch}.jsonl", reader.routing_diagnostics())


def main():
    parser = argparse.ArgumentParser(description="Train a Reader that directly consumes raw Llama Native-KV")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--sender-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--query-rank", type=int, default=32)
    parser.add_argument("--output-rank", type=int, default=32)
    parser.add_argument("--max-gate", type=float, default=0.5)
    parser.add_argument("--gate-init", type=float, default=0.02)
    parser.add_argument("--cf-weight", type=float, default=0.5)
    parser.add_argument("--cf-margin", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument("--negative-margin", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-4)
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
    cache = LazyPairCache(args.sender_index)
    pair_count = min(len(cache), args.max_pairs) if args.max_pairs > 0 else len(cache)
    if pair_count < 2:
        raise ValueError("At least two pairs are required for wrong-memory ranking")

    tokenizer = AutoTokenizer.from_pretrained(
        args.receiver_model, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)

    geometry = {
        "layers": int(cache.index["layers"]),
        "kv_heads": int(cache.index["kv_heads"]),
        "head_dim": int(cache.index["head_dim"]),
        "query_heads": int(cache.index["query_heads"]),
    }
    reader = LlamaSpecificExternalReader(
        receiver,
        sender_layers=geometry["layers"],
        sender_kv_heads=geometry["kv_heads"],
        sender_head_dim=geometry["head_dim"],
        variant=args.variant,
        top_k=args.top_k,
        query_rank=args.query_rank,
        output_rank=args.output_rank,
        max_gate=args.max_gate,
        gate_init=args.gate_init,
    ).to(device).train()
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr)
    assert_only_reader_trainable(receiver, reader, optimizer)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "RUN_CONFIG.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "sender_geometry": geometry,
                "receiver_layers": len(receiver.model.layers),
                "receiver_trainable_parameters": sum(
                    parameter.numel() for parameter in receiver.parameters() if parameter.requires_grad
                ),
                "reader_trainable_parameters": sum(
                    parameter.numel() for parameter in reader.parameters() if parameter.requires_grad
                ),
            },
            handle,
            indent=2,
        )

    history = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        order = list(range(pair_count))
        random.Random(args.seed + epoch).shuffle(order)
        negatives = negative_mapping(cache, args.seed + 1000 + epoch)
        progress = tqdm(order, desc=f"llama_specific_{args.variant}_epoch_{epoch}")
        for position, pair_index in enumerate(progress):
            pair = cache.load(pair_index)
            negative_pair = cache.load(negatives[pair_index])
            negative_kind = "shuffled" if (pair_index + epoch) % 2 == 0 else "mismatched"
            metrics = {
                "global_step": global_step,
                "epoch": epoch,
                "pair_id": pair["base"]["pair_id"],
                "variant": args.variant,
                "negative_kind": negative_kind,
            }
            for memory_variant in ("base", "counterfactual"):
                row = pair[memory_variant]
                alternative = (
                    pair["counterfactual"]["answer"]
                    if memory_variant == "base"
                    else pair["base"]["answer"]
                )
                positive_memory = memory_to(row["memory"], device, dtype)
                unrelated_memory = memory_to(
                    negative_pair[memory_variant]["memory"], device, dtype
                )
                negative_memory = (
                    unrelated_memory
                    if negative_kind == "shuffled"
                    else mismatched_memory(positive_memory, unrelated_memory)
                )
                target_nll = sequence_nll(
                    receiver, tokenizer, reader, row, positive_memory,
                    row["answer"], args.max_length, device,
                )
                alternative_nll = sequence_nll(
                    receiver, tokenizer, reader, row, positive_memory,
                    alternative, args.max_length, device,
                )
                negative_nll = sequence_nll(
                    receiver, tokenizer, reader, row, negative_memory,
                    row["answer"], args.max_length, device,
                )
                swap_loss = F.relu(args.cf_margin + target_nll - alternative_nll)
                negative_loss = F.relu(
                    args.negative_margin + target_nll - negative_nll
                )
                loss = (
                    target_nll
                    + args.cf_weight * swap_loss
                    + args.negative_weight * negative_loss
                )
                (loss / (2 * args.gradient_accumulation)).backward()
                metrics.update(
                    {
                        f"{memory_variant}_target_nll": float(target_nll.detach().cpu()),
                        f"{memory_variant}_swap_margin": float(
                            (alternative_nll - target_nll).detach().cpu()
                        ),
                        f"{memory_variant}_negative_margin": float(
                            (negative_nll - target_nll).detach().cpu()
                        ),
                    }
                )

            assert_only_reader_trainable(receiver, reader)
            should_step = (
                (position + 1) % args.gradient_accumulation == 0
                or position + 1 == len(order)
            )
            grad_norm = None
            if should_step:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0).detach().cpu()
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            metrics["grad_norm"] = grad_norm
            metrics["mean_abs_gate"] = float(reader.gates().detach().abs().mean().cpu())
            history.append(metrics)
            global_step += 1

        save_checkpoint(output, reader, args, geometry, epoch)
        write_jsonl(output / "train_history.jsonl", history)

    with open(output / "TRAIN_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "variant": args.variant,
                "pairs": pair_count,
                "epochs": args.epochs,
                "steps": global_step,
                "checkpoint": str(output / "checkpoint_latest.pt"),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
