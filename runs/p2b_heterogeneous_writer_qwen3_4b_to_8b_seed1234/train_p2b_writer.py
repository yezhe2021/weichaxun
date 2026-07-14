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
from p2b_writer import HeterogeneousNativeKVWriter


class LazyPairCache:
    def __init__(self, index_path, capacity=2):
        with open(index_path, encoding="utf-8") as handle:
            self.index = json.load(handle)
        if self.index.get("format_version") != 3:
            raise ValueError("P2-B requires format_version=3 pair caches")
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


def verify_cache_alignment(sender_cache, teacher_cache):
    if len(sender_cache) != len(teacher_cache):
        raise ValueError("Sender and teacher cache pair counts differ")
    for sender, teacher in zip(sender_cache.entries, teacher_cache.entries):
        if sender["pair_id"] != teacher["pair_id"]:
            raise ValueError(f"Pair mismatch: {sender['pair_id']} != {teacher['pair_id']}")


def compatible_negative(cache, index, candidate):
    if index == candidate:
        return False
    left = cache.entries[index]
    right = cache.entries[candidate]
    return {left["base_answer"], left["counterfactual_answer"]}.isdisjoint(
        {right["base_answer"], right["counterfactual_answer"]}
    )


def negative_mapping(cache, seed):
    rng = random.Random(seed)
    mapping = []
    for index in range(len(cache)):
        candidates = list(range(len(cache)))
        rng.shuffle(candidates)
        selected = next(candidate for candidate in candidates if compatible_negative(cache, index, candidate))
        mapping.append(selected)
    return mapping


def sequence_nll(receiver, tokenizer, reader, row, memory, answer, max_length, device, capture=False):
    prompt = student_prefixed_prompt(tokenizer, row)
    prompt_length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    ids, mask, labels = pack_prefixed_answer(tokenizer, prompt, answer, max_length, device)
    diagnostics = None
    if capture:
        diagnostics = {"_capture_training_tensors": True, "_capture_query_index": prompt_length - 1}
    with reader.inject(receiver, memory, diagnostics):
        output = receiver(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
    return output.loss.float(), diagnostics


def auxiliary_alignment(writer_memory, teacher_memory, writer_diag, teacher_diag, token_aligned):
    route_losses = []
    readout_losses = []
    for layer in range(len(writer_memory["keys"])):
        writer_slot = writer_diag[str(layer)]
        teacher_slot = teacher_diag[str(layer)]
        writer_route = writer_slot["route_tensor"]
        teacher_route = teacher_slot["route_tensor"].detach()
        if writer_route.shape == teacher_route.shape and token_aligned:
            route_losses.append(
                F.kl_div(writer_route.clamp_min(1e-8).log(), teacher_route, reduction="batchmean")
                / writer_route.shape[1]
            )
        else:
            route_losses.append(
                (writer_slot["route_entropy_tensor"] - teacher_slot["route_entropy_tensor"].detach()).square()
            )
            if "target_mass_tensor" in writer_slot and "target_mass_tensor" in teacher_slot:
                route_losses.append(
                    (writer_slot["target_mass_tensor"] - teacher_slot["target_mass_tensor"].detach()).square()
                )
        writer_readout = writer_slot["readout_tensor"].float()
        teacher_readout = teacher_slot["readout_tensor"].detach().float()
        cosine = 1.0 - F.cosine_similarity(writer_readout, teacher_readout, dim=-1).mean()
        norm_ratio = (
            writer_readout.norm(dim=-1).clamp_min(1e-6).log()
            - teacher_readout.norm(dim=-1).clamp_min(1e-6).log()
        ).square().mean()
        readout_losses.append(cosine + 0.1 * norm_ratio)

    route_loss = torch.stack(route_losses).mean()
    readout_loss = torch.stack(readout_losses).mean()
    kv_loss = writer_memory["keys"][0].new_zeros((), dtype=torch.float32)
    if token_aligned:
        kv_terms = []
        for writer_key, writer_value, teacher_key, teacher_value in zip(
            writer_memory["keys"], writer_memory["values"], teacher_memory["keys"], teacher_memory["values"]
        ):
            if writer_key.shape != teacher_key.shape or writer_value.shape != teacher_value.shape:
                kv_terms = []
                break
            kv_terms.append(
                1.0
                - F.cosine_similarity(writer_key.float(), teacher_key.detach().float(), dim=-1).mean()
            )
            kv_terms.append(
                1.0
                - F.cosine_similarity(writer_value.float(), teacher_value.detach().float(), dim=-1).mean()
            )
        if kv_terms:
            kv_loss = torch.stack(kv_terms).mean()
    return route_loss, readout_loss, kv_loss


def token_aligned(sender_example, teacher_example):
    return sender_example.get("evidence_token_ids") == teacher_example.get("evidence_token_ids")


def main():
    parser = argparse.ArgumentParser(description="Train a Qwen3-4B Writer for a frozen Qwen3-8B Query-only Reader")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--sender-index", required=True)
    parser.add_argument("--teacher-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--layer-width", type=int, default=3)
    parser.add_argument("--cf-weight", type=float, default=0.5)
    parser.add_argument("--cf-margin", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument("--negative-margin", type=float, default=0.5)
    parser.add_argument("--route-weight", type=float, default=0.1)
    parser.add_argument("--readout-weight", type=float, default=0.1)
    parser.add_argument("--kv-weight", type=float, default=0.01)
    parser.add_argument("--regularization-weight", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=2e-3)
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
    sender_cache = LazyPairCache(args.sender_index)
    teacher_cache = LazyPairCache(args.teacher_index)
    verify_cache_alignment(sender_cache, teacher_cache)
    pair_count = min(len(sender_cache), args.max_pairs) if args.max_pairs > 0 else len(sender_cache)

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

    reader_checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=False)
    reader_args = reader_checkpoint["args"]
    reader = NativeKVExternalReader(
        receiver,
        max_gate=float(reader_args["max_gate"]),
        gate_init=float(reader_args["gate_init"]),
        query_rank=int(reader_args["query_rank"]),
        output_rank=int(reader_args["output_rank"]),
    ).to(device).eval()
    reader.load_state_dict(reader_checkpoint["adapter"])
    for parameter in reader.parameters():
        parameter.requires_grad_(False)

    sender_geometry = sender_cache.index
    teacher_geometry = teacher_cache.index
    writer = HeterogeneousNativeKVWriter(
        sender_layers=int(sender_geometry["layers"]),
        sender_heads=int(sender_geometry["kv_heads"]),
        sender_head_dim=int(sender_geometry["head_dim"]),
        receiver_layers=int(teacher_geometry["layers"]),
        receiver_heads=int(teacher_geometry["kv_heads"]),
        receiver_head_dim=int(teacher_geometry["head_dim"]),
        layer_width=args.layer_width,
    ).to(device)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        order = list(range(pair_count))
        random.Random(args.seed + epoch).shuffle(order)
        negatives = negative_mapping(sender_cache, args.seed + 1000 + epoch)
        progress = tqdm(order, desc=f"p2b_writer_epoch_{epoch}")
        for position, pair_index in enumerate(progress):
            sender_pair = sender_cache.load(pair_index)
            teacher_pair = teacher_cache.load(pair_index)
            negative_sender_pair = sender_cache.load(negatives[pair_index])
            base = sender_pair["base"]
            counterfactual = sender_pair["counterfactual"]

            block_metrics = {}
            for variant, row, alternative_answer in (
                ("base", base, counterfactual["answer"]),
                ("counterfactual", counterfactual, base["answer"]),
            ):
                sender_memory = memory_to(sender_pair[variant]["memory"], device, dtype)
                teacher_memory = memory_to(teacher_pair[variant]["memory"], device, dtype)
                writer_memory = writer(sender_memory, output_dtype=dtype)
                nll_target, writer_diag = sequence_nll(
                    receiver, tokenizer, reader, row, writer_memory, row["answer"], args.max_length, device, True
                )
                nll_alternative, _ = sequence_nll(
                    receiver, tokenizer, reader, row, writer_memory, alternative_answer, args.max_length, device, False
                )
                with torch.no_grad():
                    _, teacher_diag = sequence_nll(
                        receiver, tokenizer, reader, row, teacher_memory, row["answer"], args.max_length, device, True
                    )
                route_loss, readout_loss, kv_loss = auxiliary_alignment(
                    writer_memory,
                    teacher_memory,
                    writer_diag,
                    teacher_diag,
                    token_aligned(sender_pair[variant], teacher_pair[variant]),
                )
                margin_loss = F.relu(args.cf_margin + nll_target - nll_alternative)
                block = (
                    0.5 * nll_target
                    + args.cf_weight * margin_loss
                    + args.route_weight * route_loss
                    + args.readout_weight * readout_loss
                    + args.kv_weight * kv_loss
                    + args.regularization_weight * writer.regularization()
                )
                (block / args.gradient_accumulation).backward()
                block_metrics[f"{variant}_nll_target"] = float(nll_target.detach().cpu())
                block_metrics[f"{variant}_answer_swap_margin"] = float(
                    (nll_alternative - nll_target).detach().cpu()
                )
                block_metrics[f"{variant}_route_loss"] = float(route_loss.detach().cpu())
                block_metrics[f"{variant}_readout_loss"] = float(readout_loss.detach().cpu())
                block_metrics[f"{variant}_kv_loss"] = float(kv_loss.detach().cpu())

            negative_kind = "shuffled" if (pair_index + epoch) % 2 == 0 else "mismatched"
            for variant, row in (("base", base), ("counterfactual", counterfactual)):
                positive_source = memory_to(sender_pair[variant]["memory"], device, dtype)
                negative_source = memory_to(negative_sender_pair[variant]["memory"], device, dtype)
                positive_memory = writer(positive_source, output_dtype=dtype)
                negative_memory = writer(negative_source, output_dtype=dtype)
                if negative_kind == "mismatched":
                    negative_memory = mismatched_memory(positive_memory, negative_memory)
                nll_positive, _ = sequence_nll(
                    receiver, tokenizer, reader, row, positive_memory, row["answer"], args.max_length, device, False
                )
                nll_negative, _ = sequence_nll(
                    receiver, tokenizer, reader, row, negative_memory, row["answer"], args.max_length, device, False
                )
                ranking = F.relu(args.negative_margin + nll_positive - nll_negative)
                (0.5 * args.negative_weight * ranking / args.gradient_accumulation).backward()
                block_metrics[f"{variant}_negative_probability_margin"] = float(
                    (nll_negative - nll_positive).detach().cpu()
                )

            if any(parameter.grad is not None for parameter in receiver.parameters()):
                raise RuntimeError("Frozen receiver received gradients")
            if any(parameter.grad is not None for parameter in reader.parameters()):
                raise RuntimeError("Frozen Reader received gradients")
            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            grad_norm = None
            if should_step:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0).detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            history.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "pair_id": base["pair_id"],
                    "negative_pair_id": negative_sender_pair["base"]["pair_id"],
                    "negative_kind": negative_kind,
                    "grad_norm": grad_norm,
                    **block_metrics,
                }
            )
            progress.set_postfix(
                base=round(block_metrics["base_answer_swap_margin"], 2),
                cf=round(block_metrics["counterfactual_answer_swap_margin"], 2),
            )
            global_step += 1

        checkpoint = {
            "format_version": 1,
            "writer": writer.state_dict(),
            "args": vars(args),
            "sender_geometry": {
                "layers": int(sender_geometry["layers"]),
                "kv_heads": int(sender_geometry["kv_heads"]),
                "head_dim": int(sender_geometry["head_dim"]),
            },
            "receiver_geometry": {
                "layers": int(teacher_geometry["layers"]),
                "kv_heads": int(teacher_geometry["kv_heads"]),
                "head_dim": int(teacher_geometry["head_dim"]),
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
                "sender_geometry": checkpoint["sender_geometry"],
                "receiver_geometry": checkpoint["receiver_geometry"],
                "checkpoint": str(output / "checkpoint_latest.pt"),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
