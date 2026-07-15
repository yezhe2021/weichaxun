import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2a_common import NativeKVExternalReader, memory_to, mismatched_memory, parse_dtype, resolve_device, write_jsonl
from p2c2_writer import EnhancedGlobalNativeKVWriter
from train_p2c1_writer import (
    LazyPairCache,
    auxiliary_alignment,
    negative_mapping,
    sequence_nll,
    token_aligned,
    verify_cache_alignment,
)


def mean_target_mass(diagnostics):
    masses = [
        values["target_mass_tensor"]
        for key, values in diagnostics.items()
        if key.isdigit() and "target_mass_tensor" in values
    ]
    if not masses:
        raise RuntimeError("No target attention mass was captured")
    return torch.stack(masses).mean()


def key_mismatched_memory(key_source, value_source):
    """Pair unrelated keys with the full positive value sequence.

    K-first route ranking needs the positive answer-token mask to remain valid.
    Resampling unrelated keys avoids dropping target positions when the two
    evidence sequences have different token lengths.
    """
    keys = []
    values = []
    for key, value in zip(key_source["keys"], value_source["values"]):
        target_length = value.shape[1]
        if key.shape[1] == 0 or target_length == 0:
            raise ValueError("Cannot construct key-mismatched memory from an empty sequence")
        if key.shape[1] != target_length:
            indices = torch.linspace(
                0,
                key.shape[1] - 1,
                target_length,
                device=key.device,
            ).round().long()
            key = key.index_select(1, indices)
        keys.append(key)
        values.append(value)
    output = {"keys": keys, "values": values}
    if "answer_token_mask" in value_source:
        answer_mask = value_source["answer_token_mask"]
        if answer_mask.numel() != values[0].shape[1]:
            raise ValueError("Positive answer-token mask does not match the value sequence")
        output["answer_token_mask"] = answer_mask
    return output


def build_reader(receiver, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    reader = NativeKVExternalReader(
        receiver,
        max_gate=float(args["max_gate"]),
        gate_init=float(args["gate_init"]),
        query_rank=int(args["query_rank"]),
        output_rank=int(args["output_rank"]),
    ).to(device).eval()
    reader.load_state_dict(checkpoint["adapter"])
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    return reader


def schedule_for(variant):
    if variant == "full_staged":
        return [("key", 1), ("value", 1), ("joint", 1)]
    return [("joint", 3)]


def stage_learning_rate(stage, base_lr, joint_lr, variant):
    if stage == "joint" and variant == "full_staged":
        return joint_lr
    return base_lr


def snapshot_inactive(writer):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in writer.named_parameters()
        if not parameter.requires_grad
    }


def assert_inactive_unchanged(writer, snapshot):
    current = dict(writer.named_parameters())
    for name, expected in snapshot.items():
        if not torch.equal(current[name].detach().cpu(), expected):
            raise RuntimeError(f"Frozen Writer parameter changed during stage: {name}")


def assert_frozen(receiver, reader):
    if any(parameter.grad is not None for parameter in receiver.parameters()):
        raise RuntimeError("Frozen receiver received gradients")
    if any(parameter.grad is not None for parameter in reader.parameters()):
        raise RuntimeError("Frozen Query-only Reader received gradients")


def save_checkpoint(output, writer, args, sender_geometry, receiver_geometry, stage, stage_epoch):
    checkpoint = {
        "format_version": 2,
        "writer": writer.state_dict(),
        "args": vars(args),
        "sender_geometry": sender_geometry,
        "receiver_geometry": receiver_geometry,
        "stage": stage,
        "stage_epoch": stage_epoch,
    }
    name = f"checkpoint_{stage}_epoch_{stage_epoch}.pt"
    torch.save(checkpoint, output / name)
    torch.save(checkpoint, output / "checkpoint_latest.pt")
    write_jsonl(output / f"routing_{stage}_epoch_{stage_epoch}.jsonl", writer.routing_diagnostics())
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train the enhanced P2-C2 global sparse Writer")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--sender-index", required=True)
    parser.add_argument("--teacher-index", required=True)
    parser.add_argument("--teacher-k-rms", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", choices=("global_only", "global_head", "full_staged"), default="full_staged")
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--dense-warmup-fraction", type=float, default=0.1)
    parser.add_argument("--cf-weight", type=float, default=0.5)
    parser.add_argument("--cf-margin", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument("--negative-margin", type=float, default=0.5)
    parser.add_argument("--route-weight", type=float, default=0.1)
    parser.add_argument("--target-mass-weight", type=float, default=0.1)
    parser.add_argument("--route-rank-weight", type=float, default=0.1)
    parser.add_argument("--readout-weight", type=float, default=0.1)
    parser.add_argument("--kv-weight", type=float, default=0.01)
    parser.add_argument("--base-lr", type=float, default=2e-3)
    parser.add_argument("--joint-lr", type=float, default=2e-4)
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

    tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    reader = build_reader(receiver, args.reader_checkpoint, device)

    sender_geometry = {
        key: int(sender_cache.index[key]) for key in ("layers", "kv_heads", "head_dim")
    }
    receiver_geometry = {
        key: int(teacher_cache.index[key]) for key in ("layers", "kv_heads", "head_dim")
    }
    teacher_k_rms = torch.load(args.teacher_k_rms, map_location="cpu", weights_only=True)
    adapter_mode = "shared_full" if args.variant == "global_only" else "per_head"
    writer = EnhancedGlobalNativeKVWriter(
        sender_layers=sender_geometry["layers"],
        sender_heads=sender_geometry["kv_heads"],
        sender_head_dim=sender_geometry["head_dim"],
        receiver_layers=receiver_geometry["layers"],
        receiver_heads=receiver_geometry["kv_heads"],
        receiver_head_dim=receiver_geometry["head_dim"],
        top_k=args.top_k,
        adapter_mode=adapter_mode,
        adapter_rank=args.adapter_rank,
        teacher_k_rms=teacher_k_rms,
    ).to(device)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    final_checkpoint = None
    for stage_index, (stage, stage_epochs) in enumerate(schedule_for(args.variant)):
        writer.set_trainable_part(stage)
        active = [parameter for parameter in writer.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            active,
            lr=stage_learning_rate(stage, args.base_lr, args.joint_lr, args.variant),
        )
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if optimizer_ids != {id(parameter) for parameter in active}:
            raise RuntimeError("Optimizer does not contain exactly the active Writer parameters")
        inactive_snapshot = snapshot_inactive(writer)
        optimizer.zero_grad(set_to_none=True)
        for stage_epoch in range(1, stage_epochs + 1):
            order = list(range(pair_count))
            random.Random(args.seed + 100 * stage_index + stage_epoch).shuffle(order)
            negatives = negative_mapping(sender_cache, args.seed + 1000 + 100 * stage_index + stage_epoch)
            warmup_steps = int(math.ceil(args.dense_warmup_fraction * len(order)))
            progress = tqdm(order, desc=f"p2c2_{args.variant}_{stage}_{stage_epoch}")
            for position, pair_index in enumerate(progress):
                writer.set_routing_mode(
                    dense=position < warmup_steps,
                    noise=position >= warmup_steps,
                    part=stage,
                )
                sender_pair = sender_cache.load(pair_index)
                teacher_pair = teacher_cache.load(pair_index)
                negative_pair = sender_cache.load(negatives[pair_index])
                metrics = {"stage": stage}

                if stage == "key":
                    for variant in ("base", "counterfactual"):
                        row = sender_pair[variant]
                        source = memory_to(row["memory"], device, dtype)
                        teacher_memory = memory_to(teacher_pair[variant]["memory"], device, dtype)
                        negative_source = memory_to(negative_pair[variant]["memory"], device, dtype)
                        writer_memory = writer(source, output_dtype=dtype)
                        negative_memory = writer(negative_source, output_dtype=dtype)
                        # K-first uses wrong K with correct V so the ranking signal is actually address-dependent.
                        negative_memory = key_mismatched_memory(negative_memory, writer_memory)
                        nll, writer_diag = sequence_nll(
                            receiver, tokenizer, reader, row, writer_memory, row["answer"],
                            args.max_length, device, True,
                        )
                        _, negative_diag = sequence_nll(
                            receiver, tokenizer, reader, row, negative_memory, row["answer"],
                            args.max_length, device, True,
                        )
                        with torch.no_grad():
                            _, teacher_diag = sequence_nll(
                                receiver, tokenizer, reader, row, teacher_memory, row["answer"],
                                args.max_length, device, True,
                            )
                        route_loss, _, _ = auxiliary_alignment(
                            writer_memory, teacher_memory, writer_diag, teacher_diag,
                            token_aligned(row, teacher_pair[variant]),
                        )
                        correct_mass = mean_target_mass(writer_diag)
                        teacher_mass = mean_target_mass(teacher_diag).detach()
                        negative_mass = mean_target_mass(negative_diag)
                        mass_loss = (correct_mass - teacher_mass).square() - 0.1 * correct_mass.clamp_min(1e-8).log()
                        route_ranking = F.relu(0.01 - correct_mass + negative_mass)
                        block = (
                            args.route_weight * route_loss
                            + args.target_mass_weight * mass_loss
                            + args.route_rank_weight * route_ranking
                            + 0.05 * nll
                        )
                        (block / args.gradient_accumulation).backward()
                        metrics[f"{variant}_route_loss"] = float(route_loss.detach().cpu())
                        metrics[f"{variant}_target_mass"] = float(correct_mass.detach().cpu())

                elif stage == "value":
                    for variant in ("base", "counterfactual"):
                        row = sender_pair[variant]
                        source = memory_to(row["memory"], device, dtype)
                        teacher_memory = memory_to(teacher_pair[variant]["memory"], device, dtype)
                        writer_memory = writer(source, output_dtype=dtype)
                        nll, writer_diag = sequence_nll(
                            receiver, tokenizer, reader, row, writer_memory, row["answer"],
                            args.max_length, device, True,
                        )
                        with torch.no_grad():
                            _, teacher_diag = sequence_nll(
                                receiver, tokenizer, reader, row, teacher_memory, row["answer"],
                                args.max_length, device, True,
                            )
                        _, readout_loss, _ = auxiliary_alignment(
                            writer_memory, teacher_memory, writer_diag, teacher_diag,
                            token_aligned(row, teacher_pair[variant]),
                        )
                        block = 0.5 * nll + args.readout_weight * readout_loss
                        (block / args.gradient_accumulation).backward()
                        metrics[f"{variant}_nll"] = float(nll.detach().cpu())
                        metrics[f"{variant}_readout_loss"] = float(readout_loss.detach().cpu())

                else:
                    base = sender_pair["base"]
                    counterfactual = sender_pair["counterfactual"]
                    for variant, row, alternative in (
                        ("base", base, counterfactual["answer"]),
                        ("counterfactual", counterfactual, base["answer"]),
                    ):
                        source = memory_to(row["memory"], device, dtype)
                        teacher_memory = memory_to(teacher_pair[variant]["memory"], device, dtype)
                        writer_memory = writer(source, output_dtype=dtype)
                        nll_target, writer_diag = sequence_nll(
                            receiver, tokenizer, reader, row, writer_memory, row["answer"],
                            args.max_length, device, True,
                        )
                        nll_alternative, _ = sequence_nll(
                            receiver, tokenizer, reader, row, writer_memory, alternative,
                            args.max_length, device, False,
                        )
                        with torch.no_grad():
                            _, teacher_diag = sequence_nll(
                                receiver, tokenizer, reader, row, teacher_memory, row["answer"],
                                args.max_length, device, True,
                            )
                        route_loss, readout_loss, kv_loss = auxiliary_alignment(
                            writer_memory, teacher_memory, writer_diag, teacher_diag,
                            token_aligned(row, teacher_pair[variant]),
                        )
                        margin = F.relu(args.cf_margin + nll_target - nll_alternative)
                        block = (
                            0.5 * nll_target
                            + args.cf_weight * margin
                            + 0.25 * args.route_weight * route_loss
                            + 0.25 * args.readout_weight * readout_loss
                            + args.kv_weight * kv_loss
                        )
                        (block / args.gradient_accumulation).backward()
                        metrics[f"{variant}_nll"] = float(nll_target.detach().cpu())
                        metrics[f"{variant}_answer_swap_margin"] = float(
                            (nll_alternative - nll_target).detach().cpu()
                        )
                    negative_kind = "shuffled" if (pair_index + stage_epoch) % 2 == 0 else "mismatched"
                    for variant, row in (("base", base), ("counterfactual", counterfactual)):
                        positive = writer(memory_to(row["memory"], device, dtype), output_dtype=dtype)
                        negative = writer(
                            memory_to(negative_pair[variant]["memory"], device, dtype), output_dtype=dtype
                        )
                        if negative_kind == "mismatched":
                            negative = mismatched_memory(positive, negative)
                        nll_positive, _ = sequence_nll(
                            receiver, tokenizer, reader, row, positive, row["answer"],
                            args.max_length, device, False,
                        )
                        nll_negative, _ = sequence_nll(
                            receiver, tokenizer, reader, row, negative, row["answer"],
                            args.max_length, device, False,
                        )
                        ranking = F.relu(args.negative_margin + nll_positive - nll_negative)
                        (0.5 * args.negative_weight * ranking / args.gradient_accumulation).backward()
                        metrics[f"{variant}_negative_margin"] = float(
                            (nll_negative - nll_positive).detach().cpu()
                        )
                    metrics["negative_kind"] = negative_kind

                assert_frozen(receiver, reader)
                should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
                grad_norm = None
                if should_step:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(active, 1.0).detach().cpu())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                history.append(
                    {
                        "global_step": global_step,
                        "stage_index": stage_index,
                        "stage_epoch": stage_epoch,
                        "pair_id": sender_pair["base"]["pair_id"],
                        "grad_norm": grad_norm,
                        **metrics,
                    }
                )
                global_step += 1
            writer.set_routing_mode(dense=False, noise=False, part=stage)
            final_checkpoint = save_checkpoint(
                output, writer, args, sender_geometry, receiver_geometry, stage, stage_epoch
            )
            write_jsonl(output / "train_history.jsonl", history)
        assert_inactive_unchanged(writer, inactive_snapshot)

    with open(output / "TRAIN_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "variant": args.variant,
                "args": vars(args),
                "pairs": pair_count,
                "steps": global_step,
                "final_stage": final_checkpoint["stage"],
                "checkpoint": str(output / "checkpoint_latest.pt"),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
