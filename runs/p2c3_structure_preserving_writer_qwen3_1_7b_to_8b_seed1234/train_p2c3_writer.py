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

from p2a_common import NativeKVExternalReader, memory_to, parse_dtype, resolve_device, write_jsonl
from p2c3_structure import detached_structure_metrics, structure_preservation_losses
from p2c3_writer import StructurePreservingNativeKVWriter
from train_p2c1_writer import (
    LazyPairCache,
    auxiliary_alignment,
    negative_mapping,
    sequence_nll,
    token_aligned,
    verify_cache_alignment,
)


VARIANTS = (
    "task_only",
    "shared_routing",
    "binding_relation",
    "shared_routing_relation",
)


def variant_features(variant):
    return {
        "shared_routing": variant in {"shared_routing", "shared_routing_relation"},
        "structure_loss": variant in {"binding_relation", "shared_routing_relation"},
    }


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


def _resample_token_axis(tensor, target_length):
    if tensor.shape[1] == target_length:
        return tensor
    if tensor.shape[1] == 0 or target_length == 0:
        raise ValueError("Cannot resample an empty KV sequence")
    indices = torch.linspace(
        0, tensor.shape[1] - 1, target_length, device=tensor.device
    ).round().long()
    return tensor.index_select(1, indices)


def value_mismatched_memory(positive, unrelated):
    values = [
        _resample_token_axis(value, key.shape[1])
        for key, value in zip(positive["keys"], unrelated["values"])
    ]
    output = {"keys": positive["keys"], "values": values}
    if "answer_token_mask" in positive:
        output["answer_token_mask"] = positive["answer_token_mask"]
    return output


def target_mass_alignment(writer_diag, teacher_diag, reference):
    terms = []
    for key, writer_slot in writer_diag.items():
        if not key.isdigit() or key not in teacher_diag:
            continue
        teacher_slot = teacher_diag[key]
        if "target_mass_tensor" in writer_slot and "target_mass_tensor" in teacher_slot:
            terms.append(
                (writer_slot["target_mass_tensor"] - teacher_slot["target_mass_tensor"].detach()).square()
            )
    if not terms:
        return reference.new_zeros((), dtype=torch.float32)
    return torch.stack(terms).mean()


def assert_frozen(receiver, reader):
    if any(parameter.requires_grad for parameter in receiver.parameters()):
        raise RuntimeError("Receiver backbone is not frozen")
    if any(parameter.requires_grad for parameter in reader.parameters()):
        raise RuntimeError("Query-only Reader is not frozen")
    if any(parameter.grad is not None for parameter in receiver.parameters()):
        raise RuntimeError("Frozen receiver received gradients")
    if any(parameter.grad is not None for parameter in reader.parameters()):
        raise RuntimeError("Frozen Reader received gradients")


def save_checkpoint(output, writer, args, sender_geometry, receiver_geometry, epoch):
    checkpoint = {
        "format_version": 3,
        "writer": writer.state_dict(),
        "args": vars(args),
        "features": variant_features(args.variant),
        "sender_geometry": sender_geometry,
        "receiver_geometry": receiver_geometry,
        "epoch": epoch,
    }
    torch.save(checkpoint, output / f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, output / "checkpoint_latest.pt")
    write_jsonl(output / f"routing_epoch_{epoch}.jsonl", writer.routing_diagnostics())


def main():
    parser = argparse.ArgumentParser(description="Train one P2-C3 structure-preserving Writer variant")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--sender-index", required=True)
    parser.add_argument("--teacher-index", required=True)
    parser.add_argument("--teacher-k-rms", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--route-residual-scale", type=float, default=0.25)
    parser.add_argument("--dense-warmup-fraction", type=float, default=0.1)
    parser.add_argument("--cf-weight", type=float, default=0.5)
    parser.add_argument("--cf-margin", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument("--negative-margin", type=float, default=0.5)
    parser.add_argument("--route-weight", type=float, default=0.005)
    parser.add_argument("--readout-weight", type=float, default=0.005)
    parser.add_argument("--target-mass-weight", type=float, default=0.002)
    parser.add_argument("--routing-binding-weight", type=float, default=0.01)
    parser.add_argument("--token-binding-weight", type=float, default=0.05)
    parser.add_argument("--token-relation-weight", type=float, default=0.025)
    parser.add_argument("--readout-relation-weight", type=float, default=0.025)
    parser.add_argument("--relation-max-tokens", type=int, default=64)
    parser.add_argument("--aux-every", type=int, default=8)
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
    reader = build_reader(receiver, args.reader_checkpoint, device)
    assert_frozen(receiver, reader)

    sender_geometry = {key: int(sender_cache.index[key]) for key in ("layers", "kv_heads", "head_dim")}
    receiver_geometry = {key: int(teacher_cache.index[key]) for key in ("layers", "kv_heads", "head_dim")}
    teacher_k_rms = torch.load(args.teacher_k_rms, map_location="cpu", weights_only=True)
    features = variant_features(args.variant)
    writer = StructurePreservingNativeKVWriter(
        sender_layers=sender_geometry["layers"],
        sender_heads=sender_geometry["kv_heads"],
        sender_head_dim=sender_geometry["head_dim"],
        receiver_layers=receiver_geometry["layers"],
        receiver_heads=receiver_geometry["kv_heads"],
        receiver_head_dim=receiver_geometry["head_dim"],
        top_k=args.top_k,
        adapter_rank=args.adapter_rank,
        shared_routing=features["shared_routing"],
        route_residual_scale=args.route_residual_scale,
        teacher_k_rms=teacher_k_rms,
    ).to(device).train()
    optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr)
    writer_ids = {id(parameter) for parameter in writer.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if writer_ids != optimizer_ids:
        raise RuntimeError("Optimizer must contain exactly the Writer parameters")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "RUN_CONFIG.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "features": features,
                "sender_geometry": sender_geometry,
                "receiver_geometry": receiver_geometry,
                "trainable_writer_parameters": sum(p.numel() for p in writer.parameters()),
                "receiver_trainable_parameters": sum(p.numel() for p in receiver.parameters() if p.requires_grad),
                "reader_trainable_parameters": sum(p.numel() for p in reader.parameters() if p.requires_grad),
            },
            handle,
            indent=2,
        )

    history = []
    global_step = 0
    total_steps = pair_count * args.epochs
    warmup_steps = int(math.ceil(args.dense_warmup_fraction * total_steps))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        order = list(range(pair_count))
        random.Random(args.seed + epoch).shuffle(order)
        negatives = negative_mapping(sender_cache, args.seed + 1000 + epoch)
        progress = tqdm(order, desc=f"p2c3_{args.variant}_epoch_{epoch}")
        for position, pair_index in enumerate(progress):
            writer.set_routing_mode(dense=global_step < warmup_steps)
            sender_pair = sender_cache.load(pair_index)
            teacher_pair = teacher_cache.load(pair_index)
            negative_pair = sender_cache.load(negatives[pair_index])
            negative_kind = "shuffled" if (pair_index + epoch) % 2 == 0 else "mismatched"
            row_metrics = {
                "global_step": global_step,
                "epoch": epoch,
                "pair_id": sender_pair["base"]["pair_id"],
                "variant": args.variant,
                "negative_kind": negative_kind,
                "routing_dense": writer.routing_dense,
            }

            for variant in ("base", "counterfactual"):
                row = sender_pair[variant]
                alternative = (
                    sender_pair["counterfactual"]["answer"]
                    if variant == "base"
                    else sender_pair["base"]["answer"]
                )
                source = memory_to(row["memory"], device, dtype)
                negative_source = memory_to(negative_pair[variant]["memory"], device, dtype)
                writer_memory = writer(source, output_dtype=dtype)
                use_auxiliary = global_step % args.aux_every == 0
                need_teacher = features["structure_loss"] or use_auxiliary
                nll_target, writer_diag = sequence_nll(
                    receiver,
                    tokenizer,
                    reader,
                    row,
                    writer_memory,
                    row["answer"],
                    args.max_length,
                    device,
                    need_teacher,
                )
                nll_alternative, _ = sequence_nll(
                    receiver,
                    tokenizer,
                    reader,
                    row,
                    writer_memory,
                    alternative,
                    args.max_length,
                    device,
                    False,
                )
                swap_margin = F.relu(args.cf_margin + nll_target - nll_alternative)
                block = nll_target + args.cf_weight * swap_margin

                route_loss = nll_target.new_zeros(())
                readout_loss = nll_target.new_zeros(())
                mass_loss = nll_target.new_zeros(())
                structure = {
                    "binding": nll_target.new_zeros(()),
                    "key_relation": nll_target.new_zeros(()),
                    "value_relation": nll_target.new_zeros(()),
                    "readout_relation": nll_target.new_zeros(()),
                    "layers": 0,
                    "sampled_tokens": 0,
                    "token_aligned": False,
                }
                if need_teacher:
                    teacher_memory = memory_to(teacher_pair[variant]["memory"], device, dtype)
                    with torch.no_grad():
                        _, teacher_diag = sequence_nll(
                            receiver,
                            tokenizer,
                            reader,
                            row,
                            teacher_memory,
                            row["answer"],
                            args.max_length,
                            device,
                            True,
                        )
                    aligned = token_aligned(row, teacher_pair[variant])
                    route_loss, readout_loss, _ = auxiliary_alignment(
                        writer_memory,
                        teacher_memory,
                        writer_diag,
                        teacher_diag,
                        aligned,
                    )
                    mass_loss = target_mass_alignment(writer_diag, teacher_diag, nll_target)
                    if use_auxiliary:
                        block = (
                            block
                            + args.route_weight * route_loss
                            + args.readout_weight * readout_loss
                            + args.target_mass_weight * mass_loss
                        )
                    if features["structure_loss"]:
                        structure = structure_preservation_losses(
                            writer_memory,
                            teacher_memory,
                            writer_diag,
                            teacher_diag,
                            aligned,
                            args.relation_max_tokens,
                        )
                        block = (
                            block
                            + args.token_binding_weight * structure["binding"]
                            + args.token_relation_weight
                            * (structure["key_relation"] + structure["value_relation"])
                            + args.readout_relation_weight * structure["readout_relation"]
                        )

                routing_difference = writer.routing_difference_tensors()
                if features["shared_routing"]:
                    block = block + args.routing_binding_weight * (
                        routing_difference["layer_l1"] + routing_difference["head_l1"]
                    )
                (block / (2 * args.gradient_accumulation)).backward()

                positive_for_rank = writer(source, output_dtype=dtype)
                negative_memory = writer(negative_source, output_dtype=dtype)
                if negative_kind == "mismatched":
                    negative_memory = value_mismatched_memory(positive_for_rank, negative_memory)
                nll_positive, _ = sequence_nll(
                    receiver,
                    tokenizer,
                    reader,
                    row,
                    positive_for_rank,
                    row["answer"],
                    args.max_length,
                    device,
                    False,
                )
                nll_negative, _ = sequence_nll(
                    receiver,
                    tokenizer,
                    reader,
                    row,
                    negative_memory,
                    row["answer"],
                    args.max_length,
                    device,
                    False,
                )
                negative_ranking = F.relu(
                    args.negative_margin + nll_positive - nll_negative
                )
                (
                    args.negative_weight
                    * negative_ranking
                    / (2 * args.gradient_accumulation)
                ).backward()

                row_metrics.update(
                    {
                        f"{variant}_nll": float(nll_target.detach().cpu()),
                        f"{variant}_answer_swap_margin": float(
                            (nll_alternative - nll_target).detach().cpu()
                        ),
                        f"{variant}_negative_margin": float(
                            (nll_negative - nll_positive).detach().cpu()
                        ),
                        f"{variant}_route_loss": float(route_loss.detach().cpu()),
                        f"{variant}_readout_loss": float(readout_loss.detach().cpu()),
                        f"{variant}_target_mass_loss": float(mass_loss.detach().cpu()),
                        **{
                            f"{variant}_structure_{key}": value
                            for key, value in detached_structure_metrics(structure).items()
                        },
                    }
                )

            assert_frozen(receiver, reader)
            should_step = (
                (position + 1) % args.gradient_accumulation == 0
                or position + 1 == len(order)
            )
            grad_norm = None
            if should_step:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0).detach().cpu()
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            differences = writer.routing_difference_tensors()
            row_metrics.update(
                {
                    "grad_norm": grad_norm,
                    "kv_layer_routing_l1": float(differences["layer_l1"].detach().cpu()),
                    "kv_head_routing_l1": float(differences["head_l1"].detach().cpu()),
                    "kv_layer_support_disagreement": float(
                        differences["layer_support_disagreement"].detach().cpu()
                    ),
                }
            )
            history.append(row_metrics)
            global_step += 1

        writer.set_routing_mode(dense=False)
        save_checkpoint(output, writer, args, sender_geometry, receiver_geometry, epoch)
        write_jsonl(output / "train_history.jsonl", history)

    with open(output / "TRAIN_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "variant": args.variant,
                "features": features,
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
