import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from canonical_modules import (
    CanonicalEvidenceWriter,
    CanonicalExternalReader,
    full_attention_layers,
    mismatched_slots,
)
from p2i_common import (
    LazyPairCache,
    answer_swap_loss,
    canonical_to,
    load_receiver,
    native_to,
    negative_mapping,
    parse_dtype,
    resolve_device,
    seed_everything,
    sequence_nll,
    state_sha256,
    write_jsonl,
)


RECEIVER_NAMES = ("qwen3_4b", "qwen3_5_4b")


def build_reader(model, canonical_dim, adapter_rank, max_gate, gate_init):
    return CanonicalExternalReader(
        model,
        canonical_dim=canonical_dim,
        adapter_rank=adapter_rank,
        max_gate=max_gate,
        gate_init=gate_init,
        active_layers=full_attention_layers(model),
    )


def teacher_loss(teacher_row, diagnostics, target_nll, reference):
    if teacher_row is None:
        return reference.new_zeros(()), {"teacher_layers": 0}
    terms = []
    teacher_nll = teacher_row.get("answer_nll")
    if teacher_nll is not None:
        terms.append((target_nll - float(teacher_nll)) ** 2)
    teacher_layers = teacher_row.get("layer_deltas", {})
    compared = 0
    for layer, target in teacher_layers.items():
        current = diagnostics.get(str(layer), {}).get("delta_tensor")
        if current is None:
            continue
        target = target.to(device=current.device, dtype=torch.float32)
        cosine = 1.0 - F.cosine_similarity(current.float(), target, dim=0)
        norm = (current.float().norm() - target.norm()).abs() / target.norm().clamp_min(1e-6)
        terms.append(cosine + 0.1 * norm)
        compared += 1
    loss = torch.stack([term.float() for term in terms]).mean() if terms else reference.new_zeros(())
    return loss, {"teacher_layers": compared}


def load_optional_teacher(path):
    return LazyPairCache(path, capacity=2) if path else None


def checkpoint_payload(args, writer, reader, previous, epoch, global_step):
    readers = dict(previous.get("readers", {})) if previous else {}
    reader_metadata = dict(previous.get("reader_metadata", {})) if previous else {}
    readers[args.receiver_name] = {key: value.detach().cpu() for key, value in reader.state_dict().items()}
    reader_metadata[args.receiver_name] = reader.metadata()
    return {
        "format_version": 1,
        "interface": {"slots": args.slots, "canonical_dim": args.canonical_dim},
        "writer_geometry": {
            "sender_layers": args.sender_layers,
            "sender_heads": args.sender_heads,
            "sender_head_dim": args.sender_head_dim,
            "atom_dim": args.atom_dim,
        },
        "writer": {key: value.detach().cpu() for key, value in writer.state_dict().items()},
        "writer_sha256": state_sha256(writer.state_dict()),
        "readers": readers,
        "reader_metadata": reader_metadata,
        "last_receiver": args.receiver_name,
        "args": vars(args),
        "epoch": epoch,
        "global_step": global_step,
    }


def main():
    parser = argparse.ArgumentParser(description="Train one alternating branch of the P2-I canonical mother model")
    parser.add_argument("--receiver-name", choices=RECEIVER_NAMES, required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--sender-index")
    parser.add_argument("--canonical-index")
    parser.add_argument("--teacher-index")
    parser.add_argument("--resume")
    parser.add_argument("--out", required=True)
    parser.add_argument("--freeze-writer", action="store_true")
    parser.add_argument("--reinitialize-reader", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=448)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--slots", type=int, default=256)
    parser.add_argument("--canonical-dim", type=int, default=256)
    parser.add_argument("--atom-dim", type=int, default=64)
    parser.add_argument("--sender-layers", type=int, default=36)
    parser.add_argument("--sender-heads", type=int, default=8)
    parser.add_argument("--sender-head-dim", type=int, default=128)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--max-gate", type=float, default=0.5)
    parser.add_argument("--gate-init", type=float, default=0.01)
    parser.add_argument("--cf-weight", type=float, default=0.5)
    parser.add_argument("--cf-margin", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument("--negative-margin", type=float, default=0.5)
    parser.add_argument("--negative-every", type=int, default=4)
    parser.add_argument("--teacher-weight", type=float, default=0.02)
    parser.add_argument("--usage-weight", type=float, default=0.002)
    parser.add_argument("--diversity-weight", type=float, default=0.001)
    parser.add_argument("--writer-lr", type=float, default=2e-4)
    parser.add_argument("--reader-lr", type=float, default=5e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    if bool(args.sender_index) == bool(args.canonical_index):
        raise ValueError("Specify exactly one of --sender-index or --canonical-index")
    if args.freeze_writer and not args.resume:
        raise ValueError("Frozen-Writer training requires --resume")
    if args.canonical_index and not args.freeze_writer:
        raise ValueError("Canonical caches may only be used with --freeze-writer")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    source_cache = LazyPairCache(args.sender_index or args.canonical_index, capacity=2)
    teacher_cache = load_optional_teacher(args.teacher_index)
    pair_count = min(len(source_cache), args.max_pairs) if args.max_pairs > 0 else len(source_cache)
    if teacher_cache is not None:
        if len(teacher_cache) < pair_count:
            raise ValueError("Teacher cache is shorter than the requested training prefix")
        source_ids = [entry["pair_id"] for entry in source_cache.entries[:pair_count]]
        teacher_ids = [entry["pair_id"] for entry in teacher_cache.entries[:pair_count]]
        if source_ids != teacher_ids:
            raise ValueError("Teacher cache is not aligned with the source training prefix")
    negatives = (
        negative_mapping(source_cache, args.seed + 9000)
        if args.negative_every > 0
        else list(range(len(source_cache)))
    )

    receiver, tokenizer = load_receiver(args.receiver_model, device, dtype)
    reader = build_reader(
        receiver, args.canonical_dim, args.adapter_rank, args.max_gate, args.gate_init
    ).to(device)
    writer = CanonicalEvidenceWriter(
        args.sender_layers,
        args.sender_heads,
        args.sender_head_dim,
        args.slots,
        args.canonical_dim,
        args.atom_dim,
    ).to(device)

    previous = None
    initial_writer_hash = None
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        if previous.get("format_version") != 1:
            raise ValueError("Unsupported canonical checkpoint")
        if previous["interface"] != {"slots": args.slots, "canonical_dim": args.canonical_dim}:
            raise ValueError("Checkpoint interface mismatch")
        writer.load_state_dict(previous["writer"])
        if not args.reinitialize_reader and args.receiver_name in previous.get("readers", {}):
            reader.load_state_dict(previous["readers"][args.receiver_name])
        initial_writer_hash = state_sha256(writer.state_dict())

    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    if args.freeze_writer:
        for parameter in writer.parameters():
            parameter.requires_grad_(False)
        writer.eval()
    else:
        writer.train()
    reader.train()

    groups = [{"params": list(reader.parameters()), "lr": args.reader_lr}]
    if not args.freeze_writer:
        groups.append({"params": list(writer.parameters()), "lr": args.writer_lr})
    optimizer = torch.optim.AdamW(groups)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = {id(p) for p in reader.parameters()}
    if not args.freeze_writer:
        expected |= {id(p) for p in writer.parameters()}
    if optimized != expected:
        raise RuntimeError("Optimizer parameter audit failed")
    if any(parameter.requires_grad for parameter in receiver.parameters()):
        raise RuntimeError("Receiver backbone is not frozen")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = int(previous.get("global_step", 0)) if previous else 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        order = list(range(pair_count))
        random.Random(args.seed + epoch).shuffle(order)
        progress = tqdm(order, desc=f"canonical_{args.receiver_name}_epoch_{epoch}")
        for position, pair_index in enumerate(progress):
            pair = source_cache.load(pair_index)
            negative_pair = source_cache.load(negatives[pair_index])
            teacher_pair = teacher_cache.load(pair_index) if teacher_cache is not None else None
            total = torch.zeros((), device=device)
            row_log = {
                "epoch": epoch,
                "global_step": global_step,
                "receiver": args.receiver_name,
                "pair_id": pair["base"]["pair_id"],
                "writer_frozen": args.freeze_writer,
            }
            for variant in ("base", "counterfactual"):
                row = pair[variant]
                alternative = pair["counterfactual" if variant == "base" else "base"]["answer"]
                if args.canonical_index:
                    memory = canonical_to(row["memory"], device, dtype)
                    regularizers = {"usage": total.new_zeros(()), "diversity": total.new_zeros(())}
                else:
                    source = native_to(row["memory"], device, dtype)
                    memory = writer(source, output_dtype=dtype, return_diagnostics=True)
                    regularizers = writer.regularization(memory)
                diagnostics = {"_capture_tensors": bool(teacher_pair)}
                target_nll, _ = sequence_nll(
                    receiver, tokenizer, reader, row, memory, row["answer"],
                    args.max_length, device, diagnostics,
                )
                alternative_nll, _ = sequence_nll(
                    receiver, tokenizer, reader, row, memory, alternative,
                    args.max_length, device,
                )
                swap = answer_swap_loss(target_nll, alternative_nll, args.cf_margin)
                teacher_row = teacher_pair[variant].get("teacher") if teacher_pair else None
                distill, teacher_metrics = teacher_loss(teacher_row, diagnostics, target_nll, total)
                block = (
                    target_nll
                    + args.cf_weight * swap
                    + args.teacher_weight * distill
                    + args.usage_weight * regularizers["usage"]
                    + args.diversity_weight * regularizers["diversity"]
                )

                negative_rank = total.new_zeros(())
                if args.negative_every > 0 and (global_step + 1) % args.negative_every == 0:
                    negative_row = negative_pair[variant]
                    if args.canonical_index:
                        negative_memory = canonical_to(negative_row["memory"], device, dtype)
                    else:
                        negative_source = native_to(negative_row["memory"], device, dtype)
                        negative_memory = writer(negative_source, output_dtype=dtype)
                    negative_kind = "shuffled" if (pair_index + epoch) % 2 == 0 else "mismatched"
                    if negative_kind == "mismatched":
                        negative_memory = mismatched_slots(memory, negative_memory)
                    negative_nll, _ = sequence_nll(
                        receiver, tokenizer, reader, row, negative_memory, row["answer"],
                        args.max_length, device,
                    )
                    negative_rank = F.relu(args.negative_margin + target_nll - negative_nll)
                    block = block + args.negative_weight * negative_rank
                    row_log[f"{variant}_negative_kind"] = negative_kind
                total = total + 0.5 * block
                row_log.update(
                    {
                        f"{variant}_nll": float(target_nll.detach().cpu()),
                        f"{variant}_swap_margin": float((alternative_nll - target_nll).detach().cpu()),
                        f"{variant}_negative_loss": float(negative_rank.detach().cpu()),
                        f"{variant}_teacher_loss": float(distill.detach().cpu()),
                        f"{variant}_teacher_layers": teacher_metrics["teacher_layers"],
                        f"{variant}_slot_usage_loss": float(regularizers["usage"].detach().cpu()),
                        f"{variant}_slot_diversity_loss": float(regularizers["diversity"].detach().cpu()),
                    }
                )

            if not torch.isfinite(total.detach()):
                raise RuntimeError(f"Non-finite training loss at pair {pair_index}")
            (total / args.gradient_accumulation).backward()
            if any(parameter.grad is not None for parameter in receiver.parameters()):
                raise RuntimeError("Frozen receiver received gradients")
            if args.freeze_writer and any(parameter.grad is not None for parameter in writer.parameters()):
                raise RuntimeError("Frozen Writer received gradients")
            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            if should_step:
                trainable = [p for group in optimizer.param_groups for p in group["params"]]
                nonfinite = [
                    name for name, parameter in list(writer.named_parameters()) + list(reader.named_parameters())
                    if parameter.requires_grad and parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ]
                if nonfinite:
                    raise RuntimeError(
                        "Non-finite trainable gradients; refusing optimizer step: "
                        + ", ".join(nonfinite[:8])
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("Non-finite gradient norm; refusing optimizer step")
                row_log["grad_norm"] = float(grad_norm.detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            row_log["loss"] = float(total.detach().cpu())
            history.append(row_log)
            global_step += 1
            progress.set_postfix(loss=f"{row_log['loss']:.3f}")

        checkpoint = checkpoint_payload(args, writer, reader, previous, epoch, global_step)
        torch.save(checkpoint, output / f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, output / "checkpoint_latest.pt")
        previous = checkpoint
        write_jsonl(output / "train_history.jsonl", history)

    final_hash = state_sha256(writer.state_dict())
    if args.freeze_writer and initial_writer_hash != final_hash:
        raise RuntimeError("Writer hash changed during frozen-Writer training")
    success = {
        "status": "complete",
        "receiver": args.receiver_name,
        "writer_frozen": args.freeze_writer,
        "reader_reinitialized": args.reinitialize_reader,
        "pairs": pair_count,
        "epochs": args.epochs,
        "writer_sha256_before": initial_writer_hash,
        "writer_sha256_after": final_hash,
        "checkpoint": str(output / "checkpoint_latest.pt"),
        "args": vars(args),
    }
    with open(output / "TRAIN_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(success, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
