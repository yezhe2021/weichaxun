import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiment import (
    VARIANTS,
    MemoryCache,
    adapter_parameters,
    audit_trainable_parameters,
    build_adapters,
    forward_answer,
    load_receiver,
    memory_to,
    read_json,
    seed_everything,
    write_json,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--max-answer-length", type=int, default=512)
    parser.add_argument("--gate-init", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = MemoryCache(args.memory)
    negatives_payload = read_json(args.negatives)
    negatives = negatives_payload["mapping"]
    if len(negatives) != len(cache):
        raise RuntimeError("Hard-negative mapping length differs from memory cache")
    total = min(args.max_samples or len(cache), len(cache))
    indices = list(range(total))
    model, tokenizer = load_receiver(args.receiver, device)
    memory_dim = int(cache.index["memory_dim"])
    reader, receiver_lora = build_adapters(model, memory_dim, args.variant, args.seed, args.gate_init)
    if reader is not None:
        reader.to(device)
    if receiver_lora is not None:
        receiver_lora.to(device)
    parameters = adapter_parameters(reader, receiver_lora)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    audit_trainable_parameters(model, reader, receiver_lora, optimizer)
    best_loss, best_epoch, history = float("inf"), -1, []
    reader_enabled = reader is not None
    lora_enabled = receiver_lora is not None
    for epoch in range(1, args.epochs + 1):
        order = indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        if reader is not None:
            reader.train()
        if receiver_lora is not None:
            receiver_lora.train()
        losses, correct_values, shuffled_values, active_margins = [], [], [], []
        for sample_index in tqdm(order, desc=f"train_{args.variant}_epoch{epoch}"):
            correct_payload = cache.load(sample_index)
            row = correct_payload["row"]
            correct_memory = memory_to(correct_payload, device) if reader_enabled else None
            optimizer.zero_grad(set_to_none=True)
            correct_nll = forward_answer(
                model,
                tokenizer,
                row,
                correct_memory,
                reader,
                receiver_lora,
                args.max_answer_length,
                device,
                reader_enabled=reader_enabled,
                lora_enabled=lora_enabled,
            )
            if reader_enabled:
                wrong_payload = cache.load(negatives[sample_index])
                wrong_memory = memory_to(wrong_payload, device)
                shuffled_nll = forward_answer(
                    model,
                    tokenizer,
                    row,
                    wrong_memory,
                    reader,
                    receiver_lora,
                    args.max_answer_length,
                    device,
                    reader_enabled=True,
                    lora_enabled=lora_enabled,
                )
                dependency = F.relu(args.margin + correct_nll - shuffled_nll)
                loss = correct_nll + args.margin_weight * dependency
            else:
                shuffled_nll = correct_nll.detach()
                dependency = correct_nll.new_tensor(args.margin)
                loss = correct_nll + args.margin_weight * dependency
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Gradient reached frozen Receiver backbone")
            losses.append(float(loss.detach()))
            correct_values.append(float(correct_nll.detach()))
            shuffled_values.append(float(shuffled_nll.detach()))
            active_margins.append(float(dependency.detach() > 0))
        epoch_loss = sum(losses) / len(losses)
        record = {
            "epoch": epoch,
            "steps": len(order),
            "train_loss": epoch_loss,
            "correct_answer_mean_nll": sum(correct_values) / len(correct_values),
            "shuffled_answer_mean_nll": sum(shuffled_values) / len(shuffled_values),
            "active_margin_rate": sum(active_margins) / len(active_margins),
            "gates": reader.gates().detach().cpu().tolist() if reader is not None else None,
        }
        history.append(record)
        write_jsonl(output / "training_history.jsonl", history)
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            checkpoint = {
                "variant": args.variant,
                "epoch": epoch,
                "train_loss": best_loss,
                "reader": copy.deepcopy({name: value.detach().cpu() for name, value in reader.state_dict().items()}) if reader is not None else None,
                "receiver_lora": copy.deepcopy({name: value.detach().cpu() for name, value in receiver_lora.state_dict().items()}) if receiver_lora is not None else None,
                "reader_metadata": reader.metadata() if reader is not None else None,
                "receiver_lora_metadata": receiver_lora.metadata() if receiver_lora is not None else None,
                "receiver": args.receiver,
                "memory_index": args.memory,
                "negative_mapping": args.negatives,
                "args": vars(args),
                "receiver_backbone_frozen": True,
            }
            torch.save(checkpoint, output / "checkpoint_best.pt")
    trainable_count = sum(parameter.numel() for parameter in parameters)
    write_json(output / "TRAIN_SUCCESS.json", {
        "status": "complete",
        "variant": args.variant,
        "samples": total,
        "epochs": args.epochs,
        "optimizer_steps": total * args.epochs,
        "best_epoch_by_training_objective": best_epoch,
        "best_train_loss": best_loss,
        "validation_used_for_selection": False,
        "trainable_parameters": trainable_count,
        "receiver_backbone_frozen": True,
        "optimizer_parameter_ids_exact": True,
        "loss": "answer-token mean NLL + 0.5 * max(0, 0.5 + correct NLL - shuffled NLL)",
        "checkpoint": str(output / "checkpoint_best.pt"),
    })


if __name__ == "__main__":
    main()
