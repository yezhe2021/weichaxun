import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from p2id_common import (
    add_p2i_path,
    checkpoint_reader,
    condition_metrics,
    finite_gradients,
    paired_consistency,
    parse_dtype,
    resolve_device,
    seed_everything,
    write_json,
    write_jsonl,
)

add_p2i_path()
from canonical_modules import CanonicalExternalReader, full_attention_layers, zero_slots
from p2i_common import (
    LazyPairCache,
    extract_answer,
    generate,
    load_receiver,
    normalize_answer,
    sequence_nll,
    student_prefixed_prompt,
)


class PerSampleEvidenceSlots(nn.Module):
    def __init__(self, examples, slots, dim, std=0.02):
        super().__init__()
        self.keys = nn.Parameter(torch.empty(examples, slots, dim))
        self.values = nn.Parameter(torch.empty(examples, slots, dim))
        nn.init.normal_(self.keys, std=std)
        nn.init.normal_(self.values, std=std)

    def memory(self, index, dtype):
        return {"keys": self.keys[index].to(dtype), "values": self.values[index].to(dtype)}


def load_subset(cache, subset_path):
    with open(subset_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    pairs = []
    for entry in manifest["pairs"]:
        pair = cache.load(int(entry["pair_index"]))
        if pair["base"]["pair_id"] != entry["pair_id"]:
            raise ValueError("Subset pair ID no longer matches cache")
        pairs.append(pair)
    return pairs, manifest


def build_reader(model, checkpoint, receiver_name, device, init_mode):
    state, metadata = checkpoint_reader(checkpoint, receiver_name)
    if metadata["active_layers"] != full_attention_layers(model):
        raise ValueError("Reader active layers do not match receiver architecture")
    reader = CanonicalExternalReader(
        model,
        canonical_dim=metadata["canonical_dim"],
        adapter_rank=metadata["adapter_rank"],
        max_gate=metadata["max_gate"],
        gate_init=0.01,
        active_layers=metadata["active_layers"],
    ).to(device)
    if init_mode == "mother":
        reader.load_state_dict(state)
    return reader, metadata


def slot_index(pair_position, variant):
    return 2 * pair_position + (1 if variant == "counterfactual" else 0)


@torch.inference_mode()
def evaluate_free_running(model, tokenizer, reader, slots, pairs, labels, device, dtype, max_new_tokens):
    records = []

    def run(condition, owner, memory, source_answer, enabled):
        result = generate(
            model,
            tokenizer,
            reader,
            student_prefixed_prompt(tokenizer, owner),
            memory,
            max_new_tokens,
            device,
            enabled,
        )
        prediction, method = extract_answer(result["text"], labels)
        records.append(
            {
                "pair_id": owner["pair_id"],
                "variant": owner["variant"],
                "condition": condition,
                "original_target": owner["answer"],
                "source_memory_answer": source_answer,
                "prediction": prediction,
                "generated_text": result["text"],
                "generated_token_ids": result["token_ids"],
                "eos_reached": result["eos_reached"],
                "confidence": float(bool(prediction)),
                "extraction_method": method,
                "original_target_correct": float(normalize_answer(prediction) == normalize_answer(owner["answer"])),
                "source_memory_correct": float(bool(source_answer) and normalize_answer(prediction) == normalize_answer(source_answer)),
            }
        )

    for pair_position, pair in enumerate(tqdm(pairs, desc="free_slot_oracle_eval")):
        for variant in ("base", "counterfactual"):
            owner = pair[variant]
            opposite_variant = "counterfactual" if variant == "base" else "base"
            opposite = pair[opposite_variant]
            memory = slots.memory(slot_index(pair_position, variant), dtype)
            swapped = slots.memory(slot_index(pair_position, opposite_variant), dtype)
            run("correct", owner, memory, owner["answer"], True)
            run("base_cf_state_swap", owner, swapped, opposite["answer"], True)
            run("zero", owner, zero_slots(memory), "", True)
            run("reader_off", owner, memory, owner["answer"], False)
    return records


def main():
    parser = argparse.ArgumentParser(description="P2-I-D per-sample free Evidence-KV Reader capacity oracle")
    parser.add_argument("--receiver-name", choices=("qwen3_4b", "qwen3_5_4b"), required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--mother-checkpoint", required=True)
    parser.add_argument("--canonical-index", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reader-init", choices=("mother", "fresh"), default="mother")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--reader-lr", type=float, default=2e-3)
    parser.add_argument("--slot-lr", type=float, default=1e-2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    if args.receiver_name == "qwen3_5_4b" and args.dtype != "float32":
        raise ValueError("Qwen3.5 Reader oracle must use float32 to avoid known FP16 backward NaNs")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    checkpoint = torch.load(args.mother_checkpoint, map_location="cpu", weights_only=False)
    cache = LazyPairCache(args.canonical_index, capacity=2)
    if cache.index.get("writer_sha256") != checkpoint.get("writer_sha256"):
        raise ValueError("Canonical cache and mother Writer hashes differ")
    pairs, subset = load_subset(cache, args.subset)
    model, tokenizer = load_receiver(args.receiver_model, device, dtype)
    reader, metadata = build_reader(model, checkpoint, args.receiver_name, device, args.reader_init)
    slots = PerSampleEvidenceSlots(
        examples=2 * len(pairs), slots=checkpoint["interface"]["slots"],
        dim=checkpoint["interface"]["canonical_dim"],
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": reader.parameters(), "lr": args.reader_lr},
            {"params": slots.parameters(), "lr": args.slot_lr},
        ]
    )
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in reader.parameters()} | {id(parameter) for parameter in slots.parameters()}
    if optimized != expected or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Reader oracle optimizer/freeze audit failed")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    gradient_history = []
    best_loss = float("inf")
    bad_epochs = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        order = list(range(len(pairs)))
        np.random.default_rng(args.seed + epoch).shuffle(order)
        for position, pair_position in enumerate(order):
            pair = pairs[pair_position]
            pair_loss = torch.zeros((), device=device)
            for variant in ("base", "counterfactual"):
                row = pair[variant]
                other = pair["counterfactual" if variant == "base" else "base"]
                memory = slots.memory(slot_index(pair_position, variant), dtype)
                target_nll, _ = sequence_nll(
                    model, tokenizer, reader, row, memory, row["answer"], args.max_length, device
                )
                alternative_nll, _ = sequence_nll(
                    model, tokenizer, reader, row, memory, other["answer"], args.max_length, device
                )
                margin_loss = F.relu(args.margin + target_nll - alternative_nll)
                pair_loss = pair_loss + 0.5 * (target_nll + args.margin_weight * margin_loss)
            if not torch.isfinite(pair_loss):
                raise RuntimeError("Non-finite free-slot oracle loss")
            (pair_loss / args.gradient_accumulation).backward()
            epoch_losses.append(float(pair_loss.detach().cpu()))
            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            if should_step:
                named = [(f"reader.{name}", p) for name, p in reader.named_parameters()]
                named += [(f"slots.{name}", p) for name, p in slots.named_parameters()]
                bad = finite_gradients(named)
                if bad:
                    raise RuntimeError("Non-finite Reader oracle gradients: " + ", ".join(bad[:8]))
                grad_norm = torch.nn.utils.clip_grad_norm_(list(reader.parameters()) + list(slots.parameters()), 1.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("Non-finite Reader oracle gradient norm")
                def norm(parameter):
                    return float(parameter.grad.detach().float().norm().cpu()) if parameter.grad is not None else 0.0
                gradient_history.append(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "total_grad_norm": float(grad_norm.detach().cpu()),
                        "slot_key_grad_norm": norm(slots.keys),
                        "slot_value_grad_norm": norm(slots.values),
                        "shared_query_grad_norm": norm(reader.shared_query.weight),
                        "shared_output_grad_norm": norm(reader.shared_output.weight),
                        "gate_grad_norm": norm(reader.gate_logits),
                    }
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
        mean_loss = float(np.mean(epoch_losses))
        row = {
            "epoch": epoch,
            "mean_loss": mean_loss,
            "slot_key_norm": float(slots.keys.detach().float().norm(dim=-1).mean().cpu()),
            "slot_value_norm": float(slots.values.detach().float().norm(dim=-1).mean().cpu()),
            "gate_mean": float(reader.gates().detach().float().mean().cpu()),
            "gate_abs_mean": float(reader.gates().detach().float().abs().mean().cpu()),
        }
        history.append(row)
        payload = {
            "reader": {key: value.detach().cpu() for key, value in reader.state_dict().items()},
            "slots": {key: value.detach().cpu() for key, value in slots.state_dict().items()},
            "reader_metadata": reader.metadata(),
            "subset": subset,
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(payload, output / "checkpoint_latest.pt")
        if mean_loss < best_loss - 1e-4:
            best_loss = mean_loss
            bad_epochs = 0
            torch.save(payload, output / "checkpoint_best.pt")
        else:
            bad_epochs += 1
        write_jsonl(output / "train_history.jsonl", history)
        write_jsonl(output / "gradient_diagnostics.jsonl", gradient_history)
        if mean_loss < 0.01 or bad_epochs >= args.patience:
            break

    best = torch.load(output / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    reader.load_state_dict(best["reader"])
    slots.load_state_dict(best["slots"])
    reader.eval()
    slots.eval()
    labels = sorted({answer for entry in cache.entries for answer in (entry["base_answer"], entry["counterfactual_answer"])})
    records = evaluate_free_running(
        model, tokenizer, reader, slots, pairs, labels, device, dtype, args.max_new_tokens
    )
    conditions = condition_metrics(records)
    correct = next(row for row in conditions if row["condition"] == "correct")
    swap = next(row for row in conditions if row["condition"] == "base_cf_state_swap")
    paired = paired_consistency(records, "correct")
    base_records = [row for row in records if row["condition"] == "correct" and row["variant"] == "base"]
    cf_records = [row for row in records if row["condition"] == "correct" and row["variant"] == "counterfactual"]
    base_em = float(np.mean([row["original_target_correct"] for row in base_records]))
    cf_em = float(np.mean([row["original_target_correct"] for row in cf_records]))
    passed = base_em >= 0.95 and cf_em >= 0.95 and paired >= 0.90 and swap["source_memory_accuracy"] >= 0.90
    summary = {
        "status": "complete",
        "reader_oracle_passed": passed,
        "receiver": args.receiver_name,
        "reader_init": args.reader_init,
        "base_em": base_em,
        "counterfactual_em": cf_em,
        "paired_consistency": paired,
        "conditions": conditions,
        "best_epoch": best["epoch"],
        "best_teacher_forced_loss": best_loss,
        "thresholds": {"base_em": 0.95, "counterfactual_em": 0.95, "paired": 0.90, "swap_source": 0.90},
        "args": vars(args),
    }
    write_jsonl(output / "per_sample_predictions.jsonl", records)
    write_json(output / "SUCCESS.json", summary)


if __name__ == "__main__":
    main()
