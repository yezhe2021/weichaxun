import argparse
import json
from pathlib import Path

import numpy as np
import torch
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
    state_sha256,
    write_json,
    write_jsonl,
)

add_p2i_path()
from canonical_modules import CanonicalEvidenceWriter, CanonicalExternalReader, full_attention_layers, zero_slots
from p2i_common import (
    LazyPairCache,
    extract_answer,
    generate,
    load_receiver,
    native_to,
    normalize_answer,
    sequence_nll,
    student_prefixed_prompt,
)


def load_subset(cache, path):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    pairs = []
    for entry in manifest["pairs"]:
        pair = cache.load(int(entry["pair_index"]))
        if pair["base"]["pair_id"] != entry["pair_id"]:
            raise ValueError("Subset and Native-KV cache are no longer aligned")
        pairs.append(pair)
    return pairs, manifest


def build_modules(checkpoint, model, receiver_name, device):
    interface = checkpoint["interface"]
    geometry = checkpoint["writer_geometry"]
    writer = CanonicalEvidenceWriter(
        geometry["sender_layers"], geometry["sender_heads"], geometry["sender_head_dim"],
        interface["slots"], interface["canonical_dim"], geometry["atom_dim"],
    ).to(device)
    writer.load_state_dict(checkpoint["writer"])
    state, metadata = checkpoint_reader(checkpoint, receiver_name)
    if metadata["active_layers"] != full_attention_layers(model):
        raise ValueError("Reader active layers do not match receiver")
    reader = CanonicalExternalReader(
        model,
        canonical_dim=metadata["canonical_dim"],
        adapter_rank=metadata["adapter_rank"],
        max_gate=metadata["max_gate"],
        gate_init=0.01,
        active_layers=metadata["active_layers"],
    ).to(device)
    reader.load_state_dict(state)
    return writer, reader


@torch.inference_mode()
def evaluate(model, tokenizer, writer, reader, pairs, labels, device, dtype, max_new_tokens):
    records = []

    def run(condition, owner, memory, source_answer, enabled):
        result = generate(
            model, tokenizer, reader, student_prefixed_prompt(tokenizer, owner), memory,
            max_new_tokens, device, enabled,
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

    writer.eval()
    reader.eval()
    for pair in tqdm(pairs, desc="joint_overfit_eval"):
        memories = {
            variant: writer(native_to(pair[variant]["memory"], device, dtype), output_dtype=dtype)
            for variant in ("base", "counterfactual")
        }
        for variant in ("base", "counterfactual"):
            owner = pair[variant]
            opposite_variant = "counterfactual" if variant == "base" else "base"
            opposite = pair[opposite_variant]
            run("correct", owner, memories[variant], owner["answer"], True)
            run("base_cf_state_swap", owner, memories[opposite_variant], opposite["answer"], True)
            run("zero", owner, zero_slots(memories[variant]), "", True)
            run("reader_off", owner, memories[variant], owner["answer"], False)
    return records


def main():
    parser = argparse.ArgumentParser(description="P2-I-D real Writer+Reader small-subset joint overfit")
    parser.add_argument("--receiver-name", choices=("qwen3_4b", "qwen3_5_4b"), required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--mother-checkpoint", required=True)
    parser.add_argument("--native-index", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--reader-warmup-epochs", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--reader-lr", type=float, default=2e-3)
    parser.add_argument("--writer-lr", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    if args.receiver_name == "qwen3_5_4b" and args.dtype != "float32":
        raise ValueError("Qwen3.5 joint overfit must use float32")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    checkpoint = torch.load(args.mother_checkpoint, map_location="cpu", weights_only=False)
    cache = LazyPairCache(args.native_index, capacity=2)
    pairs, subset = load_subset(cache, args.subset)
    model, tokenizer = load_receiver(args.receiver_model, device, dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    writer, reader = build_modules(checkpoint, model, args.receiver_name, device)
    initial_writer_hash = state_sha256(writer.state_dict())
    optimizer = torch.optim.AdamW(
        [
            {"params": writer.parameters(), "lr": args.writer_lr},
            {"params": reader.parameters(), "lr": args.reader_lr},
        ]
    )
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = {id(p) for p in writer.parameters()} | {id(p) for p in reader.parameters()}
    if optimized != expected or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Joint overfit optimizer/freeze audit failed")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    gradient_history = []
    best_loss = float("inf")
    bad_epochs = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        writer_trainable = epoch > args.reader_warmup_epochs
        for parameter in writer.parameters():
            parameter.requires_grad_(writer_trainable)
        writer.train()
        reader.train()
        optimizer.zero_grad(set_to_none=True)
        order = list(range(len(pairs)))
        np.random.default_rng(args.seed + epoch).shuffle(order)
        epoch_losses = []
        for position, pair_position in enumerate(order):
            pair = pairs[pair_position]
            pair_loss = torch.zeros((), device=device)
            for variant in ("base", "counterfactual"):
                row = pair[variant]
                other = pair["counterfactual" if variant == "base" else "base"]
                memory = writer(native_to(row["memory"], device, dtype), output_dtype=dtype)
                target_nll, _ = sequence_nll(
                    model, tokenizer, reader, row, memory, row["answer"], args.max_length, device
                )
                alternative_nll, _ = sequence_nll(
                    model, tokenizer, reader, row, memory, other["answer"], args.max_length, device
                )
                margin = F.relu(args.margin + target_nll - alternative_nll)
                pair_loss = pair_loss + 0.5 * (target_nll + args.margin_weight * margin)
            if not torch.isfinite(pair_loss):
                raise RuntimeError("Non-finite joint overfit loss")
            (pair_loss / args.gradient_accumulation).backward()
            epoch_losses.append(float(pair_loss.detach().cpu()))
            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order)
            if should_step:
                named = [(f"writer.{name}", p) for name, p in writer.named_parameters()]
                named += [(f"reader.{name}", p) for name, p in reader.named_parameters()]
                bad = finite_gradients(named)
                if bad:
                    raise RuntimeError("Non-finite joint gradients: " + ", ".join(bad[:8]))
                trainable = [p for p in list(writer.parameters()) + list(reader.parameters()) if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("Non-finite joint gradient norm")
                def norm(parameter):
                    return float(parameter.grad.detach().float().norm().cpu()) if parameter.grad is not None else 0.0
                gradient_history.append(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "writer_trainable": writer_trainable,
                        "total_grad_norm": float(grad_norm.detach().cpu()),
                        "writer_slot_query_grad_norm": norm(writer.slot_queries),
                        "writer_key_output_grad_norm": norm(writer.key_output.weight),
                        "writer_value_output_grad_norm": norm(writer.value_output.weight),
                        "reader_query_grad_norm": norm(reader.shared_query.weight),
                        "reader_output_grad_norm": norm(reader.shared_output.weight),
                        "reader_gate_grad_norm": norm(reader.gate_logits),
                    }
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
        mean_loss = float(np.mean(epoch_losses))
        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "writer_trainable": writer_trainable,
                "gate_abs_mean": float(reader.gates().detach().float().abs().mean().cpu()),
            }
        )
        payload = {
            "writer": {key: value.detach().cpu() for key, value in writer.state_dict().items()},
            "reader": {key: value.detach().cpu() for key, value in reader.state_dict().items()},
            "writer_geometry": checkpoint["writer_geometry"],
            "interface": checkpoint["interface"],
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
    writer.load_state_dict(best["writer"])
    reader.load_state_dict(best["reader"])
    labels = sorted({answer for entry in cache.entries for answer in (entry["base_answer"], entry["counterfactual_answer"])})
    records = evaluate(model, tokenizer, writer, reader, pairs, labels, device, dtype, args.max_new_tokens)
    conditions = condition_metrics(records)
    paired = paired_consistency(records, "correct")
    base = [row for row in records if row["condition"] == "correct" and row["variant"] == "base"]
    cf = [row for row in records if row["condition"] == "correct" and row["variant"] == "counterfactual"]
    base_em = float(np.mean([row["original_target_correct"] for row in base]))
    cf_em = float(np.mean([row["original_target_correct"] for row in cf]))
    swap = next(row for row in conditions if row["condition"] == "base_cf_state_swap")
    passed = base_em >= 0.95 and cf_em >= 0.95 and paired >= 0.90 and swap["source_memory_accuracy"] >= 0.90
    summary = {
        "status": "complete",
        "joint_overfit_passed": passed,
        "receiver": args.receiver_name,
        "base_em": base_em,
        "counterfactual_em": cf_em,
        "paired_consistency": paired,
        "conditions": conditions,
        "best_epoch": best["epoch"],
        "best_teacher_forced_loss": best_loss,
        "initial_writer_sha256": initial_writer_hash,
        "final_writer_sha256": state_sha256(writer.state_dict()),
        "reader_warmup_epochs": args.reader_warmup_epochs,
        "args": vars(args),
    }
    write_jsonl(output / "per_sample_predictions.jsonl", records)
    write_json(output / "SUCCESS.json", summary)


if __name__ == "__main__":
    main()
