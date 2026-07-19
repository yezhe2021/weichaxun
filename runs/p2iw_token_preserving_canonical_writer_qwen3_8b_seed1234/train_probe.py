import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from p2iw_common import (
    PairCache, TokenCanonicalWriter, VariableAttentionProbe, deterministic_negative,
    label_vocabulary, paired_consistency, projection, resolve_device, seed_everything,
    write_json, write_jsonl,
)


def build_memory(row, source, projections, writer=None):
    if source == "teacher":
        state = projection(row["hidden"], projections["pca"]["hidden"], whiten=True)
        state = F.layer_norm(state, (state.shape[-1],))
        return {"keys": state.half(), "values": state.half()}
    bank = projections["random"] if source == "random" else projections["pca"]
    if source in {"random", "pca"}:
        whiten = source == "pca"
        keys = projection(row["key_flat"], bank["key"], whiten=whiten)
        values = projection(row["value_flat"], bank["value"], whiten=whiten)
        return {"keys": keys.half(), "values": values.half()}
    if source == "writer":
        with torch.inference_mode():
            output = writer(row["key_flat"], row["value_flat"])
        return {"keys": output["keys"].half(), "values": output["values"].half()}
    raise ValueError(source)


def materialize(cache, indices, labels, source, projections, writer=None):
    label_to_id = {value: index for index, value in enumerate(labels)}
    examples = []
    for pair_index in tqdm(indices, desc=f"materialize_{source}", leave=False):
        pair = cache.load(pair_index)
        for variant in ("base", "counterfactual"):
            row = pair[variant]
            memory = build_memory(row, source, projections, writer)
            examples.append({
                "pair_index": pair_index, "pair_id": row["pair_id"], "variant": variant,
                "answer": row["answer"], "label": label_to_id[row["answer"]], **memory,
                "answer_mask": row["answer_mask"].bool(),
            })
    return examples


def batches(examples, size, seed, shuffle=True):
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), size):
        yield [examples[index] for index in order[start:start + size]]


def pad(batch, device):
    length = max(row["keys"].shape[0] for row in batch)
    dim = batch[0]["keys"].shape[-1]
    keys = torch.zeros(len(batch), length, dim, device=device)
    values = torch.zeros_like(keys)
    mask = torch.zeros(len(batch), length, dtype=torch.bool, device=device)
    labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device=device)
    for index, row in enumerate(batch):
        count = row["keys"].shape[0]
        keys[index, :count] = row["keys"].to(device)
        values[index, :count] = row["values"].to(device)
        mask[index, :count] = True
    return keys, values, mask, labels


@torch.inference_mode()
def evaluate(probe, examples, batch_size, device):
    probe.eval()
    losses, correct = [], []
    for batch in batches(examples, batch_size, 0, False):
        keys, values, mask, labels = pad(batch, device)
        logits = probe(keys, values, mask)
        losses.extend(F.cross_entropy(logits, labels, reduction="none").cpu().tolist())
        correct.extend((logits.argmax(-1) == labels).float().cpu().tolist())
    return float(np.mean(losses)), float(np.mean(correct))


def resize_rows(value, target):
    if value.shape[0] == target:
        return value
    index = torch.linspace(0, value.shape[0] - 1, target).round().long()
    return value[index]


@torch.inference_mode()
def controls(probe, examples, cache, labels, device, seed):
    probe.eval()
    by_pair = {}
    for row in examples:
        by_pair.setdefault(row["pair_index"], {})[row["variant"]] = row
    records = []

    def append(condition, owner, keys, values, source_answer, source_mask=None, k_answer=None, v_answer=None):
        size = keys.shape[0]
        if values.shape[0] != size:
            values = resize_rows(values, size)
        mask = torch.ones(1, size, dtype=torch.bool, device=device)
        logits, attention = probe(keys[None].to(device), values[None].to(device), mask, True)
        probability = logits.softmax(-1)[0]
        prediction = labels[int(probability.argmax())]
        target_mass = 0.0
        if source_mask is not None:
            source_mask = resize_rows(source_mask.float()[:, None], size)[:, 0].bool().to(device)
            target_mass = float(attention[0, :, source_mask].sum(-1).mean().cpu()) if source_mask.any() else 0.0
        records.append({
            "pair_id": owner["pair_id"], "pair_index": owner["pair_index"],
            "variant": owner["variant"], "condition": condition,
            "original_target": owner["answer"], "source_memory_answer": source_answer,
            "k_source_answer": k_answer or source_answer, "v_source_answer": v_answer or source_answer,
            "prediction": prediction, "confidence": float(probability.max().cpu()),
            "original_target_correct": float(prediction == owner["answer"]),
            "source_memory_correct": float(prediction == source_answer) if source_answer else 0.0,
            "k_source_correct": float(prediction == (k_answer or source_answer)) if (k_answer or source_answer) else 0.0,
            "v_source_correct": float(prediction == (v_answer or source_answer)) if (v_answer or source_answer) else 0.0,
            "attention_entropy": float((-(attention * attention.clamp_min(1e-9).log()).sum(-1).mean() / np.log(max(2, size))).cpu()),
            "source_answer_attention_mass": target_mass,
        })

    generator = torch.Generator().manual_seed(seed + 909)
    for pair_index, pair in tqdm(by_pair.items(), desc="probe_controls", leave=False):
        other_index = deterministic_negative(cache.entries, pair_index)
        other = by_pair[other_index]
        for variant in ("base", "counterfactual"):
            current = pair[variant]
            opposite = pair["counterfactual" if variant == "base" else "base"]
            shuffled = other[variant]
            append("correct", current, current["keys"], current["values"], current["answer"], current["answer_mask"])
            append("base_cf_memory_swap", current, opposite["keys"], opposite["values"], opposite["answer"], opposite["answer_mask"])
            append("cross_sample_shuffled", current, shuffled["keys"], shuffled["values"], shuffled["answer"], shuffled["answer_mask"])
            append("k_current_v_other", current, current["keys"], shuffled["values"], "", shuffled["answer_mask"], current["answer"], shuffled["answer"])
            append("zero", current, torch.zeros_like(current["keys"]), torch.zeros_like(current["values"]), "")
            order = torch.randperm(current["keys"].shape[0], generator=generator)
            append("token_permutation", current, current["keys"][order], current["values"][order], current["answer"], current["answer_mask"][order])
    summary = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        summary.append({
            "condition": condition, "n": len(selected),
            "original_target_accuracy": float(np.mean([row["original_target_correct"] for row in selected])),
            "source_memory_accuracy": float(np.mean([row["source_memory_correct"] for row in selected])),
            "k_source_accuracy": float(np.mean([row["k_source_correct"] for row in selected])),
            "v_source_accuracy": float(np.mean([row["v_source_correct"] for row in selected])),
            "mean_confidence": float(np.mean([row["confidence"] for row in selected])),
            "mean_source_answer_attention_mass": float(np.mean([row["source_answer_attention_mass"] for row in selected])),
        })
    return records, {"conditions": summary, "correct_paired_consistency": paired_consistency(records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--projections", required=True)
    parser.add_argument("--source", choices=("random", "pca", "teacher", "writer"), required=True)
    parser.add_argument("--writer-checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_cache, test_cache = PairCache(args.train_index), PairCache(args.test_index)
    if len(train_cache) != 512 or len(test_cache) != 64:
        raise ValueError("Expected exact 512 train and 64 test pair caches")
    labels = label_vocabulary(train_cache, test_cache)
    projections = torch.load(args.projections, map_location="cpu", weights_only=False)
    writer = None
    if args.source == "writer":
        if not args.writer_checkpoint:
            raise ValueError("writer source requires --writer-checkpoint")
        checkpoint = torch.load(args.writer_checkpoint, map_location="cpu", weights_only=False)
        writer = TokenCanonicalWriter(projections["pca"], **checkpoint["writer_config"]).eval()
        writer.load_state_dict(checkpoint["writer"])
        for parameter in writer.parameters():
            parameter.requires_grad_(False)
    train = materialize(train_cache, range(448), labels, args.source, projections, writer)
    validation = materialize(train_cache, range(448, 512), labels, args.source, projections, writer)
    test = materialize(test_cache, range(64), labels, args.source, projections, writer)
    probe = VariableAttentionProbe(len(labels)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history, best_loss, bad = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        probe.train()
        losses = []
        for batch in batches(train, args.batch_size, args.seed + epoch, True):
            keys, values, mask, target = pad(batch, device)
            loss = F.cross_entropy(probe(keys, values, mask), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_loss, val_accuracy = evaluate(probe, validation, args.batch_size, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss, "validation_accuracy": val_accuracy}
        history.append(row)
        write_jsonl(output / "history.jsonl", history)
        if val_loss < best_loss - 1e-5:
            best_loss, bad = val_loss, 0
            torch.save({"probe": probe.state_dict(), "labels": labels, "source": args.source, "epoch": epoch}, output / "best_probe.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break
    checkpoint = torch.load(output / "best_probe.pt", map_location=device, weights_only=False)
    probe.load_state_dict(checkpoint["probe"])
    test_loss, test_accuracy = evaluate(probe, test, args.batch_size, device)
    _, base_accuracy = evaluate(probe, [row for row in test if row["variant"] == "base"], args.batch_size, device)
    _, counterfactual_accuracy = evaluate(probe, [row for row in test if row["variant"] == "counterfactual"], args.batch_size, device)
    records, control_summary = controls(probe, test, test_cache, labels, device, args.seed)
    write_jsonl(output / "per_sample_predictions.jsonl", records)
    condition_map = {row["condition"]: row for row in control_summary["conditions"]}
    result = {
        "status": "complete", "source": args.source, "best_epoch": checkpoint["epoch"],
        "test_loss": test_loss, "test_accuracy": test_accuracy,
        "base_accuracy": base_accuracy, "counterfactual_accuracy": counterfactual_accuracy,
        **control_summary,
        "fresh_probe_success": bool(
            base_accuracy >= 0.90 and counterfactual_accuracy >= 0.90
            and control_summary["correct_paired_consistency"] >= 0.85
            and condition_map["base_cf_memory_swap"]["source_memory_accuracy"] >= 0.90
            and condition_map["cross_sample_shuffled"]["source_memory_accuracy"] >= 0.90
            and condition_map["zero"]["original_target_accuracy"] <= 0.10
        ),
    }
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
