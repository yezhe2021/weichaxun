import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from p2id_common import (
    add_p2i_path,
    condition_metrics,
    deterministic_negative,
    paired_consistency,
    resolve_device,
    seed_everything,
    write_json,
    write_jsonl,
)

add_p2i_path()
from p2i_common import LazyPairCache


class AttentionCityProbe(nn.Module):
    def __init__(self, classes, slots=256, canonical_dim=256, queries=4):
        super().__init__()
        self.classes = int(classes)
        self.slots = int(slots)
        self.key_norm = nn.LayerNorm(canonical_dim)
        self.value_norm = nn.LayerNorm(canonical_dim)
        self.key_projection = nn.Linear(canonical_dim, 128, bias=False)
        self.value_projection = nn.Linear(canonical_dim, 256, bias=False)
        self.queries = nn.Parameter(torch.empty(queries, 128))
        self.classifier = nn.Sequential(
            nn.Linear(queries * 256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, classes),
        )
        nn.init.normal_(self.queries, std=1.0 / math.sqrt(128))

    def forward(self, keys, values, return_attention=False):
        key = self.key_projection(self.key_norm(keys.float()))
        value = self.value_projection(self.value_norm(values.float()))
        scores = torch.einsum("qd,bmd->bqm", self.queries, key) / math.sqrt(128)
        attention = scores.softmax(dim=-1)
        pooled = torch.einsum("bqm,bmv->bqv", attention, value).flatten(1)
        logits = self.classifier(pooled)
        return (logits, attention) if return_attention else logits


def label_vocabulary(*caches):
    values = sorted({
        answer for cache in caches for entry in cache.entries
        for answer in (entry["base_answer"], entry["counterfactual_answer"])
    })
    if len(values) != 40:
        raise ValueError(f"Expected exactly 40 city classes, found {len(values)}")
    return values


def load_examples(cache, pair_indices, label_to_id):
    examples = []
    for pair_index in tqdm(pair_indices, desc="load_probe_slots", leave=False):
        pair = cache.load(pair_index)
        for variant in ("base", "counterfactual"):
            row = pair[variant]
            examples.append(
                {
                    "pair_index": pair_index,
                    "pair_id": row["pair_id"],
                    "variant": variant,
                    "answer": row["answer"],
                    "label": label_to_id[row["answer"]],
                    "keys": row["memory"]["keys"].half().contiguous(),
                    "values": row["memory"]["values"].half().contiguous(),
                }
            )
    return examples


def iterate_batches(examples, batch_size, seed, shuffle):
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[index] for index in order[start : start + batch_size]]


def batch_tensors(batch, device):
    keys = torch.stack([row["keys"] for row in batch]).to(device)
    values = torch.stack([row["values"] for row in batch]).to(device)
    labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device=device)
    return keys, values, labels


@torch.inference_mode()
def evaluate_correct(probe, examples, batch_size, device):
    probe.eval()
    losses = []
    correct = []
    for batch in iterate_batches(examples, batch_size, 0, False):
        keys, values, labels = batch_tensors(batch, device)
        logits = probe(keys, values)
        losses.extend(F.cross_entropy(logits, labels, reduction="none").cpu().tolist())
        correct.extend((logits.argmax(-1) == labels).float().cpu().tolist())
    return float(np.mean(losses)), float(np.mean(correct))


def synthetic_positive_control(classes, device, seed):
    seed_everything(seed)
    probe = AttentionCityProbe(classes).to(device)
    codebook = F.normalize(torch.randn(classes, 256), dim=-1)

    def make(labels):
        labels = torch.tensor(labels, dtype=torch.long)
        batch = len(labels)
        keys = torch.zeros(batch, 256, 256)
        values = 0.01 * torch.randn(batch, 256, 256)
        keys[:, 0, 0] = 8.0
        values[:, 0] = codebook[labels] * 8.0
        values[:, 1:9] += codebook[labels, None] * 2.0
        return keys, values, labels

    train_labels = [index for index in range(classes) for _ in range(8)]
    valid_labels = [index for index in range(classes) for _ in range(2)]
    train = make(train_labels)
    valid = make(valid_labels)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-3)
    for epoch in range(12):
        order = torch.randperm(len(train_labels))
        for start in range(0, len(order), 32):
            index = order[start : start + 32]
            logits = probe(train[0][index].to(device), train[1][index].to(device))
            loss = F.cross_entropy(logits, train[2][index].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        logits = probe(valid[0].to(device), valid[1].to(device))
        accuracy = float((logits.argmax(-1).cpu() == valid[2]).float().mean())
    return {"accuracy": accuracy, "passed": accuracy >= 0.95}


@torch.inference_mode()
def control_evaluation(probe, test_cache, label_to_id, labels, device, batch_size):
    base_examples = load_examples(test_cache, range(len(test_cache)), label_to_id)
    by_pair = {}
    for row in base_examples:
        by_pair.setdefault(row["pair_index"], {})[row["variant"]] = row
    records = []

    def append(condition, owner, keys, values, source_answer):
        logits, attention = probe(keys[None].to(device), values[None].to(device), True)
        probability = logits.softmax(-1)[0]
        prediction_id = int(probability.argmax())
        prediction = labels[prediction_id]
        records.append(
            {
                "pair_id": owner["pair_id"],
                "pair_index": owner["pair_index"],
                "variant": owner["variant"],
                "condition": condition,
                "original_target": owner["answer"],
                "source_memory_answer": source_answer,
                "prediction": prediction,
                "confidence": float(probability[prediction_id].cpu()),
                "original_target_correct": float(prediction == owner["answer"]),
                "source_memory_correct": float(prediction == source_answer),
                "attention_entropy": float(
                    (-(attention * attention.clamp_min(1e-9).log()).sum(-1).mean()
                     / math.log(attention.shape[-1])).cpu()
                ),
            }
        )

    probe.eval()
    for pair_index, pair in tqdm(by_pair.items(), desc="probe_controls"):
        other_index = deterministic_negative(test_cache.entries, pair_index)
        other = by_pair[other_index]
        for variant in ("base", "counterfactual"):
            current = pair[variant]
            opposite = pair["counterfactual" if variant == "base" else "base"]
            shuffled = other[variant]
            append("correct", current, current["keys"], current["values"], current["answer"])
            append("base_cf_state_swap", current, opposite["keys"], opposite["values"], opposite["answer"])
            append("cross_sample_shuffled", current, shuffled["keys"], shuffled["values"], shuffled["answer"])
            append("zero", current, torch.zeros_like(current["keys"]), torch.zeros_like(current["values"]), "")
            append("k_current_v_other", current, current["keys"], shuffled["values"], shuffled["answer"])
    summary = {
        "conditions": condition_metrics(records),
        "correct_paired_consistency": paired_consistency(records, "correct"),
    }
    return records, summary


def main():
    parser = argparse.ArgumentParser(description="P2-I-D frozen Writer-slot attention probe")
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-pairs", type=int, default=448)
    parser.add_argument("--validation-pairs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    train_cache = LazyPairCache(args.train_index, capacity=4)
    test_cache = LazyPairCache(args.test_index, capacity=4)
    if len(train_cache) < args.train_pairs + args.validation_pairs:
        raise ValueError("Canonical train cache does not contain the 448/64 split")
    labels = label_vocabulary(train_cache, test_cache)
    label_to_id = {value: index for index, value in enumerate(labels)}
    synthetic = synthetic_positive_control(len(labels), device, args.seed + 77)
    write_json(output / "synthetic_positive_control.json", synthetic)
    if not synthetic["passed"]:
        raise RuntimeError("Synthetic probe positive control failed")

    train_examples = load_examples(train_cache, range(args.train_pairs), label_to_id)
    validation_examples = load_examples(
        train_cache,
        range(args.train_pairs, args.train_pairs + args.validation_pairs),
        label_to_id,
    )
    probe = AttentionCityProbe(len(labels)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    history = []
    best_loss = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        probe.train()
        train_losses = []
        for batch in iterate_batches(train_examples, args.batch_size, args.seed + epoch, True):
            keys, values, targets = batch_tensors(batch, device)
            logits = probe(keys, values)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation_loss, validation_accuracy = evaluate_correct(
            probe, validation_examples, args.batch_size, device
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        torch.save({"probe": probe.state_dict(), "labels": labels, "epoch": epoch, "args": vars(args)}, output / "checkpoint_latest.pt")
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            bad_epochs = 0
            torch.save({"probe": probe.state_dict(), "labels": labels, "epoch": epoch, "args": vars(args)}, output / "checkpoint_best.pt")
        else:
            bad_epochs += 1
        write_jsonl(output / "train_history.jsonl", history)
        if bad_epochs >= args.patience:
            break

    checkpoint = torch.load(output / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    probe.load_state_dict(checkpoint["probe"])
    validation_loss, validation_accuracy = evaluate_correct(
        probe, validation_examples, args.batch_size, device
    )
    records, controls = control_evaluation(
        probe, test_cache, label_to_id, labels, device, args.batch_size
    )
    correct_row = next(row for row in controls["conditions"] if row["condition"] == "correct")
    swap_row = next(row for row in controls["conditions"] if row["condition"] == "base_cf_state_swap")
    zero_row = next(row for row in controls["conditions"] if row["condition"] == "zero")
    passed = (
        validation_accuracy >= 0.80
        and controls["correct_paired_consistency"] >= 0.70
        and swap_row["source_memory_accuracy"] >= 0.70
        and zero_row["original_target_accuracy"] <= 0.10
    )
    summary = {
        "status": "complete",
        "writer_probe_passed": passed,
        "thresholds": {
            "validation_accuracy": 0.80,
            "test_paired_consistency": 0.70,
            "swap_source_memory_accuracy": 0.70,
            "zero_original_accuracy_max": 0.10,
        },
        "synthetic_positive_control": synthetic,
        "best_epoch": checkpoint["epoch"],
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "test_correct_accuracy": correct_row["original_target_accuracy"],
        **controls,
        "args": vars(args),
    }
    write_jsonl(output / "per_sample_predictions.jsonl", records)
    write_json(output / "SUCCESS.json", summary)


if __name__ == "__main__":
    main()
