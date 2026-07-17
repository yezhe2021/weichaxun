import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from prefill_probes import build_probe


class PairCache:
    def __init__(self, index_path):
        with open(index_path, encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.root = Path(index_path).parent
        self.entries = self.index["pair_files"]

    def load(self, index):
        payload = torch.load(
            self.root / self.entries[index]["file"], map_location="cpu", weights_only=False
        )
        return {example["variant"]: example for example in payload["examples"]}


class FeatureDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def probe_configs(index):
    layers = [int(layer) for layer in index["layers"]]
    final_layer = max(layers)
    configs = [
        {"name": f"end_linear_layer_{layer}", "kind": "end_linear", "layer": layer}
        for layer in layers
    ]
    for slots in (8, 16):
        if slots <= int(index["summary_slots"]):
            configs.extend(
                [
                    {
                        "name": f"summary_{slots}_linear_layer_{final_layer}",
                        "kind": "summary_linear",
                        "layer": final_layer,
                        "slots": slots,
                    },
                    {
                        "name": f"summary_{slots}_attention_layer_{final_layer}",
                        "kind": "summary_attention",
                        "layer": final_layer,
                        "slots": slots,
                    },
                ]
            )
    configs.append(
        {
            "name": f"raw_evidence_attention_layer_{index['raw_layer']}",
            "kind": "raw_evidence_attention",
            "layer": int(index["raw_layer"]),
        }
    )
    return configs


def select_feature(example, condition, config):
    states = example["states"][condition]
    layer = str(config["layer"])
    if config["kind"] == "end_linear":
        return states["end"][layer]
    if config["kind"] in {"summary_linear", "summary_attention"}:
        return states["summary"][layer][: config["slots"]]
    if config["kind"] == "raw_evidence_attention":
        return states["raw_evidence"][layer]
    raise ValueError(config["kind"])


def materialize(cache, config, condition, label_to_id, pair_indices=None):
    if condition not in cache.index["conditions"]:
        return []
    if config["kind"] == "raw_evidence_attention" and condition != "correct":
        return []
    pair_indices = range(len(cache.entries)) if pair_indices is None else pair_indices
    examples = []
    for pair_index in pair_indices:
        pair = cache.load(pair_index)
        for variant in ("base", "counterfactual"):
            row = pair[variant]
            if row["answer"] not in label_to_id:
                raise ValueError(f"Test answer {row['answer']!r} is absent from training vocabulary")
            examples.append(
                {
                    "pair_id": row["pair_id"],
                    "variant": variant,
                    "answer": row["answer"],
                    "counterpart_answer": row["counterpart_answer"],
                    "label": label_to_id[row["answer"]],
                    "feature": select_feature(row, condition, config),
                }
            )
    return examples


def collate(batch):
    features = [row["feature"].float() for row in batch]
    if features[0].ndim == 1:
        states = torch.stack(features)
        mask = None
    else:
        max_tokens = max(feature.shape[0] for feature in features)
        hidden = features[0].shape[-1]
        states = torch.zeros(len(features), max_tokens, hidden, dtype=torch.float32)
        mask = torch.zeros(len(features), max_tokens, dtype=torch.bool)
        for index, feature in enumerate(features):
            states[index, : feature.shape[0]] = feature
            mask[index, : feature.shape[0]] = True
    labels = torch.tensor([row["label"] for row in batch], dtype=torch.long)
    metadata = [{key: value for key, value in row.items() if key != "feature"} for row in batch]
    return states, mask, labels, metadata


def make_loader(examples, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        FeatureDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate,
        num_workers=0,
    )


@torch.inference_mode()
def accuracy(model, examples, batch_size, device):
    if not examples:
        return 0.0
    correct = 0
    total = 0
    for states, mask, labels, _ in make_loader(examples, batch_size, False, 0):
        logits = model(states.to(device), None if mask is None else mask.to(device))
        correct += int((logits.argmax(dim=-1).cpu() == labels).sum())
        total += labels.numel()
    return correct / total


def train_probe(config, train_examples, val_examples, hidden_size, classes, args, device):
    torch.manual_seed(args.seed)
    model = build_probe(
        config,
        hidden_size,
        classes,
        attention_rank=args.attention_rank,
        value_rank=args.value_rank,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        loader = make_loader(train_examples, args.batch_size, True, args.seed + epoch)
        for states, mask, labels, _ in loader:
            states = states.to(device)
            labels = labels.to(device)
            mask = None if mask is None else mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(states, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_accuracy = accuracy(model, val_examples, args.batch_size, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_accuracy": val_accuracy,
            }
        )
        if val_accuracy > best_val + 1e-8:
            best_val = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), history, best_val, best_epoch


@torch.inference_mode()
def predict(model, examples, batch_size, device, id_to_label, condition):
    records = []
    if not examples:
        return records
    for states, mask, labels, metadata in make_loader(examples, batch_size, False, 0):
        logits = model(states.to(device), None if mask is None else mask.to(device))
        probabilities = logits.softmax(dim=-1).cpu()
        predictions = probabilities.argmax(dim=-1)
        for index, row in enumerate(metadata):
            prediction = id_to_label[int(predictions[index])]
            records.append(
                {
                    "pair_id": row["pair_id"],
                    "variant": row["variant"],
                    "condition": condition,
                    "target": row["answer"],
                    "counterpart_answer": row["counterpart_answer"],
                    "memory_answer": row.get("memory_answer", row["answer"]),
                    "prediction": prediction,
                    "correct": float(prediction == row["answer"]),
                    "memory_answer_hit": float(
                        prediction == row.get("memory_answer", row["answer"])
                    ),
                    "confidence": float(probabilities[index, predictions[index]]),
                }
            )
    return records


def shuffled_examples(examples):
    output = []
    for index, target in enumerate(examples):
        source = None
        for offset in range(1, len(examples)):
            candidate = examples[(index + offset) % len(examples)]
            if candidate["pair_id"] != target["pair_id"] and candidate["answer"] != target["answer"]:
                source = candidate
                break
        if source is None:
            raise RuntimeError("Could not construct an answer-disjoint shuffled state")
        row = {**target, "feature": source["feature"], "memory_answer": source["answer"]}
        output.append(row)
    return output


def state_swapped_examples(examples):
    by_pair = defaultdict(dict)
    for example in examples:
        by_pair[example["pair_id"]][example["variant"]] = example
    output = []
    for pair in by_pair.values():
        if set(pair) != {"base", "counterfactual"}:
            continue
        for target_variant, source_variant in (("base", "counterfactual"), ("counterfactual", "base")):
            target = pair[target_variant]
            source = pair[source_variant]
            output.append(
                {**target, "feature": source["feature"], "memory_answer": source["answer"]}
            )
    return output


def condition_summary(records):
    rows = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "n": len(selected),
                "accuracy": float(np.mean([row["correct"] for row in selected])),
                "memory_answer_hit_rate": float(
                    np.mean([row["memory_answer_hit"] for row in selected])
                ),
                "mean_confidence": float(np.mean([row["confidence"] for row in selected])),
            }
        )
    return rows


def paired_metrics(records):
    selected = [row for row in records if row["condition"] == "correct"]
    by_pair = defaultdict(dict)
    for row in selected:
        by_pair[row["pair_id"]][row["variant"]] = row
    valid = [pair for pair in by_pair.values() if set(pair) == {"base", "counterfactual"}]
    return {
        "pairs": len(valid),
        "base_accuracy": float(np.mean([pair["base"]["correct"] for pair in valid])),
        "counterfactual_accuracy": float(
            np.mean([pair["counterfactual"]["correct"] for pair in valid])
        ),
        "paired_consistency": float(
            np.mean([pair["base"]["correct"] * pair["counterfactual"]["correct"] for pair in valid])
        ),
        "prediction_switch_rate": float(
            np.mean([pair["base"]["prediction"] != pair["counterfactual"]["prediction"] for pair in valid])
        ),
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate frozen-prefill answer probes")
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", default="all")
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--attention-rank", type=int, default=128)
    parser.add_argument("--value-rank", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    train_cache = PairCache(args.train_index)
    test_cache = PairCache(args.test_index)
    for key in ("model", "hidden_size", "summary_slots", "raw_layer"):
        if train_cache.index[key] != test_cache.index[key]:
            raise ValueError(f"Train/test cache mismatch for {key}")
    labels = sorted(train_cache.index["answer_vocabulary"])
    label_to_id = {label: index for index, label in enumerate(labels)}
    test_labels = set(test_cache.index["answer_vocabulary"])
    if not test_labels.issubset(label_to_id):
        raise ValueError(f"Unseen test answers: {sorted(test_labels - set(label_to_id))}")

    configs = probe_configs(train_cache.index)
    if args.configs != "all":
        selected_names = {item.strip() for item in args.configs.split(",") if item.strip()}
        configs = [config for config in configs if config["name"] in selected_names]
        missing = selected_names - {config["name"] for config in configs}
        if missing:
            raise ValueError(f"Unknown configurations: {sorted(missing)}")

    pair_indices = list(range(len(train_cache.entries)))
    random.Random(args.seed).shuffle(pair_indices)
    validation_pairs = max(1, round(len(pair_indices) * args.validation_fraction))
    val_indices = sorted(pair_indices[:validation_pairs])
    fit_indices = sorted(pair_indices[validation_pairs:])
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    experiment_rows = []

    for config in configs:
        config_out = output / config["name"]
        config_out.mkdir(parents=True, exist_ok=True)
        train_examples = materialize(train_cache, config, "correct", label_to_id, fit_indices)
        val_examples = materialize(train_cache, config, "correct", label_to_id, val_indices)
        model, history, best_val, best_epoch = train_probe(
            config,
            train_examples,
            val_examples,
            int(train_cache.index["hidden_size"]),
            len(labels),
            args,
            device,
        )
        checkpoint = {
            "format_version": 1,
            "config": config,
            "model": model.state_dict(),
            "hidden_size": int(train_cache.index["hidden_size"]),
            "labels": labels,
            "args": vars(args),
            "best_validation_accuracy": best_val,
            "best_epoch": best_epoch,
        }
        torch.save(checkpoint, config_out / "checkpoint.pt")
        write_jsonl(config_out / "train_history.jsonl", history)

        records = []
        correct_examples = materialize(test_cache, config, "correct", label_to_id)
        for condition in test_cache.index["conditions"]:
            examples = materialize(test_cache, config, condition, label_to_id)
            records.extend(
                predict(model, examples, args.batch_size, device, labels, condition)
            )
        records.extend(
            predict(
                model,
                shuffled_examples(correct_examples),
                args.batch_size,
                device,
                labels,
                "shuffled_state",
            )
        )
        records.extend(
            predict(
                model,
                state_swapped_examples(correct_examples),
                args.batch_size,
                device,
                labels,
                "counterfactual_state_swap",
            )
        )
        summaries = condition_summary(records)
        paired = paired_metrics(records)
        result = {
            "status": "complete",
            "config": config,
            "train_pairs": len(fit_indices),
            "validation_pairs": len(val_indices),
            "test_pairs": len(test_cache.entries),
            "classes": len(labels),
            "best_validation_accuracy": best_val,
            "best_epoch": best_epoch,
            "paired": paired,
            "conditions": summaries,
        }
        with open(config_out / "SUCCESS.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        write_jsonl(config_out / "per_sample.jsonl", records)
        write_csv(config_out / "condition_summary.csv", summaries)
        experiment_rows.append(
            {
                "config": config["name"],
                "kind": config["kind"],
                "layer": config["layer"],
                "slots": config.get("slots"),
                "validation_accuracy": best_val,
                **paired,
            }
        )

    best = max(
        experiment_rows,
        key=lambda row: (row["paired_consistency"], row["base_accuracy"] + row["counterfactual_accuracy"]),
    )
    write_csv(output / "probe_comparison.csv", experiment_rows)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "experiment": "Experiment A: Prefill-state answerability",
                "args": vars(args),
                "labels": labels,
                "results": experiment_rows,
                "descriptive_best_test_probe": best,
                "interpretation_guardrail": (
                    "Summary-token success shows frozen-prefill answer recoverability. "
                    "Only raw_evidence_attention success directly supports the current evidence-token transfer object."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
