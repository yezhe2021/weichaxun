import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_common import load_manifest, write_jsonl
from functional_probes import LayerStateProbe, VectorStateProbe


class ChainCache:
    def __init__(self, root):
        self.root = Path(root)

    def load_split(self, split):
        split_root = self.root / split
        with open(split_root / "index.json", encoding="utf-8") as handle:
            index = json.load(handle)
        examples = []
        for entry in index["pair_files"]:
            payload = torch.load(
                split_root / entry["file"], map_location="cpu", weights_only=False
            )
            examples.extend(payload["examples"])
        return index, examples


def select_feature(row, stage):
    states = row["states"]
    if stage == "reader_all_readout":
        return states["reader_readouts"]
    if stage == "final_cumulative_delta":
        return states["final_cumulative_delta"]
    if stage == "receiver_final_hidden":
        return states["receiver_final_hidden"]
    raise ValueError(stage)


def build_examples(rows, stage, condition=None):
    output = []
    for row in rows:
        if condition is not None and row["condition"] != condition:
            continue
        output.append({**row, "feature": select_feature(row, stage)})
    return output


def build_probe(stage, feature, latent_dim, classes):
    if stage == "reader_all_readout":
        return LayerStateProbe(feature.shape[-1], latent_dim, classes)
    return VectorStateProbe(feature.shape[-1], latent_dim, classes)


@torch.inference_mode()
def validation_metrics(model, examples, labels, device):
    model.eval()
    losses = []
    correct = 0
    for row in examples:
        logits = model(row["feature"].to(device))
        target = labels[row["answer"]]
        losses.append(
            float(
                F.cross_entropy(
                    logits.unsqueeze(0), torch.tensor([target], device=device)
                ).cpu()
            )
        )
        correct += int(int(logits.argmax()) == target)
    return float(np.mean(losses)), correct / len(examples)


def train(model, examples, validation, labels, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_state = None
    best_accuracy = -1.0
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(examples)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for position, index in enumerate(order):
            row = examples[index]
            logits = model(row["feature"].to(device)).unsqueeze(0)
            target = torch.tensor([labels[row["answer"]]], device=device)
            loss = F.cross_entropy(logits, target)
            (loss / args.gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if (position + 1) % args.gradient_accumulation == 0 or position + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation_loss, validation_accuracy = validation_metrics(
            model, validation, labels, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        improved = validation_accuracy > best_accuracy or (
            validation_accuracy == best_accuracy and validation_loss < best_loss
        )
        if improved:
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Probe training produced no checkpoint")
    model.load_state_dict(best_state)
    return history, best_accuracy


@torch.inference_mode()
def evaluate(model, examples, labels, label_names, device, override_condition=None):
    records = []
    losses = []
    model.eval()
    for row in examples:
        logits = model(row["feature"].to(device))
        target = labels[row["answer"]]
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device))
        prediction = label_names[int(logits.argmax())]
        memory_answer = row.get("memory_answer")
        losses.append(float(loss.cpu()))
        records.append(
            {
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "condition": override_condition or row["condition"],
                "target": row["answer"],
                "memory_answer": memory_answer,
                "prediction": prediction,
                "correct": float(prediction == row["answer"]),
                "memory_answer_hit": float(
                    memory_answer is not None and prediction == memory_answer
                ),
                "loss": float(loss.cpu()),
            }
        )
    return records, float(np.mean(losses))


def summarize(records, train_loss, validation_loss, test_loss):
    correct = [row for row in records if row["condition"] == "correct"]
    by_pair = {}
    for row in correct:
        by_pair.setdefault(row["pair_id"], {})[row["variant"]] = row
    pairs = list(by_pair.values())
    conditions = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        conditions.append(
            {
                "condition": condition,
                "accuracy": float(np.mean([row["correct"] for row in selected])),
                "memory_answer_hit_rate": float(
                    np.mean([row["memory_answer_hit"] for row in selected])
                ),
                "loss": float(np.mean([row["loss"] for row in selected])),
            }
        )
    return {
        "base_em": float(np.mean([pair["base"]["correct"] for pair in pairs])),
        "counterfactual_em": float(
            np.mean([pair["counterfactual"]["correct"] for pair in pairs])
        ),
        "paired_consistency": float(
            np.mean([pair["base"]["correct"] * pair["counterfactual"]["correct"] for pair in pairs])
        ),
        "prediction_switch_rate": float(
            np.mean([pair["base"]["prediction"] != pair["counterfactual"]["prediction"] for pair in pairs])
        ),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "conditions": conditions,
    }


def direct_logits(rows, label_names):
    records = []
    for row in rows:
        prediction = label_names[int(row["states"]["candidate_first_logits"].argmax())]
        records.append(
            {
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "condition": row["condition"],
                "target": row["answer"],
                "memory_answer": row.get("memory_answer"),
                "prediction": prediction,
                "correct": float(prediction == row["answer"]),
                "memory_answer_hit": float(
                    row.get("memory_answer") is not None
                    and prediction == row["memory_answer"]
                ),
                "loss": 0.0,
            }
        )
    off_rows = []
    for row in rows:
        if row["condition"] != "correct":
            continue
        prediction = label_names[
            int(row["states"]["reader_off_candidate_first_logits"].argmax())
        ]
        off_rows.append(
            {
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "condition": "reader_off",
                "target": row["answer"],
                "memory_answer": None,
                "prediction": prediction,
                "correct": float(prediction == row["answer"]),
                "memory_answer_hit": 0.0,
                "loss": 0.0,
            }
        )
    return records + off_rows


def main():
    parser = argparse.ArgumentParser(description="Train quick Reader/readout/final-state probes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chain-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    load_manifest(args.manifest)
    cache = ChainCache(args.chain_cache)
    train_index, train_rows = cache.load_split("train")
    _, validation_rows = cache.load_split("validation")
    _, test_rows = cache.load_split("test")
    label_names = train_index["labels"]
    labels = {label: index for index, label in enumerate(label_names)}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for stage in ("reader_all_readout", "final_cumulative_delta", "receiver_final_hidden"):
        train_examples = build_examples(train_rows, stage, "correct")
        validation_examples = build_examples(validation_rows, stage, "correct")
        test_examples = build_examples(test_rows, stage)
        torch.manual_seed(args.seed)
        model = build_probe(
            stage, train_examples[0]["feature"], args.latent_dim, len(labels)
        ).to(device)
        history, best_validation_accuracy = train(
            model, train_examples, validation_examples, labels, args, device
        )
        _, validation_loss = evaluate(
            model, validation_examples, labels, label_names, device
        )
        records, test_loss = evaluate(
            model, test_examples, labels, label_names, device
        )
        if stage == "receiver_final_hidden":
            off_examples = []
            for row in test_rows:
                if row["condition"] == "correct":
                    off_examples.append(
                        {
                            **row,
                            "feature": row["states"]["reader_off_final_hidden"],
                            "memory_answer": None,
                        }
                    )
            off_records, _ = evaluate(
                model, off_examples, labels, label_names, device, "reader_off"
            )
            records.extend(off_records)
        result = summarize(
            records, history[-1]["train_loss"], validation_loss, test_loss
        )
        result.update(
            {
                "status": "complete",
                "stage": stage,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "epochs_completed": len(history),
                "best_validation_accuracy": best_validation_accuracy,
                "weight_policy": "stage_specific_trained",
            }
        )
        stage_out = output / stage
        stage_out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, stage_out / "probe.pt")
        write_jsonl(stage_out / "train_history.jsonl", history)
        write_jsonl(stage_out / "per_sample.jsonl", records)
        with open(stage_out / "SUCCESS.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        summaries.append({key: value for key, value in result.items() if key != "conditions"})

    logits_records = direct_logits(test_rows, label_names)
    logits_result = summarize(logits_records, None, None, None)
    logits_result.update({"status": "complete", "stage": "first_token_logits"})
    logits_out = output / "first_token_logits"
    logits_out.mkdir(parents=True, exist_ok=True)
    write_jsonl(logits_out / "per_sample.jsonl", logits_records)
    with open(logits_out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(logits_result, handle, indent=2, ensure_ascii=False)
    summaries.append({key: value for key, value in logits_result.items() if key != "conditions"})
    with open(output / "CHAIN_STAGES_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", "stages": summaries}, handle, indent=2)


if __name__ == "__main__":
    main()
