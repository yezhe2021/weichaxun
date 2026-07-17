import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from audit_common import LazyPairCache, load_manifest, verify_manifest_cache, write_jsonl
from functional_probes import KVFunctionalProbe, ReusedAttentionPoolProbe
from p2e_writer import StructurePreservingNativeKVWriter


def load_writer(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    sender = checkpoint["sender_geometry"]
    receiver = checkpoint["receiver_geometry"]
    writer = StructurePreservingNativeKVWriter(
        sender_layers=sender["layers"],
        sender_heads=sender["kv_heads"],
        sender_head_dim=sender["head_dim"],
        receiver_layers=receiver["layers"],
        receiver_heads=receiver["kv_heads"],
        receiver_head_dim=receiver["head_dim"],
        top_k=int(args["top_k"]),
        adapter_rank=int(args["adapter_rank"]),
        shared_routing=bool(checkpoint["features"]["shared_routing"]),
        route_residual_scale=float(args["route_residual_scale"]),
        teacher_k_rms=checkpoint["writer"]["teacher_k_rms"],
    ).to(device).eval()
    writer.load_state_dict(checkpoint["writer"])
    for parameter in writer.parameters():
        parameter.requires_grad_(False)
    return writer


def cpu_memory(memory):
    output = {
        "keys": [tensor.detach().half().cpu() for tensor in memory["keys"]],
        "values": [tensor.detach().half().cpu() for tensor in memory["values"]],
    }
    if "answer_token_mask" in memory:
        output["answer_token_mask"] = memory["answer_token_mask"].bool().cpu()
    return output


def device_memory(memory, device):
    return {
        "keys": [tensor.to(device) for tensor in memory["keys"]],
        "values": [tensor.to(device) for tensor in memory["values"]],
        **(
            {"answer_token_mask": memory["answer_token_mask"].to(device)}
            if "answer_token_mask" in memory else {}
        ),
    }


def materialize_stage(stage, split, manifest, raw_cache, hidden_cache, writer, device):
    cache = hidden_cache if stage == "sender_final_hidden" else raw_cache
    examples = []
    for record in tqdm(manifest[split], desc=f"materialize_{stage}_{split}"):
        pair = cache.load(int(record["index"]))
        for variant in ("base", "counterfactual"):
            row = pair[variant]
            if stage == "sender_final_hidden":
                feature = row["states"]["correct"]["raw_evidence"]["28"].half()
            else:
                memory = row["memory"]
                if stage == "raw_last_kv":
                    feature = {
                        "keys": [memory["keys"][-1]],
                        "values": [memory["values"][-1]],
                        "answer_token_mask": memory["answer_token_mask"],
                    }
                elif stage == "raw_all_kv":
                    feature = memory
                elif stage.startswith("writer_"):
                    with torch.inference_mode():
                        feature = cpu_memory(writer(device_memory(memory, device), output_dtype=torch.float16))
                else:
                    raise ValueError(stage)
            examples.append(
                {
                    "pair_id": row["pair_id"],
                    "variant": variant,
                    "answer": row["answer"],
                    "counterpart_answer": row["counterpart_answer"],
                    "feature": feature,
                }
            )
    return examples


def move_feature(feature, device):
    if torch.is_tensor(feature):
        return feature.to(device)
    return device_memory(feature, device)


def zero_feature(feature):
    if torch.is_tensor(feature):
        return torch.zeros_like(feature)
    return {
        "keys": [torch.zeros_like(value) for value in feature["keys"]],
        "values": [torch.zeros_like(value) for value in feature["values"]],
    }


def mismatched_feature(key_source, value_source):
    if torch.is_tensor(key_source):
        return value_source
    values = []
    for key, value in zip(key_source["keys"], value_source["values"]):
        if value.shape[1] == key.shape[1]:
            values.append(value)
        else:
            indices = torch.linspace(0, value.shape[1] - 1, key.shape[1]).round().long()
            values.append(value.index_select(1, indices))
    return {"keys": key_source["keys"], "values": values}


def build_probe(stage, examples, key_rank, value_rank, classes):
    feature = examples[0]["feature"]
    if torch.is_tensor(feature):
        raise ValueError("sender_final_hidden must use the reused Experiment-A probe")
    return KVFunctionalProbe(
        layers=len(feature["keys"]),
        heads=feature["keys"][0].shape[0],
        head_dim=feature["keys"][0].shape[-1],
        key_rank=key_rank,
        value_rank=value_rank,
        classes=classes,
    )


@torch.inference_mode()
def validation_metrics(model, examples, labels, device):
    model.eval()
    losses = []
    correct = 0
    for row in examples:
        logits = model(move_feature(row["feature"], device))
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


def train_probe(model, examples, validation, labels, args, device):
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
            logits = model(move_feature(row["feature"], device)).unsqueeze(0)
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


def load_reused_hidden_probe(path, label_names, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["labels"] != label_names:
        raise RuntimeError("Experiment-A checkpoint labels do not match this audit")
    old_args = checkpoint["args"]
    model = ReusedAttentionPoolProbe(
        hidden_size=int(checkpoint["hidden_size"]),
        classes=len(label_names),
        attention_rank=int(old_args["attention_rank"]),
        value_rank=int(old_args["value_rank"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


@torch.inference_mode()
def evaluate(model, examples, labels, id_to_label, device, condition):
    records = []
    losses = []
    model.eval()
    for row in examples:
        logits = model(move_feature(row["feature"], device))
        target_id = labels[row["answer"]]
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id], device=device))
        prediction = id_to_label[int(logits.argmax())]
        memory_answer = row.get("memory_answer", row["answer"])
        losses.append(float(loss.cpu()))
        records.append(
            {
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "condition": condition,
                "target": row["answer"],
                "memory_answer": memory_answer,
                "prediction": prediction,
                "correct": float(prediction == row["answer"]),
                "memory_answer_hit": float(prediction == memory_answer),
                "loss": float(loss.cpu()),
            }
        )
    return records, float(np.mean(losses))


def control_examples(examples, condition):
    by_pair = {}
    for row in examples:
        by_pair.setdefault(row["pair_id"], {})[row["variant"]] = row
    output = []
    for index, target in enumerate(examples):
        if condition == "zero":
            source_feature = zero_feature(target["feature"])
            memory_answer = None
        elif condition == "counterfactual_state_swap":
            source = by_pair[target["pair_id"]][
                "counterfactual" if target["variant"] == "base" else "base"
            ]
            source_feature = source["feature"]
            memory_answer = source["answer"]
        else:
            source = next(
                examples[(index + offset) % len(examples)]
                for offset in range(1, len(examples))
                if examples[(index + offset) % len(examples)]["pair_id"] != target["pair_id"]
                and examples[(index + offset) % len(examples)]["answer"] != target["answer"]
            )
            source_feature = (
                source["feature"]
                if condition == "shuffled"
                else mismatched_feature(target["feature"], source["feature"])
            )
            memory_answer = source["answer"]
        output.append(
            {**target, "feature": source_feature, "memory_answer": memory_answer}
        )
    return output


def summarize(records, train_loss, validation_loss, test_loss):
    correct = [row for row in records if row["condition"] == "correct"]
    by_pair = {}
    for row in correct:
        by_pair.setdefault(row["pair_id"], {})[row["variant"]] = row
    pairs = list(by_pair.values())
    condition_rows = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        condition_rows.append(
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
        "conditions": condition_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the quick hidden/raw/Writer functional probes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-train-index", required=True)
    parser.add_argument("--raw-test-index", required=True)
    parser.add_argument("--hidden-train-index", required=True)
    parser.add_argument("--hidden-test-index", required=True)
    parser.add_argument("--task-writer-checkpoint", required=True)
    parser.add_argument("--shared-writer-checkpoint", required=True)
    parser.add_argument("--reused-hidden-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--key-rank", type=int, default=128)
    parser.add_argument("--value-rank", type=int, default=256)
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
    manifest = load_manifest(args.manifest)
    raw_train = LazyPairCache(args.raw_train_index)
    raw_test = LazyPairCache(args.raw_test_index)
    hidden_train = LazyPairCache(args.hidden_train_index)
    hidden_test = LazyPairCache(args.hidden_test_index)
    for split, cache in (("train", raw_train), ("validation", raw_train), ("test", raw_test)):
        verify_manifest_cache(manifest, split, cache)
    for split, cache in (("train", hidden_train), ("validation", hidden_train), ("test", hidden_test)):
        verify_manifest_cache(manifest, split, cache)
    label_names = sorted(
        {
            answer
            for entry in raw_train.entries
            for answer in (entry["base_answer"], entry["counterfactual_answer"])
        }
    )
    labels = {label: index for index, label in enumerate(label_names)}
    stages = (
        "sender_final_hidden",
        "raw_last_kv",
        "raw_all_kv",
        "writer_task_only_kv",
        "writer_shared_span_relation_kv",
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for stage in stages:
        writer = None
        if stage == "writer_task_only_kv":
            writer = load_writer(args.task_writer_checkpoint, device)
        elif stage == "writer_shared_span_relation_kv":
            writer = load_writer(args.shared_writer_checkpoint, device)
        train = materialize_stage(stage, "train", manifest, raw_train, hidden_train, writer, device)
        validation = materialize_stage(
            stage, "validation", manifest, raw_train, hidden_train, writer, device
        )
        test = materialize_stage(stage, "test", manifest, raw_test, hidden_test, writer, device)
        del writer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        torch.manual_seed(args.seed)
        reused_checkpoint = None
        if stage == "sender_final_hidden":
            model, reused_checkpoint = load_reused_hidden_probe(
                args.reused_hidden_checkpoint, label_names, device
            )
            history = []
            best_validation_accuracy = float(
                reused_checkpoint.get("best_validation_accuracy", 0.0)
            )
        else:
            model = build_probe(
                stage, train, args.key_rank, args.value_rank, len(labels)
            ).to(device)
            history, best_validation_accuracy = train_probe(
                model, train, validation, labels, args, device
            )
        _, validation_loss = evaluate(
            model, validation, labels, label_names, device, "validation"
        )
        correct_records, test_loss = evaluate(
            model, test, labels, label_names, device, "correct"
        )
        records = list(correct_records)
        for condition in ("counterfactual_state_swap", "shuffled", "mismatched", "zero"):
            if stage == "sender_final_hidden" and condition == "mismatched":
                continue
            control, _ = evaluate(
                model,
                control_examples(test, condition),
                labels,
                label_names,
                device,
                condition,
            )
            records.extend(control)
        result = summarize(
            records,
            None if not history else history[-1]["train_loss"],
            validation_loss,
            test_loss,
        )
        result.update(
            {
                "status": "complete",
                "stage": stage,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "epochs": args.epochs,
                "epochs_completed": len(history),
                "best_validation_accuracy": best_validation_accuracy,
                "weight_policy": (
                    "reused_frozen_experiment_a"
                    if reused_checkpoint is not None
                    else "stage_specific_trained"
                ),
                "source_checkpoint": (
                    args.reused_hidden_checkpoint
                    if reused_checkpoint is not None
                    else None
                ),
            }
        )
        stage_out = output / stage
        stage_out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "stage": stage,
                "args": vars(args),
                "weight_policy": result["weight_policy"],
            },
            stage_out / "probe.pt",
        )
        write_jsonl(stage_out / "train_history.jsonl", history)
        write_jsonl(stage_out / "per_sample.jsonl", records)
        with open(stage_out / "SUCCESS.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        summary_rows.append({key: value for key, value in result.items() if key != "conditions"})
        del model, train, validation, test
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with open(output / "MEMORY_STAGES_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", "stages": summary_rows}, handle, indent=2)


if __name__ == "__main__":
    main()
