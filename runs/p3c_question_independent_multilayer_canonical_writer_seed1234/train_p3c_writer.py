import argparse
import copy
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from p3b_common import answer_scores, best_span, decode_span, marginal_span_loss, resize_tokens, seed_everything, write_json, write_jsonl
from train_eval_p3b_probe import Cache, MultiLayerSpanProbe

from p3c_common import (
    LAYER_CONFIGS,
    MultiLayerCanonicalWriter,
    structure_loss,
    summarize_records,
    support_recall,
    teacher_distillation,
    teacher_trace,
    temporary_span_loss,
    variance_floor,
)


def load_teacher(checkpoint_path, selected_layers, device):
    teacher = MultiLayerSpanProbe("native_kv", selected_layers).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def memory(payload, device):
    states = payload["modes"]["evidence_only"]
    return states["keys"].float().to(device), states["values"].float().to(device), payload["question_state"].float().to(device), states


@torch.inference_mode()
def evaluate(writer, probe, cache, device, limit, seed, conditions):
    records = []
    selected = writer.selected_layers
    limit = min(len(cache), limit or len(cache))
    for index in tqdm(range(limit), desc="p3c_eval_writer_probe"):
        current, other = cache.load(index), cache.load((index + 1 + seed % max(1, limit - 1)) % limit)
        current_k, current_v, question, current_meta = memory(current, device)
        other_k, other_v, _, other_meta = memory(other, device)
        current_ck, current_cv = writer(current_k, current_v)
        other_ck, other_cv = writer(other_k, other_v)
        for condition in conditions:
            keys, values, metadata, evidence, source_row = current_ck, current_cv, current_meta, current["evidence"], current["row"]
            enabled = True
            if condition in {"zero", "question_only"}:
                keys, values = torch.zeros_like(keys), torch.zeros_like(values)
                enabled = condition != "question_only"
            elif condition == "shuffled":
                keys, values, metadata, evidence, source_row = other_ck, other_cv, other_meta, other["evidence"], other["row"]
            elif condition == "kv_mismatch":
                values = resize_tokens(other_cv, current_cv.shape[1])
                source_row = other["row"]
            elif condition != "correct":
                raise ValueError(condition)
            output = probe(keys, values, question, enabled)
            start, end = best_span(output["start"], output["end"])
            prediction = decode_span(evidence, metadata["offsets"], start, end)
            current_em, current_f1 = answer_scores(prediction, current["row"]["answer"])
            source_em, source_f1 = answer_scores(prediction, source_row["answer"])
            spans = set(tuple(span) for span in metadata["answer_token_spans"])
            records.append({
                "id": current["row"]["id"], "source_id": source_row["id"], "condition": condition,
                "prediction": prediction, "current_answer": current["row"]["answer"], "source_memory_answer": source_row["answer"],
                "current_answer_em": current_em, "current_answer_f1": current_f1,
                "source_memory_em": source_em, "source_memory_f1": source_f1,
                "start_accuracy": float(start in {span[0] for span in spans}),
                "end_accuracy": float(end in {span[1] for span in spans}),
                "supporting_sentence_recall": support_recall(output["support"], metadata["support_token_mask"]),
                "loss": float(temporary_span_loss(output, metadata, 0.0)),
            })
    return records


def condition(summary, name):
    return next(row for row in summary if row["condition"] == name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--projections", required=True)
    parser.add_argument("--layer-config", choices=LAYER_CONFIGS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr-writer", type=float, default=2e-4)
    parser.add_argument("--lr-probe", type=float, default=2e-4)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--structure-weight", type=float, default=0.1)
    parser.add_argument("--variance-weight", type=float, default=0.02)
    parser.add_argument("--regularization-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    projections = torch.load(args.projections, map_location="cpu", weights_only=False)
    selected = LAYER_CONFIGS[args.layer_config]
    writer = MultiLayerCanonicalWriter(projections, selected, args.rank).to(device)
    teacher = load_teacher(args.teacher_checkpoint, selected, device)
    training_probe = MultiLayerSpanProbe("pca", list(range(len(selected)))).to(device)
    writer_optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr_writer, weight_decay=0.0)
    probe_optimizer = torch.optim.AdamW(training_probe.parameters(), lr=args.lr_probe, weight_decay=0.01)
    train_cache, validation_cache = Cache(args.train_cache), Cache(args.validation_cache)
    train_limit = min(len(train_cache), args.max_train or len(train_cache))
    validation_limit = min(len(validation_cache), args.max_validation or len(validation_cache))
    history, best_score, best_writer, best_probe, best_epoch = [], -float("inf"), None, None, 0

    for epoch in range(1, args.epochs + 1):
        writer.train(); training_probe.train()
        order = list(range(train_limit))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for offset in tqdm(range(0, len(order), 2), desc=f"p3c_writer_{args.layer_config}_s{args.seed}_e{epoch}"):
            writer_optimizer.zero_grad(set_to_none=True)
            probe_optimizer.zero_grad(set_to_none=True)
            batch_memories = []
            batch_indices = order[offset: offset + 2]
            total = torch.tensor(0.0, device=device)
            components = {"task": 0.0, "teacher": 0.0, "structure": 0.0}
            for index in batch_indices:
                payload = train_cache.load(index)
                native_k, native_v, question, metadata = memory(payload, device)
                selected_k = native_k.index_select(0, torch.tensor(selected, device=device))
                selected_v = native_v.index_select(0, torch.tensor(selected, device=device))
                canonical_k, canonical_v = writer(native_k, native_v)
                decoded_k, decoded_v = writer.decode(canonical_k, canonical_v)
                with torch.no_grad():
                    native_trace = teacher_trace(teacher, selected_k, selected_v, question)
                canonical_trace = teacher_trace(teacher, decoded_k, decoded_v, question)
                teacher_loss, _ = teacher_distillation(canonical_trace, native_trace)
                probe_output = training_probe(canonical_k, canonical_v, question)
                task_loss = temporary_span_loss(probe_output, metadata, 0.05)
                geometry_loss = structure_loss(canonical_k, canonical_v, selected_k, selected_v)
                total = total + task_loss + args.teacher_weight * teacher_loss + args.structure_weight * geometry_loss
                batch_memories.append((canonical_k, canonical_v))
                components["task"] += float(task_loss.detach())
                components["teacher"] += float(teacher_loss.detach())
                components["structure"] += float(geometry_loss.detach())
            total = total / len(batch_indices)
            if len(batch_memories) > 1:
                total = total + args.variance_weight * variance_floor(batch_memories)
            total = total + args.regularization_weight * writer.weight_regularization()
            total.backward()
            torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(training_probe.parameters(), 1.0)
            writer_optimizer.step(); probe_optimizer.step()
            if any(parameter.grad is not None for parameter in teacher.parameters()):
                raise RuntimeError("Frozen Native teacher received gradients")
            losses.append(float(total.detach()))

        writer.eval(); training_probe.eval()
        records = evaluate(writer, training_probe, validation_cache, device, validation_limit, args.seed, ["correct", "zero"])
        summary = summarize_records(records)
        correct, zero = condition(summary, "correct"), condition(summary, "zero")
        score = correct["current_answer_f1"] - zero["current_answer_f1"]
        history.append({
            "epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)),
            "validation_correct_em": correct["current_answer_em"], "validation_correct_f1": correct["current_answer_f1"],
            "validation_zero_f1": zero["current_answer_f1"], "selection_score": score,
        })
        write_jsonl(output / "training_history.jsonl", history)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_writer = {name: tensor.detach().cpu().clone() for name, tensor in writer.state_dict().items()}
            best_probe = {name: tensor.detach().cpu().clone() for name, tensor in training_probe.state_dict().items()}

    writer.load_state_dict(best_writer)
    training_probe.load_state_dict(best_probe)
    writer.eval()
    checkpoint = {
        "writer": best_writer,
        "writer_config": writer.config(),
        "layer_config": args.layer_config,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "teacher_checkpoint": args.teacher_checkpoint,
        "question_independent": True,
    }
    torch.save(checkpoint, output / "writer_best.pt")
    final_records = evaluate(writer, training_probe, validation_cache, device, validation_limit, args.seed, ["correct", "zero", "shuffled", "kv_mismatch", "question_only"])
    write_jsonl(output / "training_probe_predictions.jsonl", final_records)
    write_json(output / "TRAIN_SUCCESS.json", {
        "status": "complete", "layer_config": args.layer_config, "seed": args.seed,
        "writer_checkpoint": str(output / "writer_best.pt"), "training_probe_discarded": True,
        "best_epoch": best_epoch, "best_validation_score": best_score,
        "conditions": summarize_records(final_records), "writer_config": writer.config(),
    })


if __name__ == "__main__":
    main()
