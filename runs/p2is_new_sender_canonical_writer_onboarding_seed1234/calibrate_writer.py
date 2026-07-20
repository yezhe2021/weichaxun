import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from p2is_common import (
    PairCache, TokenCanonicalReader, anchor_terms, assign_gradient, canonical_to,
    cosine_between, full_attention_layers, gradient_vector, load_aligned_pair,
    load_receiver, memory_from_output, pack_answer, seed_everything, state_sha256,
    student_prefixed_prompt, write_json, write_jsonl, writer_from_checkpoint,
)


def load_frozen_reader(name, model_path, checkpoint_path, device, dtype):
    model, tokenizer = load_receiver(model_path, device, dtype)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["reader_metadata"]
    reader = TokenCanonicalReader(
        model, canonical_dim=256, rank=metadata["rank"], max_gate=metadata["max_gate"],
        gate_init=0.0, active_layers=metadata["active_layers"],
    ).to(device).eval()
    reader.load_state_dict(checkpoint["reader"])
    for parameter in list(model.parameters()) + list(reader.parameters()): parameter.requires_grad_(False)
    if metadata["active_layers"] != full_attention_layers(model): raise RuntimeError(f"{name} Reader layer interface drift")
    return model, tokenizer, reader


def student_target(model, tokenizer, reader, prompt_row, row, memory, max_length, device):
    ids, mask, labels = pack_answer(tokenizer, student_prefixed_prompt(tokenizer, prompt_row), row["answer"], max_length, device)
    with reader.inject(model, memory):
        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True)
    shifted = output.logits[:, :-1].float(); shifted_labels = labels[:, 1:]
    return output.loss.float(), shifted[shifted_labels != -100]


def student_nll(model, tokenizer, reader, prompt_row, answer, memory, max_length, device):
    ids, mask, labels = pack_answer(tokenizer, student_prefixed_prompt(tokenizer, prompt_row), answer, max_length, device)
    with reader.inject(model, memory):
        return model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True).loss.float()


def distill_topk(student_logits, teacher, device):
    indices = teacher["top_indices"].to(device); values = teacher["top_values"].float().to(device)
    if student_logits.shape[0] != indices.shape[0]: raise RuntimeError("Teacher/student answer-token lengths differ")
    probability = values.softmax(-1); selected = student_logits.log_softmax(-1).gather(-1, indices)
    return -(probability * selected).sum(-1).mean()


def detached_memory(item, device):
    keys = item["output"]["keys"].detach().requires_grad_(True)
    values = item["output"]["values"].detach().requires_grad_(True)
    return {
        "keys": keys, "values": values, "mask": torch.ones(keys.shape[0], dtype=torch.bool, device=device),
        "answer_token_mask": item["qrow"]["answer_mask"].to(device),
    }


def receiver_memory_gradients(items, receiver, teacher_cache, model_path, checkpoint_path, device, dtype, max_length, margin, switch_weight, distill_weight):
    model, tokenizer, reader = load_frozen_reader(receiver, model_path, checkpoint_path, device, dtype)
    key_grads, value_grads, losses = [], [], []
    for item in tqdm(items, desc=f"functional_{receiver}", leave=False):
        memory = detached_memory(item, device); pair = item["qpair"]; variant = item["variant"]
        row = pair[variant]; alternative = pair["counterfactual" if variant == "base" else "base"]["answer"]
        target_nll, answer_logits = student_target(model, tokenizer, reader, pair["base"], row, memory, max_length, device)
        alternative_nll = student_nll(model, tokenizer, reader, pair["base"], alternative, memory, max_length, device)
        teacher = teacher_cache.load(item["index"])[variant]["teacher"]
        distill = distill_topk(answer_logits, teacher, device)
        loss = target_nll + switch_weight * F.relu(margin + target_nll - alternative_nll) + distill_weight * distill
        (loss / len(items)).backward()
        key_grads.append(memory["keys"].grad.detach().float().cpu()); value_grads.append(memory["values"].grad.detach().float().cpu())
        losses.append(float(loss.detach().cpu()))
    if any(parameter.grad is not None for parameter in list(model.parameters()) + list(reader.parameters())):
        raise RuntimeError(f"Frozen {receiver} model or Reader received parameter gradients")
    del model, tokenizer, reader; gc.collect(); torch.cuda.empty_cache()
    return key_grads, value_grads, float(np.mean(losses))


def propagate_memory_gradients(items, key_grads, value_grads, parameters, device, retain):
    for position, item in enumerate(items):
        torch.autograd.backward(
            (item["output"]["keys"], item["output"]["values"]),
            (key_grads[position].to(device), value_grads[position].to(device)),
            retain_graph=retain or position + 1 < len(items),
        )
    vector = gradient_vector(parameters)
    for parameter in parameters: parameter.grad = None
    return vector


@torch.inference_mode()
def validation(writer, q4, old, indices, receiver_specs, device, max_length, margin):
    writer.eval(); anchor_values = []
    memories = []
    for index in indices:
        qpair, opair = load_aligned_pair(q4, old, index)
        for variant in ("base", "counterfactual"):
            output = writer(qpair[variant]["key_flat"].to(device), qpair[variant]["value_flat"].to(device))
            target = {name: opair[variant]["memory"][name].to(device) for name in ("keys", "values")}
            anchor_values.append(float(anchor_terms(output, target)["total"].cpu()))
            memories.append((qpair, variant, memory_from_output(output, qpair[variant]["answer_mask"])))
    result = {"anchor": float(np.mean(anchor_values))}
    for name, spec in receiver_specs.items():
        model, tokenizer, reader = load_frozen_reader(name, spec["model"], spec["checkpoint"], device, spec["dtype"])
        correct, losses = [], []
        for pair, variant, memory in memories:
            row = pair[variant]; alternative = pair["counterfactual" if variant == "base" else "base"]["answer"]
            target = student_nll(model, tokenizer, reader, pair["base"], row["answer"], memory, max_length, device)
            alt = student_nll(model, tokenizer, reader, pair["base"], alternative, memory, max_length, device)
            correct.append(float(target < alt)); losses.append(float((target + F.relu(margin + target - alt)).cpu()))
        result[f"{name}_loss"] = float(np.mean(losses)); result[f"{name}_preference"] = float(np.mean(correct))
        del model, tokenizer, reader; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", choices=("q4_only", "dual_only", "full"), required=True)
    parser.add_argument("--q4-index", required=True); parser.add_argument("--old-index", required=True); parser.add_argument("--ridge", required=True)
    parser.add_argument("--init-writer", required=True); parser.add_argument("--teacher4", required=True); parser.add_argument("--teacher35", required=True)
    parser.add_argument("--model4", required=True); parser.add_argument("--model35", required=True); parser.add_argument("--reader4", required=True); parser.add_argument("--reader35", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--train-pairs", type=int, default=448); parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--chunk-pairs", type=int, default=16); parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4); parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--switch-weight", type=float, default=0.5); parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="Resume from a completed epoch checkpoint")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device)
    q4, old = PairCache(args.q4_index, 4), PairCache(args.old_index, 4)
    teachers = {"qwen3_4b": PairCache(args.teacher4, 2), "qwen3_5_4b": PairCache(args.teacher35, 2)}
    writer, initial = writer_from_checkpoint(args.ridge, args.init_writer, device); writer.train()
    parameters = [parameter for parameter in writer.parameters() if parameter.requires_grad]
    if {id(p) for p in parameters} != {id(p) for p in writer.trainable_nonbase_parameters()}:
        raise RuntimeError("Calibration trainable set must exclude ridge base projections")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    anchor_weight = 1.0 if args.config == "full" else 0.0
    receiver_weights = {"qwen3_4b": 0.1, "qwen3_5_4b": 0.1}
    if args.config == "q4_only": receiver_weights["qwen3_5_4b"] = 0.0
    specs = {
        "qwen3_4b": {"model": args.model4, "checkpoint": args.reader4, "dtype": torch.float16},
        "qwen3_5_4b": {"model": args.model35, "checkpoint": args.reader35, "dtype": torch.float32},
    }
    output_dir = Path(args.out); output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    history = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as handle:
            history = [json.loads(line) for line in handle if line.strip()]
    best, best_epoch, start_epoch = float("inf"), 0, 1
    best_path = output_dir / "checkpoint_best.pt"
    if best_path.exists():
        previous_best = torch.load(best_path, map_location="cpu", weights_only=False)
        best = float(previous_best.get("criterion", float("inf")))
        best_epoch = int(previous_best.get("epoch", 0))
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("config") != args.config:
            raise RuntimeError(f"Resume config mismatch: {resume.get('config')} != {args.config}")
        writer.load_state_dict(resume["writer"], strict=True)
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"]) + 1
        if start_epoch > args.epochs:
            raise RuntimeError(f"Checkpoint epoch {resume['epoch']} already reaches --epochs {args.epochs}")
    for epoch in range(start_epoch, args.epochs + 1):
        order = list(range(args.train_pairs)); random.Random(args.seed + epoch).shuffle(order)
        for start in tqdm(range(0, len(order), args.chunk_pairs), desc=f"calibrate_{args.config}_{epoch}"):
            indices = order[start:start + args.chunk_pairs]; items = []
            for index in indices:
                qpair, opair = load_aligned_pair(q4, old, index)
                for variant in ("base", "counterfactual"):
                    output = writer(qpair[variant]["key_flat"].to(device), qpair[variant]["value_flat"].to(device))
                    target = {name: opair[variant]["memory"][name].to(device) for name in ("keys", "values")}
                    items.append({"index": index, "variant": variant, "qpair": qpair, "qrow": qpair[variant], "output": output, "target": target})
            anchor = torch.stack([anchor_terms(item["output"], item["target"])["total"] for item in items]).mean()
            anchor_grads = torch.autograd.grad(anchor, parameters, retain_graph=True, allow_unused=True)
            anchor_vector = torch.cat([torch.zeros_like(p).flatten().cpu() if g is None else g.detach().float().flatten().cpu() for p, g in zip(parameters, anchor_grads)])
            receiver_vectors, receiver_losses = {}, {}
            memory_gradients = {}
            for name in ("qwen3_4b", "qwen3_5_4b"):
                if receiver_weights[name] == 0.0: continue
                key_grad, value_grad, mean_loss = receiver_memory_gradients(
                    items, name, teachers[name], specs[name]["model"], specs[name]["checkpoint"], device,
                    specs[name]["dtype"], args.max_length, args.margin, args.switch_weight, args.distill_weight,
                )
                memory_gradients[name] = (key_grad, value_grad); receiver_losses[name] = mean_loss
            active = [name for name in ("qwen3_4b", "qwen3_5_4b") if name in memory_gradients]
            for position, name in enumerate(active):
                receiver_vectors[name] = propagate_memory_gradients(
                    items, *memory_gradients[name], parameters, device, retain=position + 1 < len(active)
                )
            combined = anchor_weight * anchor_vector
            for name in active: combined = combined + receiver_weights[name] * receiver_vectors[name]
            if not torch.isfinite(combined).all(): raise RuntimeError("Non-finite combined Writer gradient")
            assign_gradient(parameters, combined, device); torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if any(parameter.grad is not None for parameter in list(writer.key_base.parameters()) + list(writer.value_base.parameters())):
                raise RuntimeError("Frozen ridge projection received gradients")
            cosine = cosine_between(receiver_vectors["qwen3_4b"], receiver_vectors["qwen3_5_4b"]) if len(active) == 2 else None
            history.append({
                "epoch": epoch, "chunk_start": start, "pairs": len(indices), "anchor_loss": float(anchor.detach().cpu()),
                "qwen3_4b_loss": receiver_losses.get("qwen3_4b"), "qwen3_5_4b_loss": receiver_losses.get("qwen3_5_4b"),
                "reader_gradient_cosine": cosine,
            })
            write_jsonl(output_dir / "history.jsonl", history)
        selected_specs = {name: spec for name, spec in specs.items() if receiver_weights[name] > 0}
        validation_indices = range(args.train_pairs) if args.train_pairs <= 16 else range(448, 512)
        metrics = validation(writer, q4, old, validation_indices, selected_specs, device, args.max_length, args.margin)
        criterion = anchor_weight * metrics["anchor"] + sum(receiver_weights[name] * metrics[f"{name}_loss"] for name in selected_specs)
        payload = {
            "format_version": 1, "stage": f"functional_calibration_{args.config}", "config": args.config,
            "writer": {name: value.detach().cpu() for name, value in writer.state_dict().items()},
            "writer_config": initial["writer_config"], "writer_state_sha256": state_sha256(writer.state_dict()),
            "optimizer": optimizer.state_dict(), "epoch": epoch, "validation": metrics, "criterion": criterion, "args": vars(args),
        }
        torch.save(payload, output_dir / "checkpoint_latest.pt")
        if criterion < best: best, best_epoch = criterion, epoch; torch.save(payload, output_dir / "checkpoint_best.pt")
    write_json(output_dir / "TRAIN_SUCCESS.json", {
        "status": "complete", "config": args.config, "epochs": args.epochs, "best_epoch": best_epoch,
        "base_projection_frozen": True, "readers_frozen": True, "single_shared_writer": True,
    })


if __name__ == "__main__": main()
