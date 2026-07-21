import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d2_common import (
    LAYER_CONFIGS, SharedGroundedReader, TeacherTraceCache, answer_scores, build_cache,
    extract_prediction, gold_logit_trace, hard_negative_mapping, initialize_shared_from_p3d,
    load_receiver, load_span_probe, memory_from_payload, pack_answer, question_prompt, read_json,
    prediction_position_mask, resize_memory, seed_everything, span_teacher, write_json, write_jsonl,
)


def reader_forward(model, tokenizer, reader, row, memory, max_length, device):
    ids, mask, labels = pack_answer(tokenizer, question_prompt(tokenizer, row), row["answer"], max_length, device)
    question_position = int(labels[0].ne(-100).nonzero()[0]) - 1
    trace = {}
    with reader.inject(model, memory, trace, question_position=question_position):
        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True)
    return output, labels, trace


def normalized_kl(student, teacher):
    student = student.clamp_min(1e-8)
    teacher = teacher.clamp_min(1e-8)
    student = student / student.sum(-1, keepdim=True).clamp_min(1e-8)
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(student.log(), teacher, reduction="batchmean")


def grounding_loss(trace, answer_mask, probe_output, memory, active_layers):
    attentions, routers = [], []
    for layer in active_layers:
        attention = trace[layer]["attention"][0, answer_mask]
        router = trace[layer]["router"][0, answer_mask]
        attentions.append(attention.mean(0))
        routers.append(router.mean(0))
    student_attention = torch.stack(attentions).mean(0)
    student_router = torch.stack(routers).mean(0)
    teacher_attention = probe_output["attention"].detach().float()
    teacher_router = probe_output["layer_weights"].detach().float()
    per_group = normalized_kl(student_attention, teacher_attention)
    router = normalized_kl(student_router[None], teacher_router[None])
    student_marginal = (student_router[:, None] * student_attention).sum(0)
    teacher_span_logits = probe_output["start"].detach() + probe_output["end"].detach() + probe_output["support"].detach()
    teacher_span = teacher_span_logits.softmax(-1)
    oracle = (memory["answer_token_mask"] | memory["support_token_mask"]).float()
    oracle = oracle / oracle.sum().clamp_min(1.0)
    target = 0.75 * teacher_span + 0.25 * oracle
    token = normalized_kl(student_marginal[None], target[None])
    return per_group + 0.5 * router + token, {
        "student_attention": student_attention, "student_router": student_router,
        "teacher_attention": teacher_attention, "teacher_router": teacher_router,
    }


def execution_losses(output, labels, trace, teacher, active_layers, device):
    answer_mask = prediction_position_mask(labels)
    teacher_delta = teacher["text_delta"].float().to(device).index_select(0, torch.tensor(active_layers, device=device))
    question_hidden = teacher["question_hidden"].float().to(device).index_select(0, torch.tensor(active_layers, device=device))
    student_delta = torch.stack([trace[layer]["delta"][0, answer_mask].float() for layer in active_layers])
    answer_tokens = min(student_delta.shape[1], teacher_delta.shape[1])
    student_delta, teacher_delta = student_delta[:, :answer_tokens], teacher_delta[:, :answer_tokens]
    question_hidden = question_hidden[:, :answer_tokens]
    cosine = 1.0 - F.cosine_similarity(student_delta, teacher_delta, dim=-1).mean()
    normalized_mse = F.mse_loss(
        F.layer_norm(student_delta, (student_delta.shape[-1],)),
        F.layer_norm(teacher_delta, (teacher_delta.shape[-1],)),
    )
    reader_gold = gold_logit_trace(output.logits, labels).float()
    question_gold = teacher["question_gold_logits"].float().to(device)[: len(reader_gold)]
    text_gold_delta = teacher["text_gold_logit_delta"].float().to(device)[: len(reader_gold)]
    logit_delta = F.smooth_l1_loss(reader_gold - question_gold, text_gold_delta)
    q_rms = question_hidden.square().mean(-1).sqrt().clamp_min(1e-6)
    reader_ratio = student_delta.square().mean(-1).sqrt() / q_rms
    teacher_ratio = teacher_delta.square().mean(-1).sqrt() / q_rms
    norm = F.smooth_l1_loss(torch.log1p(reader_ratio), torch.log1p(teacher_ratio))
    return cosine + normalized_mse + 0.1 * logit_delta, norm, {
        "cosine": cosine.detach(), "normalized_mse": normalized_mse.detach(),
        "logit_delta": logit_delta.detach(), "reader_ratio": reader_ratio.detach().mean(),
        "teacher_ratio": teacher_ratio.detach().mean(),
    }


@torch.inference_mode()
def generate(model, tokenizer, reader, row, memory, max_new_tokens):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    kwargs = dict(
        **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    trace = {}
    with reader.inject(model, memory, trace):
        output = model.generate(**kwargs)
    tokens = output[0, encoded["input_ids"].shape[1]:]
    compatibility = float(np.mean([
        torch.sigmoid(values["compatibility_score"]).mean().item() for values in trace.values()
    ])) if trace else 0.0
    return tokenizer.decode(tokens, skip_special_tokens=True), compatibility


@torch.inference_mode()
def free_running_validation(model, tokenizer, reader, cache, negative, device, limit, max_new_tokens, alpha, beta, gamma):
    rows = []
    for index in range(min(limit, len(cache))):
        payload, other = cache.load(index), cache.load(negative[index])
        correct = memory_from_payload(payload, device)
        wrong = resize_memory(memory_from_payload(other, device), correct["keys"].shape[1])
        correct_text, correct_compat = generate(model, tokenizer, reader, payload["row"], correct, max_new_tokens)
        wrong_text, wrong_compat = generate(model, tokenizer, reader, payload["row"], wrong, max_new_tokens)
        correct_prediction, _ = extract_prediction(correct_text)
        wrong_prediction, _ = extract_prediction(wrong_text)
        correct_em, correct_f1 = answer_scores(correct_prediction, payload["row"]["answer"])
        wrong_em, wrong_f1 = answer_scores(wrong_prediction, payload["row"]["answer"])
        rows.append({
            "type": payload["row"].get("type", "unknown"), "correct_em": correct_em, "correct_f1": correct_f1,
            "wrong_em": wrong_em, "wrong_f1": wrong_f1, "compat": float(correct_compat > wrong_compat),
        })
    f1 = float(np.mean([row["correct_f1"] for row in rows]))
    shuffled_f1 = float(np.mean([row["wrong_f1"] for row in rows]))
    bridge = [row for row in rows if row["type"] == "bridge"]
    bridge_f1 = float(np.mean([row["correct_f1"] for row in bridge])) if bridge else 0.0
    compatibility = float(np.mean([row["compat"] for row in rows]))
    score = f1 + alpha * bridge_f1 + beta * (f1 - shuffled_f1) + gamma * compatibility
    return {
        "selection_score": score, "correct_f1": f1, "bridge_f1": bridge_f1,
        "shuffled_f1": shuffled_f1, "correct_minus_shuffled_f1": f1 - shuffled_f1,
        "compatibility_accuracy": compatibility, "n": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--p3d-checkpoint", required=True)
    parser.add_argument("--span-probe", required=True)
    parser.add_argument("--teacher-train", required=True)
    parser.add_argument("--teacher-validation", required=True)
    parser.add_argument("--layer-config", choices=tuple(LAYER_CONFIGS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("small", "full"), required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--small-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--lambda-ground", type=float, default=0.2)
    parser.add_argument("--lambda-exec", type=float, default=0.2)
    parser.add_argument("--lambda-compat", type=float, default=0.2)
    parser.add_argument("--lambda-norm", type=float, default=0.1)
    parser.add_argument("--compat-margin", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--overfit-f1", type=float, default=0.95)
    parser.add_argument("--overfit-gap", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    protocol = read_json(args.protocol)
    train_cache, validation_cache = build_cache(protocol, "train"), build_cache(protocol, "validation")
    train_teacher, validation_teacher = TeacherTraceCache(args.teacher_train), TeacherTraceCache(args.teacher_validation)
    if len(train_cache) != len(train_teacher.entries) or len(validation_cache) != len(validation_teacher.entries):
        raise RuntimeError("Execution teacher cache is not aligned with Canonical cache")
    model, tokenizer = load_receiver(args.model, device)
    active_layers = LAYER_CONFIGS[args.layer_config]
    reader = SharedGroundedReader(model, active_layers).to(device)
    old_checkpoint = torch.load(args.p3d_checkpoint, map_location="cpu", weights_only=False)
    initialize_shared_from_p3d(reader, old_checkpoint)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial["reader_metadata"] != reader.metadata():
            raise RuntimeError("Grounded Reader init interface mismatch")
        reader.load_state_dict(initial["reader"])
    probe = load_span_probe(args.span_probe, device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen3-4B backbone must remain frozen")
    trainable = [parameter for parameter in reader.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    train_limit = min(args.small_samples, len(train_cache)) if args.mode == "small" else len(train_cache)
    negative = hard_negative_mapping(train_cache)
    validation_negative = hard_negative_mapping(validation_cache)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history, validation_history = [], []
    best_score, best_epoch, update_steps = -float("inf"), 0, 0
    overfit_pass = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        reader.train()
        order = list(range(train_limit))
        random.Random(args.seed + epoch).shuffle(order)
        pending = 0
        for position, index in enumerate(tqdm(order, desc=f"p3d2_{args.layer_config}_{args.mode}_e{epoch}")):
            payload, wrong_payload = train_cache.load(index), train_cache.load(negative[index])
            memory = memory_from_payload(payload, device)
            wrong = resize_memory(memory_from_payload(wrong_payload, device), memory["keys"].shape[1])
            teacher = train_teacher.load(index)
            if teacher["sample_id"] != payload["row"]["id"]:
                raise RuntimeError("Teacher/sample ID mismatch")
            output_model, labels, trace = reader_forward(model, tokenizer, reader, payload["row"], memory, args.max_length, device)
            answer_mask = prediction_position_mask(labels)
            with torch.no_grad():
                probe_output = span_teacher(probe, payload, device)
            ground, _ = grounding_loss(trace, answer_mask, probe_output, memory, active_layers)
            execution, norm, execution_metrics = execution_losses(output_model, labels, trace, teacher, active_layers, device)
            question_summary = teacher["question_state"][active_layers[-1]].float().to(device)
            correct_score = reader.compatibility_score(question_summary, memory).mean()
            wrong_score = reader.compatibility_score(question_summary, wrong).mean()
            compatibility = F.relu(args.compat_margin - correct_score + wrong_score)
            answer = output_model.loss.float()
            loss = answer + args.lambda_ground * ground + args.lambda_exec * execution + args.lambda_compat * compatibility + args.lambda_norm * norm
            (loss / args.gradient_accumulation).backward()
            pending += 1
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Frozen Qwen3-4B parameter received gradients")
            if pending == args.gradient_accumulation or position + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                pending = 0; update_steps += 1
            history.append({
                "epoch": epoch, "index": index, "id": payload["row"]["id"],
                "answer_nll": float(answer.detach()), "ground": float(ground.detach()),
                "execution": float(execution.detach()), "compatibility": float(compatibility.detach()),
                "norm": float(norm.detach()), "loss": float(loss.detach()),
                "correct_compatibility_score": float(correct_score.detach()),
                "wrong_compatibility_score": float(wrong_score.detach()),
                **{name: float(value) for name, value in execution_metrics.items()},
            })
        reader.eval()
        selection_cache = train_cache if args.mode == "small" else validation_cache
        selection_negative = negative if args.mode == "small" else validation_negative
        selection_limit = train_limit if args.mode == "small" else min(args.validation_samples, len(validation_cache))
        validation = free_running_validation(
            model, tokenizer, reader, selection_cache, selection_negative, device,
            selection_limit, args.max_new_tokens,
            args.alpha, args.beta, args.gamma,
        )
        checkpoint = {
            "format_version": 1, "stage": "P3-D2", "layer_config": args.layer_config,
            "reader": {name: tensor.detach().cpu() for name, tensor in reader.state_dict().items()},
            "reader_metadata": reader.metadata(), "epoch": epoch, "selection": validation,
            "only_reader_trainable": True, "writer_frozen": True, "receiver_backbone_frozen": True,
            "args": vars(args),
        }
        torch.save(checkpoint, output / "checkpoint_latest.pt")
        if validation["selection_score"] > best_score:
            best_score, best_epoch = validation["selection_score"], epoch
            torch.save(checkpoint, output / "checkpoint_best.pt")
        write_jsonl(output / "history.jsonl", history)
        validation_history.append({"epoch": epoch, **validation})
        write_jsonl(output / "validation_history.jsonl", validation_history)
        if args.mode == "small":
            overfit_pass = (
                validation["correct_f1"] >= args.overfit_f1
                and validation["correct_minus_shuffled_f1"] >= args.overfit_gap
            )
            if overfit_pass:
                break
    write_json(output / "TRAIN_SUCCESS.json", {
        "status": "complete", "layer_config": args.layer_config, "mode": args.mode,
        "best_epoch": best_epoch, "best_selection_score": best_score,
        "reader_parameters": sum(parameter.numel() for parameter in reader.parameters()),
        "receiver_parameters_updated": 0, "writer_parameters_updated": 0,
        "update_steps": update_steps, "checkpoint_selection": "periodic free-running composite S",
        "overfit_pass": overfit_pass if args.mode == "small" else None,
        "overfit_thresholds": {"correct_f1": args.overfit_f1, "correct_minus_shuffled_f1": args.overfit_gap},
    })


if __name__ == "__main__":
    main()
