import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from causal_common import inject_state, iter_cache, parse_dtype, resolve_device, write_jsonl
from p15_common import (
    GeneralEvidenceAdapter,
    answer_token_view,
    pack_answer,
    render_student_prompt,
    render_teacher_prompt,
)


def memories(example, condition, device, negative=None):
    own_a = example["memory_a"].to(device).unsqueeze(0)
    own_b = example["memory_b"].to(device).unsqueeze(0)
    if condition == "correct":
        return own_a, own_b
    if condition == "zero":
        return None, None
    if condition == "shuffled":
        return own_a, negative["memory_b"].to(device).unsqueeze(0) if negative else None
    if condition == "mismatched":
        if negative is None:
            return None, None
        return negative["memory_a"].to(device).unsqueeze(0), negative["memory_b"].to(device).unsqueeze(0)
    if condition == "corrupted":
        generator = torch.Generator(device=device).manual_seed(sum(map(ord, example["id"])))
        noise_a = torch.randn(own_a.shape, generator=generator, device=device, dtype=own_a.dtype)
        noise_b = torch.randn(own_b.shape, generator=generator, device=device, dtype=own_b.dtype)
        return noise_a * own_a.float().std().clamp_min(1e-3), noise_b * own_b.float().std().clamp_min(1e-3)
    raise ValueError(f"Unknown memory condition: {condition}")


def read_state(adapter, example, condition, device, negative=None):
    question = example["question_state"].to(device=device, dtype=torch.float32).unsqueeze(0)
    memory_a, memory_b = memories(example, condition, device, negative)
    return adapter.reader(question, memory_a, memory_b)


def student_forward(receiver, tokenizer, adapter, example, state, answer, max_length, device):
    prompt = render_student_prompt(tokenizer, example)
    ids, mask, labels, prompt_length = pack_answer(tokenizer, prompt, answer, max_length, device)

    def selector(hidden):
        return prompt_length - 1, hidden.shape[1] - 1

    diagnostics = {}
    with inject_state(receiver, adapter, state, selector, diagnostics):
        output = receiver(
            input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True
        )
    answer_logits, answer_labels = answer_token_view(output.logits, labels)
    return output.loss.float(), answer_logits.float(), answer_labels, diagnostics


@torch.no_grad()
def teacher_logits(receiver, tokenizer, example, answer, max_length, device):
    prompt = render_teacher_prompt(tokenizer, example)
    ids, mask, labels, _ = pack_answer(tokenizer, prompt, answer, max_length, device)
    output = receiver(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    logits, answer_labels = answer_token_view(output.logits, labels)
    return logits.float(), answer_labels


def main():
    parser = argparse.ArgumentParser(description="Train general external-memory dependence adapter")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--state-dim", type=int, default=1024)
    parser.add_argument("--reader-heads", type=int, default=8)
    parser.add_argument("--reader-rounds", type=int, default=3)
    parser.add_argument("--writer-layers", type=int, nargs="+", default=[12, 24, 34])
    parser.add_argument("--writer-bottleneck", type=int, default=256)
    parser.add_argument("--max-gate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--depend-weight", type=float, default=0.2)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--depend-margin", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    with open(args.train_index, encoding="utf-8") as handle:
        cache_index = json.load(handle)
    memory_dim = int(cache_index["hidden_size"])

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)

    adapter = GeneralEvidenceAdapter(
        memory_dim=memory_dim,
        receiver_dim=int(receiver.config.hidden_size),
        state_dim=args.state_dim,
        reader_heads=args.reader_heads,
        reader_rounds=args.reader_rounds,
        writer_layers=args.writer_layers,
        writer_bottleneck=args.writer_bottleneck,
        max_gate=args.max_gate,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    previous = None

    for epoch in range(args.epochs):
        examples = iter_cache(args.train_index, shuffle=True, seed=args.seed + epoch)
        progress = tqdm(examples, total=int(cache_index["n"]), desc=f"p15_epoch_{epoch}")
        optimizer.zero_grad(set_to_none=True)
        for position, example in enumerate(progress):
            positive_state = read_state(adapter, example, "correct", device)
            generation_loss, student_logits, student_labels, diagnostics = student_forward(
                receiver, tokenizer, adapter, example, positive_state, example["answer"], args.max_length, device
            )
            target_teacher_logits, teacher_labels = teacher_logits(
                receiver, tokenizer, example, example["answer"], args.max_length, device
            )
            if student_labels.shape != teacher_labels.shape or not torch.equal(student_labels, teacher_labels):
                raise RuntimeError("Teacher and student answer-token alignment failed")
            temperature = args.distill_temperature
            distill_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(target_teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)

            negative_types = ("zero", "shuffled", "mismatched", "corrupted")
            negative_type = negative_types[global_step % len(negative_types)]
            negative_state = read_state(adapter, example, negative_type, device, previous)
            negative_nll, _, _, _ = student_forward(
                receiver, tokenizer, adapter, example, negative_state, example["answer"], args.max_length, device
            )
            depend_loss = F.relu(args.depend_margin + generation_loss - negative_nll)
            active = float(global_step >= args.warmup_steps)
            total = generation_loss + active * (
                args.distill_weight * distill_loss + args.depend_weight * depend_loss
            )
            (total / args.gradient_accumulation).backward()

            should_step = (position + 1) % args.gradient_accumulation == 0 or position + 1 == int(cache_index["n"])
            grad_norm = None
            if should_step:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0).detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            gates = list(diagnostics.values())
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "id": example["id"],
                "schema": example["schema"],
                "variant": example["variant"],
                "negative_type": negative_type,
                "generation_loss": float(generation_loss.detach().cpu()),
                "negative_answer_nll": float(negative_nll.detach().cpu()),
                "distill_loss": float(distill_loss.detach().cpu()),
                "depend_loss": float(depend_loss.detach().cpu()),
                "depend_margin": float((negative_nll - generation_loss).detach().cpu()),
                "gate_mean": float(np.mean(gates)) if gates else 0.0,
                "grad_norm": grad_norm,
            }
            history.append(row)
            progress.set_postfix(gen=round(row["generation_loss"], 2), dep=round(row["depend_margin"], 2))
            previous = example
            global_step += 1

        checkpoint = {
            "format_version": 2,
            "adapter": adapter.state_dict(),
            "args": vars(args),
            "memory_dim": memory_dim,
            "receiver_hidden_size": int(receiver.config.hidden_size),
            "epoch": epoch,
        }
        torch.save(checkpoint, output / f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(checkpoint, output / "checkpoint_latest.pt")
        write_jsonl(output / "train_history.jsonl", history)

    with open(output / "TRAIN_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"status": "complete", "args": vars(args), "steps": global_step, "checkpoint": str(output / "checkpoint_latest.pt")},
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
