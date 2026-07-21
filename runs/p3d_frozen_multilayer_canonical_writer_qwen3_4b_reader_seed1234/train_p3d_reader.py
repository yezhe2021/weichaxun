import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d_common import (
    INSUFFICIENT, EvidenceMemoryCache, MultiLayerEvidenceReader, compose_memory, forward_answer,
    full_text_prompt, load_receiver, memory_to, question_prompt, read_json, resize_memory,
    seed_everything, write_json, write_jsonl, zero_memory,
)


def source_spec(protocol, source, split):
    if source == "native16":
        item = protocol["native16"]
        return item["cache"][split], item["groups"], item["memory_dim"], item["original_layer_indices"]
    item = protocol[source]
    return item["canonical_cache"][split], item["groups"], item["memory_dim"], item["original_layer_indices"]


def build_cache(protocol, source, split):
    path, _, _, layers = source_spec(protocol, source, split)
    return EvidenceMemoryCache(path, source, layers)


def negative_mapping(cache):
    answers = [entry["answer"] for entry in cache.entries]
    mapping = []
    for index, answer in enumerate(answers):
        length = len(answer.split())
        candidates = [candidate for candidate, other in enumerate(answers) if candidate != index and other.casefold() != answer.casefold()]
        mapping.append(min(candidates, key=lambda candidate: (abs(len(answers[candidate].split()) - length), abs(len(answers[candidate]) - len(answer)), candidate)))
    return mapping


def condition_for(position, epoch):
    slot = (position + epoch * 3) % 10
    if slot < 5: return "correct"
    if slot < 7: return "shuffled"
    if slot == 7: return "zero"
    if slot == 8: return "kv_mismatch"
    return "reader_off"


def answer_kl(student, teacher):
    length = min(student.shape[0], teacher.shape[0])
    return F.kl_div(F.log_softmax(student[:length].float(), dim=-1), F.softmax(teacher[:length].float(), dim=-1), reduction="batchmean")


@torch.inference_mode()
def validation_loss(model, tokenizer, reader, cache, negative, device, limit, max_length):
    values = []
    for index in range(min(limit, len(cache))):
        payload, other = cache.load(index), cache.load(negative[index])
        correct = memory_to(payload, device)
        wrong = resize_memory(memory_to(other, device), correct["keys"].shape[1])
        correct_nll, _ = forward_answer(model, tokenizer, reader, payload["row"], correct, payload["row"]["answer"], max_length, device)
        wrong_nll, _ = forward_answer(model, tokenizer, reader, payload["row"], wrong, INSUFFICIENT, max_length, device)
        values.append(float(correct_nll + 0.5 * wrong_nll))
    return sum(values) / max(1, len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--protocol", required=True)
    parser.add_argument("--source", choices=("canonical16", "native16", "canonical36"), required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--mode", choices=("small", "full"), required=True)
    parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--small-samples", type=int, default=16)
    parser.add_argument("--init-checkpoint"); parser.add_argument("--rank", type=int, default=64); parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--gate-init", type=float, default=0.005); parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--distill-weight", type=float, default=0.1); parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384); parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); protocol = read_json(args.protocol)
    train_cache = build_cache(protocol, args.source, "train")
    validation_cache = train_cache if args.mode == "small" else build_cache(protocol, args.source, "validation")
    _, groups, memory_dim, _ = source_spec(protocol, args.source, "train")
    model, tokenizer = load_receiver(args.model, device)
    reader = MultiLayerEvidenceReader(model, groups, memory_dim, args.rank, args.adapter_rank, args.gate_init).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["reader_metadata"] != reader.metadata(): raise RuntimeError("Reader init checkpoint interface mismatch")
        reader.load_state_dict(checkpoint["reader"])
    if any(parameter.requires_grad for parameter in model.parameters()): raise RuntimeError("Qwen3-4B backbone must be frozen")
    trainable = [parameter for parameter in reader.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in trainable}: raise RuntimeError("Optimizer contains non-Reader parameters")
    train_limit = min(args.small_samples, len(train_cache)) if args.mode == "small" else len(train_cache)
    validation_limit = train_limit if args.mode == "small" else len(validation_cache)
    train_negative, validation_negative = negative_mapping(train_cache), negative_mapping(validation_cache)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch, update_steps = [], float("inf"), 0, 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        reader.train(); order = list(range(train_limit)); random.Random(args.seed + epoch).shuffle(order)
        pending = 0
        for position, index in enumerate(tqdm(order, desc=f"p3d_{args.source}_{args.mode}_e{epoch}")):
            payload, other = train_cache.load(index), train_cache.load(train_negative[index])
            current, wrong = memory_to(payload, device), memory_to(other, device)
            wrong = resize_memory(wrong, current["keys"].shape[1])
            condition = condition_for(position, epoch)
            target, enabled = payload["row"]["answer"], True
            if condition == "correct": memory = current
            elif condition == "shuffled": memory, target = wrong, INSUFFICIENT
            elif condition == "zero": memory, target = zero_memory(current), INSUFFICIENT
            elif condition == "kv_mismatch":
                memory = compose_memory(current, wrong) if (index + epoch) % 2 == 0 else compose_memory(wrong, current, current["keys"].shape[1])
                target = INSUFFICIENT
            else: memory, enabled = current, False
            if not enabled:
                with torch.no_grad(): loss, _ = forward_answer(model, tokenizer, reader, payload["row"], memory, target, args.max_length, device, False)
                history.append({"epoch": epoch, "index": index, "condition": condition, "loss": float(loss), "gate_mean": float(reader.gates().detach().mean())})
                continue
            loss, student_logits = forward_answer(model, tokenizer, reader, payload["row"], memory, target, args.max_length, device, True)
            distill = loss.new_tensor(0.0)
            if condition == "correct":
                with torch.no_grad(): _, teacher_logits = forward_answer(model, tokenizer, reader, payload["row"], memory, target, args.max_length, device, False, True)
                distill = answer_kl(student_logits, teacher_logits)
                loss = loss + args.distill_weight * distill
            (loss / args.gradient_accumulation).backward(); pending += 1
            if any(parameter.grad is not None for parameter in model.parameters()): raise RuntimeError("Frozen receiver backbone received gradients")
            if pending == args.gradient_accumulation or position + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); pending = 0; update_steps += 1
            history.append({"epoch": epoch, "index": index, "condition": condition, "loss": float(loss.detach()), "distill": float(distill.detach()), "gate_mean": float(reader.gates().detach().mean())})
        if pending:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            update_steps += 1
        reader.eval()
        score = validation_loss(model, tokenizer, reader, validation_cache, validation_negative, device, validation_limit, args.max_length)
        checkpoint = {"format_version": 1, "source": args.source, "mode": args.mode, "reader": {name: tensor.detach().cpu() for name, tensor in reader.state_dict().items()}, "reader_metadata": reader.metadata(), "epoch": epoch, "validation_loss": score, "protocol": protocol, "only_reader_trainable": True, "args": vars(args)}
        torch.save(checkpoint, output / "checkpoint_latest.pt")
        if score < best: best, best_epoch = score, epoch; torch.save(checkpoint, output / "checkpoint_best.pt")
        write_jsonl(output / "history.jsonl", history)
    write_json(output / "TRAIN_SUCCESS.json", {"status": "complete", "source": args.source, "mode": args.mode, "best_epoch": best_epoch, "validation_loss": best, "reader_parameters": sum(p.numel() for p in reader.parameters()), "backbone_parameters_updated": 0, "optimizer_reader_only": True, "update_steps": update_steps, "initial_gate": args.gate_init, "final_gate_mean": float(reader.gates().detach().mean())})


if __name__ == "__main__": main()
