import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import (LayerAlignedNativeQueryReader, MemoryCache, answer_scores, forward_answer, generate,
                         hard_negative_mapping, load_receiver, memory_to, read_json, seed_everything, write_json, write_jsonl)


def check_trainable(model, reader, optimizer):
    if any(parameter.requires_grad for parameter in model.parameters()): raise RuntimeError("Receiver backbone is not frozen")
    expected = {id(parameter) for parameter in reader.parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual: raise RuntimeError("Optimizer does not contain exactly Reader parameters")


@torch.inference_mode()
def training_monitor(model, tokenizer, reader, cache, negatives, indices, device, max_new_tokens):
    rows = []
    for index in indices:
        payload, wrong = cache.load(index), cache.load(negatives[index])
        correct_output = generate(model, tokenizer, reader, payload["row"], memory_to(payload, device), max_new_tokens)
        wrong_output = generate(model, tokenizer, reader, payload["row"], memory_to(wrong, device), max_new_tokens)
        _, correct_f1 = answer_scores(correct_output["prediction"], payload["row"]["answer"])
        _, wrong_f1 = answer_scores(wrong_output["prediction"], payload["row"]["answer"])
        rows.append({"id": payload["row"]["id"], "correct": correct_output, "shuffled": wrong_output, "correct_f1": correct_f1, "shuffled_current_f1": wrong_f1})
    correct = sum(row["correct_f1"] for row in rows) / len(rows); shuffled = sum(row["shuffled_current_f1"] for row in rows) / len(rows)
    return {"correct_f1": correct, "shuffled_current_f1": shuffled, "gap": correct - shuffled}, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--branch", choices=["canonical16", "native_projected16"], required=True)
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--lr", type=float, default=2e-4); parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=32); parser.add_argument("--gate-init", type=float, default=0.01)
    parser.add_argument("--depend-weight", type=float, default=0.5); parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=512); parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--monitor-samples", type=int, default=8); parser.add_argument("--monitor-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    cache = MemoryCache(args.memory); index = read_json(args.memory)
    if index["original_layer_indices"] != list(index["original_layer_indices"]): raise RuntimeError("Invalid layer order")
    negatives = hard_negative_mapping(cache)
    model, tokenizer = load_receiver(args.model, device)
    reader = LayerAlignedNativeQueryReader(model, index["memory_dim"], index["original_layer_indices"], args.rank, args.gate_init).to(device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=args.weight_decay); check_trainable(model, reader, optimizer)
    order = list(range(len(cache))); monitor_indices = random.Random(args.seed + 91).sample(order, min(args.monitor_samples, len(order)))
    history, best_loss, best_state, best_epoch = [], float("inf"), None, -1
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(order); reader.train(); losses, correct_nlls, wrong_nlls = [], [], []
        for sample_index in tqdm(order, desc=f"p3d3_{args.branch}_epoch{epoch}"):
            payload, wrong_payload = cache.load(sample_index), cache.load(negatives[sample_index])
            correct = memory_to(payload, device); wrong = memory_to(wrong_payload, device)
            optimizer.zero_grad(set_to_none=True)
            correct_nll = forward_answer(model, tokenizer, reader, payload["row"], correct, args.max_length, device)
            wrong_nll = forward_answer(model, tokenizer, reader, payload["row"], wrong, args.max_length, device)
            depend = F.relu(args.margin + correct_nll - wrong_nll)
            loss = correct_nll + args.depend_weight * depend; loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0); optimizer.step()
            if any(parameter.grad is not None for parameter in model.parameters()): raise RuntimeError("Gradient reached frozen Receiver")
            losses.append(float(loss.detach())); correct_nlls.append(float(correct_nll.detach())); wrong_nlls.append(float(wrong_nll.detach()))
        epoch_loss = sum(losses) / len(losses); record = {"epoch": epoch, "train_loss": epoch_loss,
            "correct_answer_mean_nll": sum(correct_nlls) / len(correct_nlls), "shuffled_answer_mean_nll": sum(wrong_nlls) / len(wrong_nlls),
            "gates": reader.gates().detach().cpu().tolist()}
        if epoch % args.monitor_every == 0 or epoch == args.epochs:
            reader.eval(); metrics, rows = training_monitor(model, tokenizer, reader, cache, negatives, monitor_indices, device, args.max_new_tokens)
            record["train_free_running_monitor"] = metrics; write_jsonl(output / f"monitor_epoch_{epoch:03d}.jsonl", rows)
        history.append(record); write_jsonl(output / "training_history.jsonl", history)
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch; best_state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in reader.state_dict().items()})
            torch.save({"reader": best_state, "reader_metadata": reader.metadata(), "branch": args.branch, "epoch": epoch,
                        "train_loss": best_loss, "memory_index": args.memory, "args": vars(args), "receiver_backbone_frozen": True}, output / "checkpoint_best.pt")
    result = {"status": "complete", "branch": args.branch, "best_epoch_by_train_objective": best_epoch, "best_train_loss": best_loss,
              "validation_used_for_training_or_selection": False, "answer_loss": "answer_tokens_only_mean_nll", "hard_negative_leakage_filters": True,
              "reader_metadata": reader.metadata(), "trainable_parameters": sum(parameter.numel() for parameter in reader.parameters()),
              "receiver_backbone_frozen": True, "checkpoint": str(output / "checkpoint_best.pt")}
    write_json(output / "TRAIN_SUCCESS.json", result)


if __name__ == "__main__": main()
