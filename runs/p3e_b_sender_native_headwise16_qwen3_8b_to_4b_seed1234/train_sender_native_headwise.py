import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import answer_scores, forward_answer, generate, hard_negative_mapping, load_receiver, seed_everything, write_json, write_jsonl
from p3e_b_common import NativeHeadwiseReader, SenderNativeHeadwiseCache, native_memory_to


def audit_optimizer(model, reader, optimizer):
    if any(parameter.requires_grad for parameter in model.parameters()): raise RuntimeError("Qwen3-4B Receiver backbone is not frozen")
    expected = {id(parameter) for parameter in reader.parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual: raise RuntimeError("Optimizer must contain exactly Query adapters and scalar gates")


@torch.inference_mode()
def monitor(model, tokenizer, reader, cache, negatives, indices, device, max_new_tokens):
    rows = []
    for index in indices:
        payload, wrong = cache.load(index), cache.load(negatives[index])
        correct = generate(model, tokenizer, reader, payload["row"], native_memory_to(payload, device), max_new_tokens)
        shuffled = generate(model, tokenizer, reader, payload["row"], native_memory_to(wrong, device), max_new_tokens)
        correct_em, correct_f1 = answer_scores(correct["prediction"], payload["row"]["answer"])
        _, shuffled_f1 = answer_scores(shuffled["prediction"], payload["row"]["answer"])
        rows.append({"id": payload["row"]["id"], "answer": payload["row"]["answer"], "correct": correct, "shuffled": shuffled,
                     "correct_em": correct_em, "correct_f1": correct_f1, "shuffled_current_f1": shuffled_f1})
    return {"correct_em": sum(row["correct_em"] for row in rows) / len(rows), "correct_f1": sum(row["correct_f1"] for row in rows) / len(rows),
            "shuffled_current_f1": sum(row["shuffled_current_f1"] for row in rows) / len(rows),
            "correct_shuffled_gap": sum(row["correct_f1"] - row["shuffled_current_f1"] for row in rows) / len(rows)}, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["overfit16", "formal512"], required=True); parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--lr", type=float, default=2e-4); parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=32); parser.add_argument("--gate-init", type=float, default=0.01)
    parser.add_argument("--depend-weight", type=float, default=0.5); parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=512); parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--monitor-samples", type=int, default=8); parser.add_argument("--monitor-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    cache = SenderNativeHeadwiseCache(args.memory); total = min(args.max_samples or len(cache), len(cache)); indices = list(range(total)); negatives = hard_negative_mapping(cache)
    model, tokenizer = load_receiver(args.model, device); reader = NativeHeadwiseReader(model, cache.inner.index["original_layer_indices"], args.rank, args.gate_init).to(device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=args.weight_decay); audit_optimizer(model, reader, optimizer)
    monitor_indices = random.Random(args.seed + 88).sample(indices, min(args.monitor_samples, total)); history, best_loss, best_epoch = [], float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        order = indices.copy(); random.Random(args.seed + epoch).shuffle(order); reader.train(); losses, correct_values, wrong_values = [], [], []
        for sample_index in tqdm(order, desc=f"p3e_b_{args.mode}_epoch{epoch}"):
            payload, wrong_payload = cache.load(sample_index), cache.load(negatives[sample_index]); optimizer.zero_grad(set_to_none=True)
            correct_nll = forward_answer(model, tokenizer, reader, payload["row"], native_memory_to(payload, device), args.max_length, device)
            wrong_nll = forward_answer(model, tokenizer, reader, payload["row"], native_memory_to(wrong_payload, device), args.max_length, device)
            dependency = F.relu(args.margin + correct_nll - wrong_nll); loss = correct_nll + args.depend_weight * dependency
            loss.backward(); torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0); optimizer.step()
            if any(parameter.grad is not None for parameter in model.parameters()): raise RuntimeError("Gradient reached frozen Qwen3-4B")
            losses.append(float(loss.detach())); correct_values.append(float(correct_nll.detach())); wrong_values.append(float(wrong_nll.detach()))
        epoch_loss = sum(losses) / len(losses); record = {"epoch": epoch, "train_loss": epoch_loss,
            "correct_answer_mean_nll": sum(correct_values) / len(correct_values), "shuffled_answer_mean_nll": sum(wrong_values) / len(wrong_values),
            "gates": reader.gates().detach().cpu().tolist()}
        if epoch % args.monitor_every == 0 or epoch == args.epochs:
            reader.eval(); metrics, rows = monitor(model, tokenizer, reader, cache, negatives, monitor_indices, device, args.max_new_tokens)
            record["train_free_running_monitor"] = metrics; write_jsonl(output / f"monitor_epoch_{epoch:03d}.jsonl", rows)
        history.append(record); write_jsonl(output / "training_history.jsonl", history)
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch; state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in reader.state_dict().items()})
            torch.save({"reader": state, "reader_metadata": reader.metadata(), "mode": args.mode, "epoch": epoch, "train_loss": best_loss,
                        "memory_index": args.memory, "args": vars(args), "receiver_backbone_frozen": True, "sender_model": "Qwen3-8B",
                        "writer_loaded": False, "canonical_projection_used": False}, output / "checkpoint_best.pt")
    write_json(output / "TRAIN_SUCCESS.json", {"status": "complete", "experiment": "P3-E-B sender_native_headwise16 Qwen3-8B to Qwen3-4B", "mode": args.mode,
        "samples": total, "best_epoch_by_train_objective": best_epoch, "best_train_loss": best_loss, "validation_used_for_selection": False,
        "loss": "answer-token mean NLL + 0.5 * shuffled margin", "reader_metadata": reader.metadata(),
        "trainable_parameters": sum(parameter.numel() for parameter in reader.parameters()), "only_query_adapters_and_gates_trainable": True,
        "checkpoint": str(output / "checkpoint_best.pt")})


if __name__ == "__main__": main()
