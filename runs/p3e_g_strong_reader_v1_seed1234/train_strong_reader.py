import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import answer_scores, forward_answer, generate, load_receiver, seed_everything
from p3e_f_common import CanonicalCache, memory_to, read_json, sha256_file, write_json, write_jsonl
from p3e_g_common import StrongCanonicalReader, assert_frozen_gradients, assert_trainable_boundary


def cpu_state(module):
    return copy.deepcopy({name: tensor.detach().cpu() for name, tensor in module.state_dict().items()})


@torch.inference_mode()
def smoke_eval(model, tokenizer, reader, cache, negatives, indices, device, max_new_tokens):
    rows = []
    for index in indices:
        payload, wrong = cache.load(index), cache.load(negatives[index])
        correct = generate(model, tokenizer, reader, payload["row"], memory_to(payload, device), max_new_tokens)
        shuffled = generate(model, tokenizer, reader, payload["row"], memory_to(wrong, device), max_new_tokens)
        reader_off = generate(model, tokenizer, reader, payload["row"], memory_to(payload, device), max_new_tokens, enabled=False)
        correct_em, correct_f1 = answer_scores(correct["prediction"], payload["row"]["answer"])
        _, shuffled_f1 = answer_scores(shuffled["prediction"], payload["row"]["answer"])
        rows.append({"id": payload["row"]["id"], "correct": correct, "shuffled": shuffled,
                     "reader_off": reader_off, "correct_em": correct_em,
                     "correct_f1": correct_f1, "shuffled_current_f1": shuffled_f1})
    return {
        "correct_em": sum(row["correct_em"] for row in rows) / len(rows),
        "correct_f1": sum(row["correct_f1"] for row in rows) / len(rows),
        "shuffled_current_f1": sum(row["shuffled_current_f1"] for row in rows) / len(rows),
        "correct_shuffled_gap": sum(row["correct_f1"] - row["shuffled_current_f1"] for row in rows) / len(rows),
    }, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--base-reader", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["smoke16", "formal512"], required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output-rank", type=int, default=128)
    parser.add_argument("--depend-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = CanonicalCache(args.memory_index, args.data, capacity=2)
    negatives = read_json(args.negatives)["train512"]
    count = min(args.max_samples, 512)
    indices = list(range(count))
    model, tokenizer = load_receiver(args.model, device)
    old_checkpoint = torch.load(args.base_reader, map_location="cpu", weights_only=False)
    reader = StrongCanonicalReader(model, old_checkpoint, args.output_rank).to(device)
    reader.set_temperature(0.25)
    initial_equivalence_error = reader.initial_equivalence_error()
    if initial_equivalence_error > 1e-6:
        raise RuntimeError("Strong Reader initialization does not reproduce old scalar gates")
    optimizer = torch.optim.AdamW(reader.new_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    assert_trainable_boundary(model, reader, optimizer)
    initial_state = cpu_state(reader)
    history, best_loss, best_epoch = [], float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        order = indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        reader.train()
        losses, correct_values, shuffled_values = [], [], []
        gradient_norms = {"adapter_up": [], "gate_linear": []}
        for index in tqdm(order, desc=f"p3e_g_{args.mode}_epoch{epoch}"):
            payload, wrong = cache.load(index), cache.load(negatives[index])
            optimizer.zero_grad(set_to_none=True)
            correct_nll = forward_answer(model, tokenizer, reader, payload["row"],
                                         memory_to(payload, device), args.max_length, device)
            shuffled_nll = forward_answer(model, tokenizer, reader, payload["row"],
                                          memory_to(wrong, device), args.max_length, device)
            dependency = F.relu(args.margin + correct_nll - shuffled_nll)
            loss = correct_nll + args.depend_weight * dependency
            loss.backward()
            assert_frozen_gradients(model, reader)
            gradient_norms["adapter_up"].append(sum(
                float(control.adapter_up.weight.grad.float().norm())
                for control in reader.controls if control.adapter_up.weight.grad is not None
            ))
            gradient_norms["gate_linear"].append(sum(
                float(control.gate_linear.weight.grad.float().norm())
                for control in reader.controls if control.gate_linear.weight.grad is not None
            ))
            torch.nn.utils.clip_grad_norm_(reader.new_parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            correct_values.append(float(correct_nll.detach()))
            shuffled_values.append(float(shuffled_nll.detach()))
        epoch_loss = sum(losses) / len(losses)
        record = {
            "epoch": epoch, "train_loss": epoch_loss,
            "correct_answer_mean_nll": sum(correct_values) / len(correct_values),
            "shuffled_answer_mean_nll": sum(shuffled_values) / len(shuffled_values),
            "mean_adapter_up_gradient_norm": sum(gradient_norms["adapter_up"]) / len(gradient_norms["adapter_up"]),
            "mean_gate_linear_gradient_norm": sum(gradient_norms["gate_linear"]) / len(gradient_norms["gate_linear"]),
            "gate_bias_sigmoid": reader.static_gate_values().detach().cpu().tolist(),
        }
        history.append(record)
        write_jsonl(output / "training_history.jsonl", history)
        state = {
            "reader": cpu_state(reader), "reader_metadata": reader.metadata(), "mode": args.mode,
            "epoch": epoch, "train_loss": epoch_loss, "args": vars(args),
            "base_reader": args.base_reader, "base_reader_sha256": sha256_file(args.base_reader),
            "receiver_backbone_frozen": True, "base_reader_frozen": True,
            "optimizer_parameter_ids_verified": True,
        }
        torch.save(state, output / "checkpoint_last.pt")
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            torch.save(state, output / "checkpoint_best.pt")

    smoke_metrics = None
    if args.mode == "smoke16":
        reader.eval()
        smoke_metrics, rows = smoke_eval(
            model, tokenizer, reader, cache, negatives, indices, device, args.max_new_tokens
        )
        write_jsonl(output / "smoke_generations.jsonl", rows)
    changed = 0.0
    final_state = reader.state_dict()
    for name, tensor in initial_state.items():
        if name.startswith("controls."):
            changed += float((final_state[name].detach().cpu() - tensor).float().pow(2).sum())
    result = {
        "status": "complete", "experiment": "P3-E-G Strong Reader V1",
        "mode": args.mode, "samples": count, "epochs": args.epochs,
        "best_epoch_by_training_loss": best_epoch, "best_train_loss": best_loss,
        "new_parameter_squared_change": changed, "smoke_free_running": smoke_metrics,
        "trainable_parameters": sum(parameter.numel() for parameter in reader.new_parameters()),
        "frozen_base_parameters": sum(parameter.numel() for parameter in reader.base.parameters()),
        "initial_equivalence_error": initial_equivalence_error,
        "checkpoint_best": str(output / "checkpoint_best.pt"),
        "checkpoint_last": str(output / "checkpoint_last.pt"),
    }
    write_json(output / "TRAIN_SUCCESS.json", result)


if __name__ == "__main__":
    main()
