import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import answer_scores, forward_answer, generate, load_receiver, seed_everything
from p3e_c1_common import LearnableCanonicalHeadReader
from p3e_f_common import CanonicalCache, memory_to, read_json, sha256_file, write_json, write_jsonl


def audit(model, reader, optimizer):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Receiver backbone is not frozen")
    expected = {id(parameter) for parameter in reader.parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual:
        raise RuntimeError("Optimizer must contain exactly Reader parameters")


@torch.inference_mode()
def monitor(model, tokenizer, reader, cache, negatives, indices, device, max_new_tokens):
    rows = []
    for index in indices:
        payload = cache.load(index)
        wrong = cache.load(negatives[index])
        correct = generate(model, tokenizer, reader, payload["row"], memory_to(payload, device), max_new_tokens)
        shuffled = generate(model, tokenizer, reader, payload["row"], memory_to(wrong, device), max_new_tokens)
        correct_em, correct_f1 = answer_scores(correct["prediction"], payload["row"]["answer"])
        _, shuffled_f1 = answer_scores(shuffled["prediction"], payload["row"]["answer"])
        rows.append({"id": payload["row"]["id"], "correct": correct, "shuffled": shuffled,
                     "correct_em": correct_em, "correct_f1": correct_f1,
                     "shuffled_current_f1": shuffled_f1})
    return {
        "correct_em": sum(row["correct_em"] for row in rows) / len(rows),
        "correct_f1": sum(row["correct_f1"] for row in rows) / len(rows),
        "shuffled_current_f1": sum(row["shuffled_current_f1"] for row in rows) / len(rows),
        "correct_shuffled_gap": sum(row["correct_f1"] - row["shuffled_current_f1"] for row in rows) / len(rows),
    }, rows


def cpu_state(module):
    return copy.deepcopy({name: tensor.detach().cpu() for name, tensor in module.state_dict().items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--init-reader", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=int, choices=[1024, 2048], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--depend-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--monitor-samples", type=int, default=8)
    parser.add_argument("--monitor-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = CanonicalCache(args.memory_index, args.data, capacity=2)
    if len(cache) < args.scale:
        raise RuntimeError("Canonical cache is smaller than requested scale")
    negatives = read_json(args.negatives)[f"train{args.scale}"]
    if len(negatives) != args.scale or any(index >= args.scale for index in negatives):
        raise RuntimeError("Hard-negative mapping violates the scale prefix")

    model, tokenizer = load_receiver(args.model, device)
    checkpoint = torch.load(args.init_reader, map_location="cpu", weights_only=False)
    metadata = checkpoint["reader_metadata"]
    reader = LearnableCanonicalHeadReader(
        model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"],
        metadata["top_k"], args.temperature_start
    ).to(device)
    reader.load_state_dict(checkpoint["reader"])
    init_state_hash = sha256_file(args.init_reader)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    audit(model, reader, optimizer)
    indices = list(range(args.scale))
    monitor_indices = random.Random(args.seed + 288).sample(indices, min(args.monitor_samples, args.scale))
    history = []
    best_loss = float("inf")
    best_epoch = -1
    start_epoch = 1
    last_path = output / "checkpoint_last.pt"
    if last_path.exists():
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume["scale"] != args.scale or resume["init_reader_sha256"] != init_state_hash:
            raise RuntimeError("Resume checkpoint does not match this scale/initialization")
        reader.load_state_dict(resume["reader"])
        optimizer.load_state_dict(resume["optimizer"])
        history = resume["history"]
        best_loss = resume["best_loss"]
        best_epoch = resume["best_epoch"]
        start_epoch = resume["epoch"] + 1

    optimizer_steps = (start_epoch - 1) * args.scale
    for epoch in range(start_epoch, args.epochs + 1):
        ratio = 0.0 if args.epochs == 1 else (epoch - 1) / (args.epochs - 1)
        temperature = args.temperature_start + ratio * (args.temperature_end - args.temperature_start)
        reader.set_temperature(temperature)
        order = indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        reader.train()
        losses, correct_values, shuffled_values, active_margin = [], [], [], []
        for index in tqdm(order, desc=f"p3e_f_train{args.scale}_epoch{epoch}"):
            payload = cache.load(index)
            wrong = cache.load(negatives[index])
            optimizer.zero_grad(set_to_none=True)
            correct_nll = forward_answer(model, tokenizer, reader, payload["row"],
                                         memory_to(payload, device), args.max_length, device)
            shuffled_nll = forward_answer(model, tokenizer, reader, payload["row"],
                                          memory_to(wrong, device), args.max_length, device)
            dependency = F.relu(args.margin + correct_nll - shuffled_nll)
            loss = correct_nll + args.depend_weight * dependency
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Gradient reached frozen Receiver backbone")
            losses.append(float(loss.detach()))
            correct_values.append(float(correct_nll.detach()))
            shuffled_values.append(float(shuffled_nll.detach()))
            active_margin.append(float(dependency.detach() > 0))
        epoch_loss = sum(losses) / len(losses)
        reader.eval()
        routes = reader.routes().detach().cpu()
        record = {
            "epoch": epoch, "scale": args.scale, "optimizer_steps": optimizer_steps,
            "train_loss": epoch_loss,
            "correct_answer_mean_nll": sum(correct_values) / len(correct_values),
            "shuffled_answer_mean_nll": sum(shuffled_values) / len(shuffled_values),
            "active_margin_fraction": sum(active_margin) / len(active_margin),
            "temperature": temperature, "gates": reader.gates().detach().cpu().tolist(),
            "selected_top2": routes.topk(2, dim=-1).indices.tolist(),
            "canonical_head_usage": routes.mean(dim=1).tolist(),
        }
        if epoch % args.monitor_every == 0 or epoch == args.epochs:
            metrics, rows = monitor(model, tokenizer, reader, cache, negatives, monitor_indices,
                                    device, args.max_new_tokens)
            record["train_free_running_monitor"] = metrics
            write_jsonl(output / f"monitor_epoch_{epoch:03d}.jsonl", rows)
        history.append(record)
        write_jsonl(output / "training_history.jsonl", history)
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            torch.save({
                "reader": cpu_state(reader), "reader_metadata": reader.metadata(), "scale": args.scale,
                "epoch": epoch, "train_loss": best_loss, "args": vars(args),
                "init_reader": args.init_reader, "init_reader_sha256": init_state_hash,
                "independent_scale_training": True, "receiver_backbone_frozen": True,
                "selection_rule": "minimum_predefined_training_objective",
            }, output / "checkpoint_best.pt")
        torch.save({
            "reader": cpu_state(reader), "optimizer": optimizer.state_dict(), "history": history,
            "scale": args.scale, "epoch": epoch, "best_loss": best_loss, "best_epoch": best_epoch,
            "init_reader_sha256": init_state_hash,
        }, last_path)

    write_json(output / "TRAIN_SUCCESS.json", {
        "status": "complete", "experiment": "P3-E-F Reader scale study",
        "scale": args.scale, "epochs": args.epochs, "optimizer_steps": optimizer_steps,
        "best_epoch_by_training_objective": best_epoch, "best_train_loss": best_loss,
        "init_reader": args.init_reader, "init_reader_sha256": init_state_hash,
        "validation_used_for_selection": False, "warmup_ratio": 0.0,
        "constant_learning_rate": args.lr, "batch_size": 1,
        "checkpoint": str(output / "checkpoint_best.pt"),
    })


if __name__ == "__main__":
    main()
