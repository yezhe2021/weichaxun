import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import answer_scores, file_sha256, forward_answer, generate, hard_negative_mapping, load_receiver, seed_everything, write_json, write_jsonl
from p3e_c1_common import LearnableCanonicalHeadReader
from p3e_c2_common import SenderNativeHeadwiseCache, load_writer, writer_memory


def audit_optimizer(model, writer, reader, optimizer):
    if any(parameter.requires_grad for parameter in model.parameters()): raise RuntimeError("Receiver backbone is not frozen")
    if any(parameter.requires_grad for parameter in writer.parameters()): raise RuntimeError("C2 Writer is not frozen")
    expected = {id(parameter) for parameter in reader.parameters() if parameter.requires_grad}; actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual: raise RuntimeError("Optimizer must contain exactly fresh Reader parameters")


@torch.inference_mode()
def monitor(model, tokenizer, reader, writer, cache, negatives, indices, device, max_new_tokens):
    rows = []
    for index in indices:
        payload, wrong = cache.load(index), cache.load(negatives[index])
        correct = generate(model, tokenizer, reader, payload["row"], writer_memory(writer, payload, device, no_grad=True), max_new_tokens)
        shuffled = generate(model, tokenizer, reader, payload["row"], writer_memory(writer, wrong, device, no_grad=True), max_new_tokens)
        correct_em, correct_f1 = answer_scores(correct["prediction"], payload["row"]["answer"]); _, shuffled_f1 = answer_scores(shuffled["prediction"], payload["row"]["answer"])
        rows.append({"id": payload["row"]["id"], "correct": correct, "shuffled": shuffled, "correct_em": correct_em, "correct_f1": correct_f1, "shuffled_current_f1": shuffled_f1})
    return {"correct_em": sum(row["correct_em"] for row in rows) / len(rows), "correct_f1": sum(row["correct_f1"] for row in rows) / len(rows),
            "shuffled_current_f1": sum(row["shuffled_current_f1"] for row in rows) / len(rows),
            "correct_shuffled_gap": sum(row["correct_f1"] - row["shuffled_current_f1"] for row in rows) / len(rows)}, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--writer", required=True); parser.add_argument("--native-reader", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["overfit16", "formal512"], required=True); parser.add_argument("--max-samples", type=int); parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4); parser.add_argument("--weight-decay", type=float, default=0.01); parser.add_argument("--rank", type=int, default=32); parser.add_argument("--gate-init", type=float, default=0.01); parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--depend-weight", type=float, default=0.5); parser.add_argument("--margin", type=float, default=0.5); parser.add_argument("--temperature-start", type=float, default=1.0); parser.add_argument("--temperature-end", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=512); parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--monitor-samples", type=int, default=8); parser.add_argument("--monitor-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2345); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    cache = SenderNativeHeadwiseCache(args.memory); total = min(args.max_samples or len(cache), len(cache)); indices = list(range(total)); negatives = hard_negative_mapping(cache)
    model, tokenizer = load_receiver(args.model, device); model.requires_grad_(False); model.eval()
    writer, _ = load_writer(args.writer, device); writer.requires_grad_(False); writer.eval()
    native_checkpoint = torch.load(args.native_reader, map_location="cpu", weights_only=False); metadata = native_checkpoint["reader_metadata"]
    reader = LearnableCanonicalHeadReader(model, metadata["selected_layers"], args.rank, args.gate_init, args.top_k, args.temperature_start).to(device)
    copied = reader.load_native_reader(native_checkpoint)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=args.weight_decay); audit_optimizer(model, writer, reader, optimizer)
    monitor_indices = random.Random(args.seed + 288).sample(indices, min(args.monitor_samples, total)); history, best_loss, best_epoch = [], float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        ratio = 0.0 if args.epochs == 1 else (epoch - 1) / (args.epochs - 1); reader.set_temperature(args.temperature_start + ratio * (args.temperature_end - args.temperature_start))
        order = indices.copy(); random.Random(args.seed + epoch).shuffle(order); reader.train(); losses, correct_values, wrong_values = [], [], []
        for sample_index in tqdm(order, desc=f"p3e_c2_fresh_reader_{args.mode}_epoch{epoch}"):
            payload, wrong_payload = cache.load(sample_index), cache.load(negatives[sample_index]); optimizer.zero_grad(set_to_none=True)
            correct_nll = forward_answer(model, tokenizer, reader, payload["row"], writer_memory(writer, payload, device, no_grad=True), args.max_length, device)
            wrong_nll = forward_answer(model, tokenizer, reader, payload["row"], writer_memory(writer, wrong_payload, device, no_grad=True), args.max_length, device)
            dependency = F.relu(args.margin + correct_nll - wrong_nll); loss = correct_nll + args.depend_weight * dependency
            loss.backward(); torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0); optimizer.step()
            if any(parameter.grad is not None for parameter in model.parameters()): raise RuntimeError("Gradient reached Receiver backbone")
            if any(parameter.grad is not None for parameter in writer.parameters()): raise RuntimeError("Gradient reached frozen Writer")
            losses.append(float(loss.detach())); correct_values.append(float(correct_nll.detach())); wrong_values.append(float(wrong_nll.detach()))
        epoch_loss = sum(losses) / len(losses); reader.eval(); routes = reader.routes().detach().cpu()
        record = {"epoch": epoch, "train_loss": epoch_loss, "correct_answer_mean_nll": sum(correct_values) / len(correct_values), "shuffled_answer_mean_nll": sum(wrong_values) / len(wrong_values),
                  "gates": reader.gates().detach().cpu().tolist(), "selected_top2": routes.topk(2, dim=-1).indices.tolist(), "canonical_head_usage": routes.mean(dim=1).tolist()}
        if epoch % args.monitor_every == 0 or epoch == args.epochs:
            metrics, rows = monitor(model, tokenizer, reader, writer, cache, negatives, monitor_indices, device, args.max_new_tokens)
            record["train_free_running_monitor"] = metrics; write_jsonl(output / f"monitor_epoch_{epoch:03d}.jsonl", rows)
        history.append(record); write_jsonl(output / "training_history.jsonl", history)
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch; state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in reader.state_dict().items()})
            torch.save({"reader": state, "reader_metadata": reader.metadata(), "mode": args.mode, "epoch": epoch, "train_loss": best_loss, "args": vars(args),
                        "fresh_reader": True, "old_c1_reader_loaded": False, "native_reader_warm_start": args.native_reader,
                        "native_reader_sha256": file_sha256(args.native_reader), "warm_start_tensors": copied,
                        "writer": args.writer, "writer_sha256": file_sha256(args.writer), "writer_frozen": True, "receiver_backbone_frozen": True}, output / "checkpoint_best.pt")
    write_json(output / "TRAIN_SUCCESS.json", {"status": "complete", "experiment": "P3-E-C2 fresh Reader on frozen learned Writer", "mode": args.mode, "samples": total,
        "best_epoch_by_train_objective": best_epoch, "best_train_loss": best_loss, "fresh_reader": True, "writer_frozen": True,
        "trainable_parameters": sum(parameter.numel() for parameter in reader.parameters()), "checkpoint": str(output / "checkpoint_best.pt")})


if __name__ == "__main__": main()
