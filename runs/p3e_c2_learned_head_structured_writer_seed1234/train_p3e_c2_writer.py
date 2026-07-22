import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import answer_mean_nll, answer_scores, file_sha256, generate, hard_negative_mapping, load_receiver, pack_answer, seed_everything, write_json, write_jsonl
from p3e_b_common import NativeHeadwiseReader, native_memory_to
from p3e_c1_common import LearnableCanonicalHeadReader
from p3e_c2_common import HeadStructuredWriter, SenderNativeHeadwiseCache, head_diversity_loss, writer_memory


def frozen(module):
    module.requires_grad_(False); module.eval(); return module


def audit_optimizer(model, native_reader, canonical_reader, writer, optimizer):
    for name, module in (("receiver", model), ("native teacher", native_reader), ("C1 reader", canonical_reader)):
        if any(parameter.requires_grad for parameter in module.parameters()): raise RuntimeError(f"{name} is not frozen")
    expected = {id(parameter) for parameter in writer.parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual: raise RuntimeError("Optimizer must contain exactly Writer parameters")


def traced_answer(model, tokenizer, reader, row, memory, max_length, device):
    ids, mask, labels = pack_answer(tokenizer, row, row["answer"], max_length, device); trace = {}
    with reader.inject(model, memory, trace): output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    return answer_mean_nll(output.logits, labels), trace


def functional_alignment(native_reader, canonical_reader, native_trace, canonical_trace):
    route_losses, readout_losses = [], []
    for local, layer in enumerate(canonical_reader.selected_layers):
        teacher = native_trace[layer][0]; student = canonical_trace[layer][0]
        native_attention = teacher["attention"].float().reshape(teacher["attention"].shape[0], teacher["attention"].shape[1], 32, -1)
        effective_attention = torch.einsum("qc,bsqct->bsqt", student["route"].float(), student["attention"].float())
        native_attention = native_attention.clamp_min(1e-8); effective_attention = effective_attention.clamp_min(1e-8)
        route_losses.append((native_attention * (native_attention.log() - effective_attention.log())).sum(-1).mean())
        gate = native_reader.branches[local].gate.detach().float().abs().clamp_min(1e-4)
        native_projected = teacher["delta"].float() / gate
        canonical_projected = student["projected"].float()
        cosine = 1.0 - F.cosine_similarity(canonical_projected, native_projected, dim=-1).mean()
        norm_error = (canonical_projected.norm(dim=-1).clamp_min(1e-6).log() - native_projected.norm(dim=-1).clamp_min(1e-6).log()).square().mean()
        readout_losses.append(cosine + 0.1 * norm_error)
    return torch.stack(route_losses).mean(), torch.stack(readout_losses).mean()


@torch.inference_mode()
def monitor(model, tokenizer, reader, writer, cache, negatives, indices, device, max_new_tokens):
    writer.eval(); reader.eval(); rows = []
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
    parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--native-reader", required=True); parser.add_argument("--canonical-reader", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["overfit16", "formal512"], required=True); parser.add_argument("--max-samples", type=int); parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=0.01); parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--depend-weight", type=float, default=0.5); parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--route-weight", type=float, default=0.05); parser.add_argument("--readout-weight", type=float, default=0.10); parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--temperature-start", type=float, default=1.0); parser.add_argument("--temperature-end", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=512); parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--monitor-samples", type=int, default=8); parser.add_argument("--monitor-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    cache = SenderNativeHeadwiseCache(args.memory); total = min(args.max_samples or len(cache), len(cache)); indices = list(range(total)); negatives = hard_negative_mapping(cache)
    model, tokenizer = load_receiver(args.model, device); frozen(model)
    native_checkpoint = torch.load(args.native_reader, map_location="cpu", weights_only=False); native_meta = native_checkpoint["reader_metadata"]
    native_reader = NativeHeadwiseReader(model, native_meta["selected_layers"], native_meta["rank"], native_meta["gate_init"]).to(device); native_reader.load_state_dict(native_checkpoint["reader"]); frozen(native_reader)
    canonical_checkpoint = torch.load(args.canonical_reader, map_location="cpu", weights_only=False); canonical_meta = canonical_checkpoint["reader_metadata"]
    canonical_reader = LearnableCanonicalHeadReader(model, canonical_meta["selected_layers"], canonical_meta["rank"], canonical_meta["gate_init"], canonical_meta["top_k"], 0.25).to(device)
    canonical_reader.load_state_dict(canonical_checkpoint["reader"]); frozen(canonical_reader)
    writer = HeadStructuredWriter(layers=len(canonical_meta["selected_layers"]), rank=args.rank, temperature=args.temperature_start).to(device)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr, weight_decay=args.weight_decay); audit_optimizer(model, native_reader, canonical_reader, writer, optimizer)
    monitor_indices = random.Random(args.seed + 188).sample(indices, min(args.monitor_samples, total)); history, best_loss, best_epoch = [], float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        ratio = 0.0 if args.epochs == 1 else (epoch - 1) / (args.epochs - 1); writer.temperature = args.temperature_start + ratio * (args.temperature_end - args.temperature_start)
        order = indices.copy(); random.Random(args.seed + epoch).shuffle(order); writer.train(); totals = {name: [] for name in ("loss", "correct", "wrong", "dependency", "route", "readout", "diversity")}
        for sample_index in tqdm(order, desc=f"p3e_c2_writer_{args.mode}_epoch{epoch}"):
            payload, wrong_payload = cache.load(sample_index), cache.load(negatives[sample_index]); optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): _, native_trace = traced_answer(model, tokenizer, native_reader, payload["row"], native_memory_to(payload, device), args.max_length, device)
            correct_memory = writer_memory(writer, payload, device); correct_nll, canonical_trace = traced_answer(model, tokenizer, canonical_reader, payload["row"], correct_memory, args.max_length, device)
            wrong_nll, _ = traced_answer(model, tokenizer, canonical_reader, payload["row"], writer_memory(writer, wrong_payload, device), args.max_length, device)
            dependency = F.relu(args.margin + correct_nll - wrong_nll); route_loss, readout_loss = functional_alignment(native_reader, canonical_reader, native_trace, canonical_trace)
            diversity = head_diversity_loss(correct_memory)
            loss = correct_nll + args.depend_weight * dependency + args.route_weight * route_loss + args.readout_weight * readout_loss + args.diversity_weight * diversity
            loss.backward(); torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0); optimizer.step()
            if any(parameter.grad is not None for parameter in model.parameters()): raise RuntimeError("Gradient reached Receiver backbone")
            for module in (native_reader, canonical_reader):
                if any(parameter.grad is not None for parameter in module.parameters()): raise RuntimeError("Gradient reached frozen Reader")
            for name, value in (("loss", loss), ("correct", correct_nll), ("wrong", wrong_nll), ("dependency", dependency), ("route", route_loss), ("readout", readout_loss), ("diversity", diversity)):
                totals[name].append(float(value.detach()))
        epoch_loss = sum(totals["loss"]) / len(totals["loss"]); writer.eval(); routes = writer.routing_weights().detach().cpu()
        record = {"epoch": epoch, "temperature": writer.temperature, **{f"train_{name}": sum(values) / len(values) for name, values in totals.items()},
                  "selected_native_head": routes.argmax(-1).tolist(), "native_head_usage": routes.mean(dim=1).tolist()}
        if epoch % args.monitor_every == 0 or epoch == args.epochs:
            metrics, rows = monitor(model, tokenizer, canonical_reader, writer, cache, negatives, monitor_indices, device, args.max_new_tokens)
            record["train_free_running_monitor"] = metrics; write_jsonl(output / f"monitor_epoch_{epoch:03d}.jsonl", rows)
        history.append(record); write_jsonl(output / "training_history.jsonl", history)
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch; state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in writer.state_dict().items()})
            torch.save({"writer": state, "writer_metadata": writer.metadata(), "mode": args.mode, "epoch": epoch, "train_loss": best_loss, "args": vars(args),
                        "native_reader": args.native_reader, "native_reader_sha256": file_sha256(args.native_reader),
                        "canonical_reader": args.canonical_reader, "canonical_reader_sha256": file_sha256(args.canonical_reader),
                        "receiver_backbone_frozen": True, "readers_frozen": True, "only_writer_updated": True}, output / "checkpoint_best.pt")
    write_json(output / "TRAIN_SUCCESS.json", {"status": "complete", "experiment": "P3-E-C2 Head-Structured Writer", "mode": args.mode, "samples": total,
        "best_epoch_by_train_objective": best_epoch, "best_train_loss": best_loss, "writer_metadata": writer.metadata(),
        "trainable_parameters": sum(parameter.numel() for parameter in writer.parameters()), "only_writer_updated": True, "checkpoint": str(output / "checkpoint_best.pt")})


if __name__ == "__main__": main()
