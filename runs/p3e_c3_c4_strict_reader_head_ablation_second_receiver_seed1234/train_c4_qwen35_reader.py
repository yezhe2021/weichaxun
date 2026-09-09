import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import answer_mean_nll, answer_scores, file_sha256, hard_negative_mapping, pack_answer, question_prompt, seed_everything, write_json, write_jsonl
from p3e_c2_common import SenderNativeHeadwiseCache, load_writer, writer_memory
from p3e_c4_common import Qwen35CanonicalReader, load_qwen35


def forward_answer(model, tokenizer, reader, row, memory, max_length, device):
    ids, mask, labels = pack_answer(tokenizer, row, row["answer"], max_length, device)
    with reader.inject(model, memory): output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    return answer_mean_nll(output.logits, labels)


@torch.inference_mode()
def generate(model, tokenizer, reader, row, memory, max_new_tokens, enabled=True):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False); encoded = {name: value.to(model.device) for name, value in encoded.items()}
    kwargs = dict(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if enabled:
        with reader.inject(model, memory): output = model.generate(**kwargs)
    else: output = model.generate(**kwargs)
    from p3d3_common import extract_prediction
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist(); text = tokenizer.decode(tokens, skip_special_tokens=True); prediction, method = extract_prediction(text)
    return {"text": text, "prediction": prediction, "parse_method": method, "token_ids": tokens, "eos_reached": tokenizer.eos_token_id in tokens}


@torch.inference_mode()
def monitor(model, tokenizer, reader, writer, cache, negatives, indices, device, max_new_tokens):
    rows = []
    for index in indices:
        payload, wrong = cache.load(index), cache.load(negatives[index])
        correct = generate(model, tokenizer, reader, payload["row"], writer_memory(writer, payload, device, no_grad=True), max_new_tokens)
        shuffled = generate(model, tokenizer, reader, payload["row"], writer_memory(writer, wrong, device, no_grad=True), max_new_tokens)
        _, correct_f1 = answer_scores(correct["prediction"], payload["row"]["answer"]); _, shuffled_f1 = answer_scores(shuffled["prediction"], payload["row"]["answer"])
        rows.append({"id": payload["row"]["id"], "correct": correct, "shuffled": shuffled, "correct_f1": correct_f1, "shuffled_current_f1": shuffled_f1})
    return {"correct_f1": sum(row["correct_f1"] for row in rows) / len(rows), "shuffled_current_f1": sum(row["shuffled_current_f1"] for row in rows) / len(rows),
            "correct_shuffled_gap": sum(row["correct_f1"] - row["shuffled_current_f1"] for row in rows) / len(rows)}, rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--writer", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--max-samples", type=int, default=512); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--lr", type=float, default=2e-4); parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--depend-weight", type=float, default=0.5); parser.add_argument("--margin", type=float, default=0.5); parser.add_argument("--max-length", type=int, default=512); parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--monitor-samples", type=int, default=4); parser.add_argument("--monitor-every", type=int, default=5); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    cache = SenderNativeHeadwiseCache(args.memory); total = min(args.max_samples, len(cache)); indices = list(range(total)); negatives = hard_negative_mapping(cache)
    model, tokenizer = load_qwen35(args.model, device); writer, _ = load_writer(args.writer, device); writer.requires_grad_(False); writer.eval(); reader = Qwen35CanonicalReader(model, seed=args.seed).to(device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=args.weight_decay); expected = {id(parameter) for parameter in reader.parameters()}; actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual or any(parameter.requires_grad for parameter in model.parameters()) or any(parameter.requires_grad for parameter in writer.parameters()): raise RuntimeError("C4 trainable boundary failed")
    monitor_indices = random.Random(args.seed + 488).sample(indices, min(args.monitor_samples, total)); history, best_loss, best_epoch = [], float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        temperature = 1.0 + (0.25 - 1.0) * ((epoch - 1) / max(1, args.epochs - 1))
        for branch in reader.branches: branch.temperature = temperature
        order = indices.copy(); random.Random(args.seed + epoch).shuffle(order); reader.train(); losses, correct_values, wrong_values = [], [], []
        for sample_index in tqdm(order, desc=f"c4_qwen35_seed{args.seed}_epoch{epoch}"):
            payload, wrong = cache.load(sample_index), cache.load(negatives[sample_index]); optimizer.zero_grad(set_to_none=True)
            correct_nll = forward_answer(model, tokenizer, reader, payload["row"], writer_memory(writer, payload, device, no_grad=True), args.max_length, device)
            wrong_nll = forward_answer(model, tokenizer, reader, payload["row"], writer_memory(writer, wrong, device, no_grad=True), args.max_length, device)
            dependency = F.relu(args.margin + correct_nll - wrong_nll); loss = correct_nll + args.depend_weight * dependency
            loss.backward(); torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0); optimizer.step()
            if any(parameter.grad is not None for parameter in model.parameters()) or any(parameter.grad is not None for parameter in writer.parameters()): raise RuntimeError("Gradient escaped C4 Reader")
            losses.append(float(loss.detach())); correct_values.append(float(correct_nll.detach())); wrong_values.append(float(wrong_nll.detach()))
        epoch_loss = sum(losses) / len(losses); record = {"epoch": epoch, "temperature": temperature, "train_loss": epoch_loss,
            "correct_answer_mean_nll": sum(correct_values) / len(correct_values), "shuffled_answer_mean_nll": sum(wrong_values) / len(wrong_values),
            "gates": reader.gates().detach().cpu().tolist(), "head_routes": [branch.head_route().detach().cpu().tolist() for branch in reader.branches],
            "group_routes": [branch.group_logits.detach().float().softmax(-1).cpu().tolist() for branch in reader.branches]}
        if epoch % args.monitor_every == 0 or epoch == args.epochs:
            reader.eval(); metrics, rows = monitor(model, tokenizer, reader, writer, cache, negatives, monitor_indices, device, args.max_new_tokens); record["train_free_running_monitor"] = metrics
            write_jsonl(output / f"monitor_epoch_{epoch:03d}.jsonl", rows)
        history.append(record); write_jsonl(output / "training_history.jsonl", history)
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch; state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in reader.state_dict().items()})
            torch.save({"reader": state, "reader_metadata": reader.metadata(), "epoch": epoch, "train_loss": best_loss, "args": vars(args),
                        "writer": args.writer, "writer_sha256": file_sha256(args.writer), "writer_frozen": True, "qwen35_backbone_frozen": True}, output / "checkpoint_best.pt")
    write_json(output / "TRAIN_SUCCESS.json", {"status": "complete", "experiment": "C4 frozen C2 Writer to Qwen3.5-4B Reader", "seed": args.seed, "samples": total,
        "best_epoch_by_train_objective": best_epoch, "best_train_loss": best_loss, "reader_metadata": reader.metadata(), "writer_frozen": True, "checkpoint": str(output / "checkpoint_best.pt")})


if __name__ == "__main__": main()
