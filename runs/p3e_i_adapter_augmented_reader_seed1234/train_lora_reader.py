import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import (answer_scores, forward_answer, generate, load_receiver,
                         pack_answer, seed_everything)
from p3e_f_common import CanonicalCache, memory_to, read_json, sha256_file, write_json, write_jsonl
from p3e_h_common import load_c1_reader
from p3e_i_common import (assert_frozen_gradients, assert_optimizer_boundary,
                          install_lora, load_lora_state, lora_diagnostics,
                          lora_enabled, lora_parameters, lora_state_dict)


@torch.inference_mode()
def initial_equivalence(model, tokenizer, reader, modules, payload, device, max_length):
    ids, mask, _ = pack_answer(tokenizer, payload["row"], payload["row"]["answer"], max_length, device)
    memory = memory_to(payload, device)
    with lora_enabled(modules, False), reader.inject(model, memory):
        base = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
    with lora_enabled(modules, True), reader.inject(model, memory):
        augmented = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
    difference = (base.float() - augmented.float()).abs()
    return {"max_abs_logit_difference": float(difference.max()),
            "mean_abs_logit_difference": float(difference.mean())}


@torch.inference_mode()
def smoke_eval(model, tokenizer, reader, modules, cache, negatives, count, device, max_new_tokens):
    rows = []
    for index in range(count):
        payload, wrong = cache.load(index), cache.load(negatives[index])
        with lora_enabled(modules, True):
            correct = generate(model, tokenizer, reader, payload["row"], memory_to(payload, device), max_new_tokens)
            shuffled = generate(model, tokenizer, reader, payload["row"], memory_to(wrong, device), max_new_tokens)
        with lora_enabled(modules, False):
            question_only = generate(model, tokenizer, reader, payload["row"], memory_to(payload, device),
                                     max_new_tokens, enabled=False)
            reader_off = generate(model, tokenizer, reader, payload["row"], memory_to(payload, device),
                                  max_new_tokens, enabled=False)
        correct_em, correct_f1 = answer_scores(correct["prediction"], payload["row"]["answer"])
        _, shuffled_f1 = answer_scores(shuffled["prediction"], payload["row"]["answer"])
        rows.append({
            "id": payload["row"]["id"], "correct": correct, "shuffled": shuffled,
            "question_only": question_only, "reader_off": reader_off,
            "correct_em": correct_em, "correct_f1": correct_f1,
            "shuffled_current_f1": shuffled_f1,
            "reader_off_exact": float(question_only["token_ids"] == reader_off["token_ids"]),
        })
    return {
        "correct_em": sum(row["correct_em"] for row in rows) / len(rows),
        "correct_f1": sum(row["correct_f1"] for row in rows) / len(rows),
        "shuffled_current_f1": sum(row["shuffled_current_f1"] for row in rows) / len(rows),
        "correct_shuffled_gap": sum(row["correct_f1"] - row["shuffled_current_f1"] for row in rows) / len(rows),
        "reader_off_exact": sum(row["reader_off_exact"] for row in rows) / len(rows),
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
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
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
    c1_checkpoint = torch.load(args.base_reader, map_location="cpu", weights_only=False)
    reader = load_c1_reader(model, c1_checkpoint)
    receiver_layers = list(range(len(model.model.layers) - 8, len(model.model.layers)))
    modules = install_lora(model, receiver_layers, args.rank, args.alpha, args.dropout)
    optimizer = torch.optim.AdamW(lora_parameters(modules), lr=args.lr,
                                  weight_decay=args.weight_decay)
    assert_optimizer_boundary(model, reader, modules, optimizer)
    equivalence = initial_equivalence(
        model, tokenizer, reader, modules, cache.load(0), device, args.max_length
    )
    if equivalence["max_abs_logit_difference"] > 1e-4:
        raise RuntimeError(f"Zero-init LoRA is not C1-equivalent: {equivalence}")
    history, best_loss, best_epoch = [], float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        order = indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        losses, correct_values, shuffled_values = [], [], []
        a_gradients, b_gradients = [], []
        for index in tqdm(order, desc=f"p3e_i_{args.mode}_epoch{epoch}"):
            payload, wrong = cache.load(index), cache.load(negatives[index])
            optimizer.zero_grad(set_to_none=True)
            with lora_enabled(modules, True):
                correct_nll = forward_answer(
                    model, tokenizer, reader, payload["row"], memory_to(payload, device),
                    args.max_length, device
                )
                shuffled_nll = forward_answer(
                    model, tokenizer, reader, payload["row"], memory_to(wrong, device),
                    args.max_length, device
                )
            dependency = F.relu(args.margin + correct_nll - shuffled_nll)
            loss = correct_nll + args.depend_weight * dependency
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite LoRA training loss")
            loss.backward()
            assert_frozen_gradients(model, reader, modules)
            a_gradients.append(sum(float(module.lora_a.grad.float().norm())
                                   for module in modules.values() if module.lora_a.grad is not None))
            b_gradients.append(sum(float(module.lora_b.grad.float().norm())
                                   for module in modules.values() if module.lora_b.grad is not None))
            torch.nn.utils.clip_grad_norm_(lora_parameters(modules), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            correct_values.append(float(correct_nll.detach()))
            shuffled_values.append(float(shuffled_nll.detach()))
        epoch_loss = sum(losses) / len(losses)
        record = {
            "epoch": epoch, "total_loss": epoch_loss,
            "correct_answer_nll": sum(correct_values) / len(correct_values),
            "shuffled_answer_nll": sum(shuffled_values) / len(shuffled_values),
            "mean_lora_a_gradient_norm": sum(a_gradients) / len(a_gradients),
            "mean_lora_b_gradient_norm": sum(b_gradients) / len(b_gradients),
            "lora_diagnostics": lora_diagnostics(modules),
        }
        history.append(record)
        write_jsonl(output / "training_history.jsonl", history)
        state = {
            "lora_state": lora_state_dict(modules),
            "lora_metadata": {
                "receiver_layers": receiver_layers, "targets": list(("q_proj", "v_proj", "o_proj", "down_proj")),
                "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
                "base_receiver_frozen": True, "c1_reader_frozen": True,
            },
            "mode": args.mode, "epoch": epoch, "total_train_loss": epoch_loss,
            "args": vars(args), "base_reader": args.base_reader,
            "base_reader_sha256": sha256_file(args.base_reader),
            "initial_equivalence": equivalence,
        }
        torch.save(state, output / "checkpoint_last.pt")
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            torch.save(state, output / "checkpoint_best.pt")
    smoke_metrics = None
    if args.mode == "smoke16":
        smoke_metrics, rows = smoke_eval(
            model, tokenizer, reader, modules, cache, negatives, count, device,
            args.max_new_tokens
        )
        write_jsonl(output / "smoke_generations.jsonl", rows)
    write_json(output / "TRAIN_SUCCESS.json", {
        "status": "complete", "experiment": "P3-E-I Adapter-Augmented Reader QA-only",
        "mode": args.mode, "samples": count, "epochs": args.epochs,
        "best_epoch_by_training_loss": best_epoch, "best_train_loss": best_loss,
        "initial_equivalence": equivalence, "smoke_free_running": smoke_metrics,
        "trainable_lora_parameters": sum(parameter.numel() for parameter in lora_parameters(modules)),
        "receiver_layers": receiver_layers, "targets": list(("q_proj", "v_proj", "o_proj", "down_proj")),
        "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
        "checkpoint_best": str(output / "checkpoint_best.pt"),
        "checkpoint_last": str(output / "checkpoint_last.pt"),
    })


if __name__ == "__main__":
    main()
