import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from p3b_common import answer_scores, best_span, decode_span, marginal_span_loss, resize_tokens, seed_everything, stable_permutation, write_json, write_jsonl
from train_eval_p3b_probe import MultiLayerSpanProbe

from p3c_common import CanonicalCache, summarize_records, support_recall, temporary_span_loss


def tensors(payload, device):
    return payload["keys"].float().to(device), payload["values"].float().to(device), payload["question_state"].float().to(device), payload["metadata"]


@torch.inference_mode()
def evaluate(probe, cache, device, limit, seed, conditions):
    records = []
    limit = min(len(cache), limit or len(cache))
    for index in tqdm(range(limit), desc="p3c_fresh_probe_eval"):
        current = cache.load(index)
        other_index = (index + 1 + seed % max(1, limit - 1)) % limit
        if other_index == index:
            other_index = (index + 1) % limit
        other = cache.load(other_index)
        current_k, current_v, question, current_meta = tensors(current, device)
        other_k, other_v, _, other_meta = tensors(other, device)
        for condition in conditions:
            keys, values, metadata, evidence, source_row = current_k, current_v, current_meta, current["evidence"], current["row"]
            enabled, inverse = True, None
            if condition in {"zero", "question_only"}:
                keys, values = torch.zeros_like(keys), torch.zeros_like(values)
                enabled = condition != "question_only"
            elif condition == "shuffled":
                keys, values, metadata, evidence, source_row = other_k, other_v, other_meta, other["evidence"], other["row"]
            elif condition == "kv_mismatch":
                values = resize_tokens(other_v, current_v.shape[1])
                source_row = other["row"]
            elif condition == "token_permutation":
                permutation = stable_permutation(current_k.shape[1], seed + index, current_k.device)
                inverse = torch.argsort(permutation)
                keys = current_k.index_select(1, permutation)
                values = current_v.index_select(1, permutation)
            elif condition == "layer_permutation":
                permutation = torch.arange(current_k.shape[0] - 1, -1, -1, device=current_k.device)
                keys = current_k.index_select(0, permutation)
                values = current_v.index_select(0, permutation)
            elif condition != "correct":
                raise ValueError(condition)
            output = probe(keys, values, question, enabled)
            if inverse is not None:
                for name in ("start", "end", "support"):
                    output[name] = output[name].index_select(0, inverse)
            start, end = best_span(output["start"], output["end"])
            prediction = decode_span(evidence, metadata["offsets"], start, end)
            current_em, current_f1 = answer_scores(prediction, current["row"]["answer"])
            source_em, source_f1 = answer_scores(prediction, source_row["answer"])
            spans = set(tuple(span) for span in metadata["answer_token_spans"])
            records.append({
                "id": current["row"]["id"], "source_id": source_row["id"], "condition": condition,
                "question": current["row"]["question"], "prediction": prediction,
                "current_answer": current["row"]["answer"], "source_memory_answer": source_row["answer"],
                "current_answer_em": current_em, "current_answer_f1": current_f1,
                "source_memory_em": source_em, "source_memory_f1": source_f1,
                "start_accuracy": float(start in {span[0] for span in spans}),
                "end_accuracy": float(end in {span[1] for span in spans}),
                "supporting_sentence_recall": support_recall(output["support"], metadata["support_token_mask"]),
                "loss": float(temporary_span_loss(output, metadata, 0.0)),
                "layer_weights": output["layer_weights"].float().cpu().tolist(),
            })
    return records


def condition(summary, name):
    return next(row for row in summary if row["condition"] == name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--native-result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_cache, validation_cache, test_cache = CanonicalCache(args.train_cache), CanonicalCache(args.validation_cache), CanonicalCache(args.test_cache)
    layers = train_cache.index["layers"]
    probe = MultiLayerSpanProbe("pca", list(range(layers))).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
    train_limit = min(len(train_cache), args.max_train or len(train_cache))
    validation_limit = min(len(validation_cache), args.max_validation or len(validation_cache))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history, best_score, best_state, best_epoch = [], -float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        probe.train()
        order = list(range(train_limit))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, index in enumerate(tqdm(order, desc=f"p3c_fresh_probe_s{args.seed}_e{epoch}"), 1):
            payload = train_cache.load(index)
            keys, values, question, metadata = tensors(payload, device)
            result = probe(keys, values, question)
            loss = temporary_span_loss(result, metadata, 0.05) / 4
            loss.backward()
            losses.append(float(loss.detach()) * 4)
            if step % 4 == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        probe.eval()
        records = evaluate(probe, validation_cache, device, validation_limit, args.seed, ["correct", "zero"])
        summary = summarize_records(records)
        correct, zero = condition(summary, "correct"), condition(summary, "zero")
        score = correct["current_answer_f1"] - zero["current_answer_f1"]
        history.append({"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)), "correct_f1": correct["current_answer_f1"], "zero_f1": zero["current_answer_f1"], "selection_score": score})
        write_jsonl(output / "training_history.jsonl", history)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    probe.eval()
    torch.save({"model": best_state, "seed": args.seed, "layers": layers, "best_epoch": best_epoch}, output / "fresh_probe_best.pt")
    conditions = ["correct", "question_only", "zero", "shuffled", "kv_mismatch", "token_permutation", "layer_permutation"]
    records = evaluate(probe, test_cache, device, args.max_test, args.seed, conditions)
    write_jsonl(output / "predictions.jsonl", records)
    summaries = summarize_records(records)
    correct, zero, shuffled, mismatch = (condition(summaries, name) for name in ("correct", "zero", "shuffled", "kv_mismatch"))
    with open(args.native_result, encoding="utf-8") as handle:
        native = __import__("json").load(handle)
    native_correct = next(row for row in native["conditions"] if row["condition"] == "correct")
    native_zero = next(row for row in native["conditions"] if row["condition"] == "zero")
    denominator = native_correct["current_answer_f1"] - native_zero["current_answer_f1"]
    retention = (correct["current_answer_f1"] - zero["current_answer_f1"]) / denominator if abs(denominator) > 1e-9 else None
    write_json(output / "SUCCESS.json", {
        "status": "complete", "seed": args.seed, "layers": layers, "best_epoch": best_epoch,
        "fresh_probe_randomly_initialized": True, "writer_frozen": True,
        "conditions": summaries, "native_reference": {"correct_f1": native_correct["current_answer_f1"], "zero_f1": native_zero["current_answer_f1"]},
        "retention": retention,
        "causal_gaps": {
            "correct_minus_zero_f1": correct["current_answer_f1"] - zero["current_answer_f1"],
            "correct_minus_shuffled_current_f1": correct["current_answer_f1"] - shuffled["current_answer_f1"],
            "correct_minus_mismatch_f1": correct["current_answer_f1"] - mismatch["current_answer_f1"],
            "shuffled_source_answer_f1": shuffled["source_memory_f1"],
        },
    })


if __name__ == "__main__":
    main()
