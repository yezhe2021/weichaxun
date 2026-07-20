import argparse
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from p3b_common import (
    LAYER_SETS,
    SOURCES,
    answer_scores,
    best_span,
    decode_span,
    layer_set,
    marginal_span_loss,
    mean,
    resize_tokens,
    seed_everything,
    stable_permutation,
    write_json,
    write_jsonl,
)


class Cache:
    def __init__(self, index_path, capacity=2):
        self.path = Path(index_path)
        with self.path.open(encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.entries = self.index["entries"]
        self.root = self.path.parent
        self.capacity = capacity
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index not in self.loaded:
            self.loaded[index] = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]


class LayerReader(nn.Module):
    def __init__(self, input_dim, question_dim, model_dim, max_tokens, trainable_writer=False):
        super().__init__()
        self.trainable_writer = trainable_writer
        if trainable_writer:
            self.writer_k = nn.Linear(input_dim, 256)
            self.writer_v = nn.Linear(input_dim, 256)
            nn.init.orthogonal_(self.writer_k.weight)
            nn.init.orthogonal_(self.writer_v.weight)
            nn.init.zeros_(self.writer_k.bias)
            nn.init.zeros_(self.writer_v.bias)
            input_dim = 256
        self.k_norm = nn.LayerNorm(input_dim)
        self.v_norm = nn.LayerNorm(input_dim)
        self.k_proj = nn.Linear(input_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, model_dim, bias=False)
        self.q_key = nn.Linear(question_dim, model_dim, bias=False)
        self.q_value = nn.Linear(question_dim, model_dim, bias=False)
        self.position = nn.Embedding(max_tokens, model_dim)
        self.start_head = nn.Linear(model_dim, 1)
        self.end_head = nn.Linear(model_dim, 1)
        self.support_head = nn.Linear(model_dim, 1)
        self.router = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.Tanh(), nn.Linear(model_dim, 1))

    def forward(self, keys, values, question, memory_enabled=True):
        if self.trainable_writer:
            keys = self.writer_k(keys)
            values = self.writer_v(values)
        keys = self.k_proj(self.k_norm(keys))
        values = self.v_proj(self.v_norm(values))
        q_key = F.normalize(self.q_key(question), dim=-1)
        q_value = self.q_value(question)
        if memory_enabled:
            route_logits = keys @ q_key / math.sqrt(keys.shape[-1])
            attention = route_logits.softmax(dim=0)
            readout = torch.sum(attention[:, None] * values, dim=0)
        else:
            route_logits = torch.zeros(keys.shape[0], device=keys.device)
            attention = torch.zeros_like(route_logits)
            readout = torch.zeros_like(q_value)
        positions = self.position(torch.arange(keys.shape[0], device=keys.device))
        token_state = torch.tanh(values + q_value[None] + readout[None] + positions)
        start = self.start_head(token_state).squeeze(-1) + route_logits
        end = self.end_head(token_state).squeeze(-1) + route_logits
        support = self.support_head(token_state).squeeze(-1)
        router = self.router(torch.cat((q_value, readout), dim=-1)).squeeze(-1)
        return start, end, support, router, attention


class MultiLayerSpanProbe(nn.Module):
    def __init__(self, source, selected_layers, question_dim=4096, model_dim=128, max_tokens=512):
        super().__init__()
        self.source = source
        self.selected_layers = list(selected_layers)
        input_dim = 4096 if source == "hidden" else (1024 if source in {"native_kv", "trainable"} else 256)
        self.readers = nn.ModuleList(
            [LayerReader(input_dim, question_dim, model_dim, max_tokens, source == "trainable") for _ in self.selected_layers]
        )

    def forward(self, keys, values, question, memory_enabled=True):
        outputs = []
        for reader, layer in zip(self.readers, self.selected_layers):
            outputs.append(reader(keys[layer], values[layer], question, memory_enabled))
        routers = torch.stack([output[3] for output in outputs]).softmax(dim=0)
        start = sum(weight * output[0] for weight, output in zip(routers, outputs))
        end = sum(weight * output[1] for weight, output in zip(routers, outputs))
        support = sum(weight * output[2] for weight, output in zip(routers, outputs))
        attention = torch.stack([output[4] for output in outputs])
        return {"start": start, "end": end, "support": support, "layer_weights": routers, "attention": attention}


def project_memory(payload, sender_mode, source, projections, device):
    memory = payload["modes"][sender_mode]
    if source == "hidden":
        values = memory["hidden"].float().to(device)
        return values, values
    keys = memory["keys"].float().to(device)
    values = memory["values"].float().to(device)
    if source in {"native_kv", "trainable"}:
        return keys, values
    branch = projections[source]
    key_projection = branch["key_projection"].to(device)
    value_projection = branch["value_projection"].to(device)
    if source == "pca":
        keys = keys - branch["key_mean"].to(device)[:, None, :]
        values = values - branch["value_mean"].to(device)[:, None, :]
    keys = torch.einsum("ltd,ldh->lth", keys, key_projection)
    values = torch.einsum("ltd,ldh->lth", values, value_projection)
    return F.layer_norm(keys, (256,)), F.layer_norm(values, (256,))


def span_loss(output, metadata, support_weight):
    loss = marginal_span_loss(output["start"], output["end"], metadata["answer_token_spans"])
    support = metadata["support_token_mask"].float().to(output["support"].device)
    return loss + support_weight * F.binary_cross_entropy_with_logits(output["support"], support)


def summarize(records):
    summaries = []
    for condition in sorted({record["condition"] for record in records}):
        rows = [record for record in records if record["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "n": len(rows),
                "current_answer_em": mean([row["current_answer_em"] for row in rows]),
                "current_answer_f1": mean([row["current_answer_f1"] for row in rows]),
                "source_memory_em": mean([row["source_memory_em"] for row in rows]),
                "source_memory_f1": mean([row["source_memory_f1"] for row in rows]),
                "start_accuracy": mean([row["start_accuracy"] for row in rows]),
                "end_accuracy": mean([row["end_accuracy"] for row in rows]),
                "supporting_fact_recall": mean([row["supporting_fact_recall"] for row in rows]),
                "loss": mean([row["loss"] for row in rows]),
            }
        )
    return summaries


def support_recall(logits, mask):
    mask = mask.bool().to(logits.device)
    count = int(mask.sum())
    if count == 0:
        return 0.0
    selected = logits.topk(min(count, len(logits))).indices
    return float(mask.index_select(0, selected).float().mean())


@torch.inference_mode()
def evaluate(models, cache, sender_mode, source, projections, device, max_samples, seed, conditions):
    records = {name: [] for name in models}
    limit = min(len(cache), max_samples or len(cache))
    for index in tqdm(range(limit), desc=f"p3b_eval_{sender_mode}_{source}"):
        current = cache.load(index)
        other_index = (index + 1 + seed % max(1, limit - 1)) % limit
        if other_index == index:
            other_index = (index + 1) % limit
        other = cache.load(other_index)
        current_keys, current_values = project_memory(current, sender_mode, source, projections, device)
        other_keys, other_values = project_memory(other, sender_mode, source, projections, device)
        question = current["question_state"].float().to(device)
        current_meta = current["modes"][sender_mode]
        other_meta = other["modes"][sender_mode]
        for condition in conditions:
            keys, values, metadata, source_row = current_keys, current_values, current_meta, current["row"]
            memory_enabled, inverse = True, None
            if condition in {"zero", "question_only"}:
                keys, values = torch.zeros_like(current_keys), torch.zeros_like(current_values)
                memory_enabled = condition != "question_only"
            elif condition == "shuffled":
                keys, values, metadata, source_row = other_keys, other_values, other_meta, other["row"]
            elif condition == "kv_mismatch":
                values = resize_tokens(other_values, current_values.shape[1])
                source_row = other["row"]
            elif condition == "token_permutation":
                permutation = stable_permutation(current_keys.shape[1], seed + index, current_keys.device)
                inverse = torch.argsort(permutation)
                keys = current_keys.index_select(1, permutation)
                values = current_values.index_select(1, permutation)
            elif condition == "layer_permutation":
                layer_permutation = torch.arange(current_keys.shape[0] - 1, -1, -1, device=current_keys.device)
                keys = current_keys.index_select(0, layer_permutation)
                values = current_values.index_select(0, layer_permutation)
            elif condition != "correct":
                raise ValueError(condition)

            for name, model in models.items():
                output = model(keys, values, question, memory_enabled)
                if inverse is not None:
                    for key in ("start", "end", "support"):
                        output[key] = output[key].index_select(0, inverse)
                start, end = best_span(output["start"], output["end"])
                prediction = decode_span(metadata and (other["evidence"] if condition == "shuffled" else current["evidence"]), metadata["offsets"], start, end)
                current_em, current_f1 = answer_scores(prediction, current["row"]["answer"])
                source_em, source_f1 = answer_scores(prediction, source_row["answer"])
                gold_spans = set(tuple(span) for span in metadata["answer_token_spans"])
                start_targets = {span[0] for span in gold_spans}
                end_targets = {span[1] for span in gold_spans}
                loss = span_loss(output, metadata, 0.0)
                records[name].append(
                    {
                        "id": current["row"]["id"],
                        "source_id": source_row["id"],
                        "condition": condition,
                        "question": current["row"]["question"],
                        "current_answer": current["row"]["answer"],
                        "source_memory_answer": source_row["answer"],
                        "prediction": prediction,
                        "predicted_start": start,
                        "predicted_end": end,
                        "current_answer_em": current_em,
                        "current_answer_f1": current_f1,
                        "source_memory_em": source_em,
                        "source_memory_f1": source_f1,
                        "start_accuracy": float(start in start_targets),
                        "end_accuracy": float(end in end_targets),
                        "supporting_fact_recall": support_recall(output["support"], metadata["support_token_mask"]),
                        "loss": float(loss),
                        "layer_weights": output["layer_weights"].float().cpu().tolist(),
                    }
                )
    return records


def save_checkpoint(path, model, source, sender_mode, layers, epoch, metric):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "source": source,
            "sender_mode": sender_mode,
            "layers": layers,
            "epoch": epoch,
            "validation_em": metric,
            "config": {"question_dim": 4096, "model_dim": 128, "max_tokens": 512},
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--projections", required=True)
    parser.add_argument("--sender-mode", choices=("evidence_only", "question_evidence"), required=True)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--layer-sets", default="last1,last4,last8,uniform16,all36")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--support-weight", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    projections = torch.load(args.projections, map_location="cpu", weights_only=False)
    train_cache, validation_cache, test_cache = Cache(args.train_cache), Cache(args.validation_cache), Cache(args.test_cache)
    names = [name for name in args.layer_sets.split(",") if name]
    models = {name: MultiLayerSpanProbe(args.source, layer_set(name)).to(device) for name in names}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for name, model in models.items()
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    best = {name: -1.0 for name in names}
    history = []
    train_limit = min(len(train_cache), args.max_train or len(train_cache))
    validation_limit = min(len(validation_cache), args.max_validation or len(validation_cache))
    conditions = ["correct", "shuffled", "zero", "question_only", "kv_mismatch", "token_permutation", "layer_permutation"]

    for epoch in range(1, args.epochs + 1):
        for model in models.values():
            model.train()
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        order = list(range(train_limit))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_losses = {name: [] for name in names}
        for step, index in enumerate(tqdm(order, desc=f"p3b_train_{args.sender_mode}_{args.source}_e{epoch}"), 1):
            payload = train_cache.load(index)
            keys, values = project_memory(payload, args.sender_mode, args.source, projections, device)
            question = payload["question_state"].float().to(device)
            metadata = payload["modes"][args.sender_mode]
            total = 0.0
            for name, model in models.items():
                result = model(keys, values, question)
                loss = span_loss(result, metadata, args.support_weight)
                epoch_losses[name].append(float(loss.detach()))
                total = total + loss / (len(models) * args.gradient_accumulation)
            total.backward()
            if step % args.gradient_accumulation == 0 or step == len(order):
                for name in names:
                    torch.nn.utils.clip_grad_norm_(models[name].parameters(), 1.0)
                    optimizers[name].step()
                    optimizers[name].zero_grad(set_to_none=True)

        for model in models.values():
            model.eval()
        validation_records = evaluate(
            models,
            validation_cache,
            args.sender_mode,
            args.source,
            projections,
            device,
            validation_limit,
            args.seed,
            ["correct", "zero"],
        )
        for name in names:
            summary = summarize(validation_records[name])
            correct = next(row for row in summary if row["condition"] == "correct")
            zero = next(row for row in summary if row["condition"] == "zero")
            score = correct["current_answer_em"] - zero["current_answer_em"]
            history.append(
                {
                    "epoch": epoch,
                    "layer_set": name,
                    "train_loss": mean(epoch_losses[name]),
                    "validation_correct_em": correct["current_answer_em"],
                    "validation_zero_em": zero["current_answer_em"],
                    "selection_score": score,
                }
            )
            if score > best[name]:
                best[name] = score
                save_checkpoint(output / name / "checkpoint_best.pt", models[name], args.source, args.sender_mode, layer_set(name), epoch, score)
        write_jsonl(output / "training_history.jsonl", history)

    for name, model in models.items():
        checkpoint = torch.load(output / name / "checkpoint_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    records = evaluate(
        models,
        test_cache,
        args.sender_mode,
        args.source,
        projections,
        device,
        args.max_test,
        args.seed,
        conditions,
    )
    family = {"status": "complete", "sender_mode": args.sender_mode, "source": args.source, "layer_sets": {}}
    for name in names:
        write_jsonl(output / name / "predictions.jsonl", records[name])
        summaries = summarize(records[name])
        correct = next(row for row in summaries if row["condition"] == "correct")
        shuffled = next(row for row in summaries if row["condition"] == "shuffled")
        result = {
            "status": "complete",
            "sender_mode": args.sender_mode,
            "source": args.source,
            "layer_set": name,
            "layers": layer_set(name),
            "conditions": summaries,
            "correct_shuffled_em_gap": correct["current_answer_em"] - shuffled["current_answer_em"],
            "best_validation_score": best[name],
        }
        write_json(output / name / "SUCCESS.json", result)
        family["layer_sets"][name] = result
    write_json(output / "SUCCESS.json", family)


if __name__ == "__main__":
    main()
