import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from causal_common import iter_cache, resolve_device, write_csv, write_jsonl
from p15_common import GeneralEvidenceAdapter


CONDITIONS = ("correct", "counterfactual", "shuffled", "mismatched", "zero")


def grouped_pairs(index_path, max_pairs):
    grouped = defaultdict(dict)
    for example in iter_cache(index_path):
        grouped[example["pair_id"]][example["variant"]] = example
    pairs = [value for value in grouped.values() if {"base", "counterfactual"}.issubset(value)]
    return pairs[:max_pairs] if max_pairs > 0 else pairs


def run_reader(adapter, question_example, memory_a, memory_b, device):
    question = question_example["question_state"].to(device=device, dtype=torch.float32).unsqueeze(0)
    memory_a = memory_a.to(device).unsqueeze(0) if memory_a is not None else None
    memory_b = memory_b.to(device).unsqueeze(0) if memory_b is not None else None
    final, rounds, attentions = adapter.reader(question, memory_a, memory_b, return_rounds=True)
    return (
        final[0].detach().cpu(),
        [state[0].detach().cpu() for state in rounds],
        [attention[0].detach().float().cpu() for attention in attentions],
    )


def cosine_distance(left, right):
    return float(1.0 - F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def normalized_l2(left, right):
    return float((left - right).norm().div(left.norm().clamp_min(1e-8)).item())


def attention_stats(attention, a_length):
    probability = attention.mean(dim=0)[0]
    probability = probability / probability.sum().clamp_min(1e-8)
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum()
    normalized_entropy = entropy / np.log(max(2, probability.numel()))
    return {
        "a_mass": float(probability[:a_length].sum()),
        "b_mass": float(probability[a_length:].sum()),
        "entropy": float(normalized_entropy),
        "probability": probability,
    }


def js_divergence(left, right):
    length = min(left.numel(), right.numel())
    left = left[:length] / left[:length].sum().clamp_min(1e-8)
    right = right[:length] / right[:length].sum().clamp_min(1e-8)
    middle = 0.5 * (left + right)
    value = 0.5 * (
        F.kl_div(middle.clamp_min(1e-8).log(), left, reduction="sum")
        + F.kl_div(middle.clamp_min(1e-8).log(), right, reduction="sum")
    )
    return float(value)


@torch.inference_mode()
def extract_condition_states(adapter, pairs, device, desc):
    rows = []
    geometry = []
    for index, pair in enumerate(tqdm(pairs, desc=desc)):
        base = pair["base"]
        counterfactual = pair["counterfactual"]
        other = pairs[(index + 1) % len(pairs)]["base"]
        sources = {
            "correct": (base["memory_a"], base["memory_b"]),
            "counterfactual": (base["memory_a"], counterfactual["memory_b"]),
            "shuffled": (base["memory_a"], other["memory_b"]),
            "mismatched": (other["memory_a"], other["memory_b"]),
            "zero": (None, None),
        }
        states = {}
        round_states = {}
        attention_by_condition = {}
        for condition, (memory_a, memory_b) in sources.items():
            final, rounds, attentions = run_reader(adapter, base, memory_a, memory_b, device)
            states[condition] = final
            round_states[condition] = rounds
            attention_by_condition[condition] = attentions
            rows.append(
                {
                    "pair_id": base["pair_id"],
                    "condition": condition,
                    "state": final,
                }
            )

        record = {"pair_id": base["pair_id"], "schema": base["schema"]}
        for condition in CONDITIONS[1:]:
            record[f"cosine_distance_correct_{condition}"] = cosine_distance(
                states["correct"], states[condition]
            )
            record[f"normalized_l2_correct_{condition}"] = normalized_l2(
                states["correct"], states[condition]
            )
        for round_index in range(len(round_states["correct"])):
            record[f"round_{round_index + 1}_cf_cosine_distance"] = cosine_distance(
                round_states["correct"][round_index], round_states["counterfactual"][round_index]
            )
            if round_index == 0:
                previous = base["question_state"].float()
                if previous.numel() == round_states["correct"][round_index].numel():
                    record[f"round_{round_index + 1}_state_update_l2"] = normalized_l2(
                        previous, round_states["correct"][round_index]
                    )
            else:
                record[f"round_{round_index + 1}_state_update_l2"] = normalized_l2(
                    round_states["correct"][round_index - 1], round_states["correct"][round_index]
                )
            correct_attention = attention_stats(
                attention_by_condition["correct"][round_index], base["memory_a"].shape[0]
            )
            cf_attention = attention_stats(
                attention_by_condition["counterfactual"][round_index], base["memory_a"].shape[0]
            )
            record[f"round_{round_index + 1}_correct_a_mass"] = correct_attention["a_mass"]
            record[f"round_{round_index + 1}_correct_b_mass"] = correct_attention["b_mass"]
            record[f"round_{round_index + 1}_correct_entropy"] = correct_attention["entropy"]
            record[f"round_{round_index + 1}_cf_attention_js"] = js_divergence(
                correct_attention["probability"], cf_attention["probability"]
            )
        geometry.append(record)
    return rows, geometry


def pool_candidates(example):
    memory = example["memory_b"].float()
    masks = example["b_answer_masks"].float()
    weights = masks / masks.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.einsum("ct,td->cd", weights, memory)


@torch.inference_mode()
def extract_answer_examples(adapter, pairs, device, desc):
    rows = []
    for pair in tqdm(pairs, desc=desc):
        for variant in ("base", "counterfactual"):
            example = pair[variant]
            _, rounds, _ = run_reader(
                adapter, example, example["memory_a"], example["memory_b"], device
            )
            rows.append(
                {
                    "id": example["id"],
                    "pair_id": example["pair_id"],
                    "variant": variant,
                    "round_states": torch.stack(rounds),
                    "candidates": pool_candidates(example),
                    "target": int(example["target_candidate_index"]),
                }
            )
    return rows


class ConditionProbe(nn.Module):
    def __init__(self, state_dim, classes):
        super().__init__()
        self.linear = nn.Linear(state_dim, classes)

    def forward(self, states):
        return self.linear(F.normalize(states, dim=-1))


class CandidateProbe(nn.Module):
    def __init__(self, state_dim, memory_dim, probe_dim):
        super().__init__()
        self.state_projection = nn.Linear(state_dim, probe_dim, bias=False)
        self.candidate_projection = nn.Linear(memory_dim, probe_dim, bias=False)

    def forward(self, states, candidates):
        query = F.normalize(self.state_projection(states), dim=-1)
        keys = F.normalize(self.candidate_projection(candidates), dim=-1)
        return torch.einsum("bd,bcd->bc", query, keys)


def train_condition_probe(train_rows, test_rows, epochs, lr, device):
    label = {condition: index for index, condition in enumerate(CONDITIONS)}
    train_x = torch.stack([row["state"] for row in train_rows]).to(device)
    train_y = torch.tensor([label[row["condition"]] for row in train_rows], device=device)
    test_x = torch.stack([row["state"] for row in test_rows]).to(device)
    test_y = torch.tensor([label[row["condition"]] for row in test_rows], device=device)
    probe = ConditionProbe(train_x.shape[-1], len(CONDITIONS)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = probe(train_x).argmax(dim=-1).eq(train_y).float().mean()
        test_accuracy = probe(test_x).argmax(dim=-1).eq(test_y).float().mean()
    return float(train_accuracy), float(test_accuracy)


def stack_answer_rows(rows, round_index, device):
    states = torch.stack([row["round_states"][round_index] for row in rows]).to(device)
    candidates = torch.stack([row["candidates"] for row in rows]).to(device)
    targets = torch.tensor([row["target"] for row in rows], dtype=torch.long, device=device)
    return states, candidates, targets


def ranking_metrics(logits, targets):
    ordering = logits.argsort(dim=-1, descending=True)
    ranks = ordering.eq(targets.unsqueeze(1)).float().argmax(dim=1) + 1
    return {
        "top1": float(logits.argmax(dim=-1).eq(targets).float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
    }


def train_candidate_probe(train_rows, test_rows, round_index, args, device, shuffle_labels=False):
    train_s, train_c, train_y = stack_answer_rows(train_rows, round_index, device)
    test_s, test_c, test_y = stack_answer_rows(test_rows, round_index, device)
    if shuffle_labels:
        generator = torch.Generator(device=device).manual_seed(args.seed + round_index + 1000)
        train_y = train_y[torch.randperm(train_y.numel(), generator=generator, device=device)]
    probe = CandidateProbe(train_s.shape[-1], train_c.shape[-1], args.probe_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=1e-3)
    for _ in range(args.probe_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_s, train_c), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_metrics = ranking_metrics(probe(train_s, train_c), train_y)
        test_metrics = ranking_metrics(probe(test_s, test_c), test_y)
        base_indices = torch.tensor(
            [index for index, row in enumerate(test_rows) if row["variant"] == "base"], device=device
        )
        cf_indices = torch.tensor(
            [index for index, row in enumerate(test_rows) if row["variant"] == "counterfactual"], device=device
        )
        base_metrics = ranking_metrics(probe(test_s[base_indices], test_c[base_indices]), test_y[base_indices])
        cf_metrics = ranking_metrics(probe(test_s[cf_indices], test_c[cf_indices]), test_y[cf_indices])
    return {
        "train_top1": train_metrics["top1"],
        "test_top1": test_metrics["top1"],
        "test_mrr": test_metrics["mrr"],
        "test_base_top1": base_metrics["top1"],
        "test_counterfactual_top1": cf_metrics["top1"],
    }


def effective_rank(states):
    centered = states - states.mean(dim=0, keepdim=True)
    values = torch.linalg.svdvals(centered.float())
    probability = values.square() / values.square().sum().clamp_min(1e-8)
    return float(torch.exp(-(probability * probability.clamp_min(1e-8).log()).sum()))


def mean_fields(rows):
    result = {}
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    for key in numeric:
        result[f"mean_{key}"] = float(np.mean([row[key] for row in rows if key in row]))
    return result


def main():
    parser = argparse.ArgumentParser(description="Diagnose whether the frozen P1.5 Reader state contains evidence content")
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-pairs", type=int, default=32)
    parser.add_argument("--test-pairs", type=int, default=32)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-dim", type=int, default=128)
    parser.add_argument("--probe-lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    adapter = GeneralEvidenceAdapter(
        memory_dim=int(checkpoint["memory_dim"]),
        receiver_dim=int(checkpoint["receiver_hidden_size"]),
        state_dim=int(train_args["state_dim"]),
        reader_heads=int(train_args["reader_heads"]),
        reader_rounds=int(train_args["reader_rounds"]),
        writer_layers=train_args["writer_layers"],
        writer_bottleneck=int(train_args["writer_bottleneck"]),
        max_gate=float(train_args["max_gate"]),
    ).to(device).eval()
    adapter.load_state_dict(checkpoint["adapter"])
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)

    train_pairs = grouped_pairs(args.train_index, args.train_pairs)
    test_pairs = grouped_pairs(args.test_index, args.test_pairs)
    train_conditions, train_geometry = extract_condition_states(adapter, train_pairs, device, "train_geometry")
    test_conditions, test_geometry = extract_condition_states(adapter, test_pairs, device, "test_geometry")
    train_answers = extract_answer_examples(adapter, train_pairs, device, "train_answer_states")
    test_answers = extract_answer_examples(adapter, test_pairs, device, "test_answer_states")

    condition_train, condition_test = train_condition_probe(
        train_conditions, test_conditions, args.probe_epochs, args.probe_lr, device
    )
    probe_rows = []
    for round_index in range(int(train_args["reader_rounds"])):
        metrics = train_candidate_probe(train_answers, test_answers, round_index, args, device)
        shuffled = train_candidate_probe(
            train_answers, test_answers, round_index, args, device, shuffle_labels=True
        )
        probe_rows.append(
            {
                "round": round_index + 1,
                **metrics,
                "label_shuffle_test_top1": shuffled["test_top1"],
            }
        )

    correct_test_states = torch.stack(
        [row["state"] for row in test_conditions if row["condition"] == "correct"]
    )
    candidate_count = int(test_answers[0]["candidates"].shape[0])
    summary = {
        "status": "complete",
        "args": vars(args),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "random_condition_baseline": 1.0 / len(CONDITIONS),
        "random_candidate_baseline": 1.0 / candidate_count,
        "condition_probe_train_accuracy": condition_train,
        "condition_probe_test_accuracy": condition_test,
        "correct_state_effective_rank": effective_rank(correct_test_states),
        "geometry": mean_fields(test_geometry),
        "candidate_probes": probe_rows,
    }
    best_probe = max(row["test_top1"] for row in probe_rows)
    summary["diagnostic_flags"] = {
        "condition_information_present": condition_test >= 2.0 / len(CONDITIONS),
        "answer_information_present": best_probe >= min(1.0, 2.0 / candidate_count),
        "best_candidate_probe_top1": best_probe,
    }

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train_pair_geometry.jsonl", train_geometry)
    write_jsonl(output / "test_pair_geometry.jsonl", test_geometry)
    write_csv(output / "candidate_probe_by_round.csv", probe_rows)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
