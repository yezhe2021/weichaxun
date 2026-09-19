import argparse
import json

import torch

from common import configuration, load_model, run_root, save_json, seed_all
from data import load_pair, load_source, manifest_rows
from experiment import load_teacher, receiver_logits
from protocol import capture_decision, final_logits


CONDITIONS = (
    "qwen_full_native", "qwen_native_selected32", "gemma_full_native",
    "gemma_native_oracle32", "gemma_self_selected32", "gemma_zero32",
    "gemma_no_memory", "gemma_shuffled_oracle32",
)


def pair_counts(first, second):
    both = sum(a and b for a, b in zip(first, second))
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    return {"both_correct": both, "only_first_correct": only_first,
            "only_second_correct": only_second,
            "both_wrong": len(first) - both - only_first - only_second}


@torch.no_grad()
def run(cfg):
    rows = manifest_rows(cfg, "test")
    qwen_selected = {}
    qwen = load_model(cfg, "qwen")
    try:
        for number, row in enumerate(rows, 1):
            source = load_source(cfg, "test", row)
            fields = row["encoded"]["qwen"]
            full_k, full_v, _, captured = capture_decision(
                qwen, fields["full"], len(fields["body"]), fields["choice_ids"])
            if not torch.allclose(captured.float(), source["full_choice_logits"].float(), atol=2e-3, rtol=2e-3):
                raise RuntimeError(f"Qwen3 full-native cache drift: {row['id']}")
            qlen = fields["question_prefix_length"]
            key = torch.cat((full_k[:, :qlen], source["source_k"][:, 1:]), 1).cuda()
            value = torch.cat((full_v[:, :qlen], source["source_v"][:, 1:]), 1).cuda()
            logits = final_logits(qwen, fields["receiver_answer"], key, value,
                                  positions=torch.arange(key.shape[1], device="cuda"), suffix_start=key.shape[1])
            qwen_selected[row["id"]] = logits[fields["choice_ids"]].cpu()
            if number % 32 == 0 or number == len(rows):
                print(f"Qwen3 baseline pass: {number}/{len(rows)}", flush=True)
    finally:
        del qwen
        torch.cuda.empty_cache()

    totals = {name: 0 for name in CONDITIONS}
    correctness = {name: [] for name in CONDITIONS}
    records = []
    gemma = load_model(cfg, "gemma")
    try:
        for number, row in enumerate(rows, 1):
            source = load_source(cfg, "test", row)
            pair = load_pair(cfg, "test", row)
            teacher = load_teacher(cfg, "test", row)
            shuffled = load_pair(cfg, "test", rows[(number % len(rows))])
            fields = row["encoded"]["gemma"]
            target_k, target_v = pair["target_k"][:, 1:].cuda(), pair["target_v"][:, 1:].cuda()
            zero_logits = receiver_logits(gemma, row, pair, torch.zeros_like(target_k), torch.zeros_like(target_v))
            no_memory_logits = final_logits(gemma, fields["question_prefix"] + fields["receiver_answer"])
            shuffled_logits = receiver_logits(
                gemma, row, pair, shuffled["target_k"][:, 1:].cuda(), shuffled["target_v"][:, 1:].cuda())
            self_logits = receiver_logits(gemma, row, pair,
                                          pair["gemma_self_k"].cuda(), pair["gemma_self_v"].cuda())
            choices = {
                "qwen_full_native": source["full_choice_logits"],
                "qwen_native_selected32": qwen_selected[row["id"]],
                "gemma_full_native": pair["receiver_full_choice_logits"],
                "gemma_native_oracle32": teacher["choice_logits"],
                "gemma_self_selected32": self_logits[fields["choice_ids"]].cpu(),
                "gemma_zero32": zero_logits[fields["choice_ids"]].cpu(),
                "gemma_no_memory": no_memory_logits[fields["choice_ids"]].cpu(),
                "gemma_shuffled_oracle32": shuffled_logits[fields["choice_ids"]].cpu(),
            }
            record = {"id": row["id"], "gold_index": row["gold_index"], "conditions": {}}
            for name, logits in choices.items():
                prediction = int(logits.argmax())
                correct = prediction == row["gold_index"]
                totals[name] += correct
                correctness[name].append(correct)
                record["conditions"][name] = {"prediction": prediction, "correct": bool(correct),
                                                "choice_logits": logits.float().tolist()}
            records.append(record)
            if number % 16 == 0 or number == len(rows):
                print(f"Gemma baseline/control pass: {number}/{len(rows)}", flush=True)
    finally:
        del gemma
        torch.cuda.empty_cache()

    key_pairs = (
        ("gemma_full_native", "gemma_native_oracle32"),
        ("qwen_full_native", "qwen_native_selected32"),
        ("gemma_native_oracle32", "gemma_self_selected32"),
        ("gemma_native_oracle32", "gemma_zero32"),
        ("gemma_native_oracle32", "gemma_no_memory"),
        ("gemma_native_oracle32", "gemma_shuffled_oracle32"),
    )
    summary = {
        "status": "completed", "protocol": cfg["protocol"], "sample_count": len(rows),
        "metrics": {name: {"correct": totals[name], "accuracy": totals[name] / len(rows)} for name in CONDITIONS},
        "pairwise_correctness": {
            f"{first}_vs_{second}": pair_counts(correctness[first], correctness[second])
            for first, second in key_pairs
        },
        "definitions": {
            "qwen_native_selected32": "Qwen-native Question + its selected 32 native Option KVs + native Answer.",
            "gemma_native_oracle32": "Gemma-native Question + Gemma-native KVs at Qwen-selected RawAnchor positions + native Answer.",
            "gemma_self_selected32": "Gemma-native Question + its own decision-selected 32 native Option KVs + native Answer.",
            "gemma_zero32": "The Oracle32 K/V tensors are replaced by exact zeros after construction.",
            "gemma_no_memory": "Gemma processes native Question and Answer prefix with no Option-memory cache.",
            "gemma_shuffled_oracle32": "Each question receives the next test sample's native Oracle32 memory.",
        },
    }
    root = run_root(cfg) / "baselines"
    save_json(root / "summary.json", summary)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    args = parser.parse_args()
    cfg = configuration(args.mode)
    seed_all(cfg["seed"])
    run(cfg)
