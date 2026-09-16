import argparse
import json

import torch

from common import configuration, load_model, run_root, save_json, seed_all
from data import load_pair, load_source, manifest_rows
from experiment import load_teacher, receiver_logits
from protocol import capture_decision, final_logits


CONDITIONS = (
    "qwen_full_native", "qwen_native_selected32", "llama1_full_native",
    "llama1_native_oracle32", "llama1_zero32", "llama1_no_memory", "llama1_shuffled_oracle32",
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
                raise RuntimeError(f"Qwen full-native cache drift: {row['id']}")
            qlen = fields["question_prefix_length"]
            key = torch.cat((full_k[:, :qlen], source["source_k"][:, 1:]), 1).cuda()
            value = torch.cat((full_v[:, :qlen], source["source_v"][:, 1:]), 1).cuda()
            logits = final_logits(qwen, fields["receiver_answer"], key, value,
                                  positions=torch.arange(key.shape[1], device="cuda"), suffix_start=key.shape[1])
            qwen_selected[row["id"]] = logits[fields["choice_ids"]].cpu()
            if number % 32 == 0 or number == len(rows):
                print(f"Qwen baseline pass: {number}/{len(rows)}", flush=True)
    finally:
        del qwen
        torch.cuda.empty_cache()

    totals = {name: 0 for name in CONDITIONS}
    correctness = {name: [] for name in CONDITIONS}
    records = []
    llama = load_model(cfg, "llama")
    try:
        for number, row in enumerate(rows, 1):
            source = load_source(cfg, "test", row)
            pair = load_pair(cfg, "test", row)
            teacher = load_teacher(cfg, "test", row)
            shuffled = load_pair(cfg, "test", rows[number % len(rows)])
            fields = row["encoded"]["llama"]
            target_k, target_v = pair["target_k"][:, 1:].cuda(), pair["target_v"][:, 1:].cuda()
            zero_logits = receiver_logits(llama, row, pair, torch.zeros_like(target_k), torch.zeros_like(target_v))
            no_memory_logits = final_logits(llama, fields["question_prefix"] + fields["receiver_answer"])
            shuffled_logits = receiver_logits(
                llama, row, pair, shuffled["target_k"][:, 1:].cuda(), shuffled["target_v"][:, 1:].cuda())
            choices = {
                "qwen_full_native": source["full_choice_logits"],
                "qwen_native_selected32": qwen_selected[row["id"]],
                "llama1_full_native": pair["llama_full_choice_logits"],
                "llama1_native_oracle32": teacher["choice_logits"],
                "llama1_zero32": zero_logits[fields["choice_ids"]].cpu(),
                "llama1_no_memory": no_memory_logits[fields["choice_ids"]].cpu(),
                "llama1_shuffled_oracle32": shuffled_logits[fields["choice_ids"]].cpu(),
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
                print(f"Llama1B baseline/control pass: {number}/{len(rows)}", flush=True)
    finally:
        del llama
        torch.cuda.empty_cache()

    pairs = (
        ("qwen_full_native", "qwen_native_selected32"),
        ("llama1_full_native", "llama1_native_oracle32"),
        ("llama1_native_oracle32", "llama1_zero32"),
        ("llama1_native_oracle32", "llama1_no_memory"),
        ("llama1_native_oracle32", "llama1_shuffled_oracle32"),
    )
    summary = {
        "status": "completed", "protocol": cfg["protocol"], "sample_count": len(rows),
        "metrics": {name: {"correct": totals[name], "accuracy": totals[name] / len(rows)} for name in CONDITIONS},
        "pairwise_correctness": {
            f"{first}_vs_{second}": pair_counts(correctness[first], correctness[second])
            for first, second in pairs
        },
        "definitions": {
            "qwen_native_selected32": "Qwen-native Question + its selected 32 native Option KVs + native Answer.",
            "llama1_native_oracle32": "Llama1B-native Question + Llama1B-native KVs at Qwen-selected RawAnchor positions + native Answer.",
            "llama1_zero32": "Oracle32 K/V replaced by exact zeros.",
            "llama1_no_memory": "Llama1B processes native Question and Answer prefix without Option memory.",
            "llama1_shuffled_oracle32": "Each question receives the next test sample's Llama1B Oracle32 memory.",
        },
    }
    root = run_root(cfg) / "baselines"
    save_json(root / "summary.json", summary)
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
