import argparse
import json

import torch

from common import configuration, load_model, read_json, run_root, save_json, seed_all
from data import load_pair, load_source, manifest_rows
from experiment import load_teacher
from protocol import capture_decision, final_logits


CONDITIONS = (
    "qwen_full_native",
    "qwen_native_selected32",
    "llama_full_native",
    "llama_native_oracle32",
)


def pair_counts(first, second):
    both = sum(a and b for a, b in zip(first, second))
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    return {
        "both_correct": both,
        "only_first_correct": only_first,
        "only_second_correct": only_second,
        "both_wrong": len(first) - both - only_first - only_second,
    }


@torch.no_grad()
def run(cfg):
    rows = manifest_rows(cfg, "test")
    qwen = load_model(cfg, "qwen")
    totals = {name: 0 for name in CONDITIONS}
    correctness = {name: [] for name in CONDITIONS}
    records = []
    try:
        for number, row in enumerate(rows, 1):
            source = load_source(cfg, "test", row)
            pair = load_pair(cfg, "test", row)
            teacher = load_teacher(cfg, "test", row)
            fields = row["encoded"]["qwen"]

            full_k, full_v, _, captured_full_choice = capture_decision(
                qwen, fields["full"], len(fields["body"]), fields["choice_ids"]
            )
            cached_full_choice = source["full_choice_logits"]
            if not torch.allclose(
                captured_full_choice.float(), cached_full_choice.float(), atol=2e-3, rtol=2e-3
            ):
                raise RuntimeError(f"Qwen full-native cache drift: {row['id']}")

            question_length = fields["question_prefix_length"]
            key = torch.cat((full_k[:, :question_length], source["source_k"][:, 1:]), dim=1).cuda()
            value = torch.cat((full_v[:, :question_length], source["source_v"][:, 1:]), dim=1).cuda()
            qwen_selected_logits = final_logits(
                qwen,
                fields["receiver_answer"],
                key,
                value,
                positions=torch.arange(key.shape[1], device="cuda"),
                suffix_start=key.shape[1],
            )

            choice_logits = {
                "qwen_full_native": cached_full_choice,
                "qwen_native_selected32": qwen_selected_logits[fields["choice_ids"]].cpu(),
                "llama_full_native": pair["llama_full_choice_logits"],
                "llama_native_oracle32": teacher["choice_logits"],
            }
            record = {"id": row["id"], "gold_index": row["gold_index"], "conditions": {}}
            for name, logits in choice_logits.items():
                prediction = int(logits.argmax())
                correct = prediction == row["gold_index"]
                totals[name] += correct
                correctness[name].append(correct)
                record["conditions"][name] = {
                    "prediction": prediction,
                    "correct": bool(correct),
                    "choice_logits": logits.float().tolist(),
                }
            records.append(record)
            if number % 16 == 0 or number == len(rows):
                print(f"Reverse native baseline evaluation: {number}/{len(rows)}", flush=True)
    finally:
        del qwen
        torch.cuda.empty_cache()

    summary = {
        "status": "completed",
        "protocol": cfg["protocol"],
        "sample_count": len(rows),
        "metrics": {
            name: {"correct": totals[name], "accuracy": totals[name] / len(rows)}
            for name in CONDITIONS
        },
        "pairwise_correctness": {
            f"{first}_vs_{second}": pair_counts(correctness[first], correctness[second])
            for index, first in enumerate(CONDITIONS)
            for second in CONDITIONS[index + 1 :]
        },
        "definitions": {
            "qwen_native_selected32": (
                "Qwen-native Question cache + the same 32 Qwen-selected native Option KVs + native Answer suffix."
            ),
            "llama_native_oracle32": (
                "Llama-native Question cache + Llama-native Option KVs at the 32 RawAnchor positions selected by Qwen "
                "+ native Answer suffix."
            ),
        },
    }
    root = run_root(cfg) / "results"
    save_json(root / "native_baselines.json", summary)
    with (root / "native_baselines_per_sample.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    comparison_path = root / "comparison.json"
    comparison = read_json(comparison_path)
    comparison["native_baselines"] = summary
    save_json(comparison_path, comparison)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    args = parser.parse_args()
    cfg = configuration(args.mode)
    seed_all(cfg["seed"])
    run(cfg)
