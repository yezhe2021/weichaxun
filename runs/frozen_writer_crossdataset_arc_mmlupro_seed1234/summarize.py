import argparse
import json
import math
from pathlib import Path

from common import ROOT, read_json, save_json


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def paired(first, second):
    both = sum(a and b for a, b in zip(first, second))
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    discordant = only_first + only_second
    p = 1.0 if not discordant else min(
        1.0, 2 * sum(math.comb(discordant, k) for k in range(min(only_first, only_second) + 1)) /
        2 ** discordant)
    return {"both_correct": both, "only_first_correct": only_first,
            "only_second_correct": only_second,
            "both_wrong": len(first) - both - only_first - only_second,
            "second_minus_first": (only_second - only_first) / len(first), "mcnemar_p": p}


def aggregate(rows):
    names = rows[0]["conditions"].keys(); output = {}
    for name in names:
        values = [row["conditions"][name] for row in rows]
        correct = [value["prediction"] == row["gold_index"] for row, value in zip(rows, values)]
        result = {"correct": sum(correct), "accuracy": sum(correct) / len(correct)}
        for source, target in (("choice_kl_to_oracle", "choice_kl_to_oracle"),
                               ("oracle_agreement", "oracle_agreement")):
            available = [value[source] for value in values if source in value]
            if available: result[target] = sum(available) / len(available)
        for kind in ("k", "v"):
            key = f"{kind}_representation"
            available = [value[key] for value in values if key in value]
            if available:
                result[f"{kind}_nmse"] = sum(value["nmse"] for value in available) / len(available)
                result[f"{kind}_cosine"] = sum(value["cosine"] for value in available) / len(available)
        output[name] = result
    return output


COMPARISONS = {
    "qwen": [("lq_oracle32", "lq_stage_a"), ("lq_stage_a", "lq_residual"),
             ("gq_oracle32", "gq_stage_a"), ("gq_stage_a", "gq_residual")],
    "gemma": [("qg_oracle32", "qg_stage_a"), ("qg_stage_a", "qg_old_residual"),
              ("qg_stage_a", "qg_mixed_residual"),
              ("lqg_oracle32", "lqg_stage_a_stage_a"),
              ("lqg_stage_a_stage_a", "lqg_stage_a_mixed_residual"),
              ("lqg_stage_a_stage_a", "lqg_qwen_residual_stage_a"),
              ("lqg_stage_a_mixed_residual", "lqg_both_residuals")],
    "llama": [("ql_oracle32", "ql_stage_a"), ("ql_stage_a", "ql_residual"),
              ("gql_oracle32", "gql_stage_a_stage_a"),
              ("gql_stage_a_stage_a", "gql_stage_a_llama_residual"),
              ("gql_stage_a_stage_a", "gql_qwen_residual_stage_a"),
              ("gql_stage_a_llama_residual", "gql_both_residuals")]
}


def summarize(mode, dataset):
    root = ROOT / "runs" / mode / dataset / "results"
    by_receiver, metrics, pairwise = {}, {}, {}
    for receiver in ("qwen", "gemma", "llama"):
        rows = read_jsonl(root / f"{receiver}_receiver.jsonl")
        by_receiver[receiver] = rows; metrics[receiver] = aggregate(rows); pairwise[receiver] = {}
        for first, second in COMPARISONS[receiver]:
            first_correct = [row["conditions"][first]["prediction"] == row["gold_index"] for row in rows]
            second_correct = [row["conditions"][second]["prediction"] == row["gold_index"] for row in rows]
            pairwise[receiver][f"{first}_vs_{second}"] = paired(first_correct, second_correct)
    manifest = read_json(ROOT / "runs" / mode / dataset / "manifest.json")
    audit = read_json(root / "dataset_audit.json")
    result = {"status": "completed", "dataset": dataset, "sample_count": len(manifest["rows"]),
              "protocol": "frozen OpenBookQA writers; bidirectional one-hop and two-hop zero-shot transfer",
              "no_training_or_target_checkpoint_selection": True,
              "dataset_audit": audit, "metrics": metrics, "pairwise_correctness": pairwise}
    save_json(root / "summary.json", result)
    combined = []
    for index, row in enumerate(manifest["rows"]):
        combined.append({"id": row["id"], "category": row["category"], "gold_index": row["gold_index"],
                         "receivers": {receiver: by_receiver[receiver][index]["conditions"]
                                       for receiver in by_receiver}})
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in combined: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({receiver: {name: round(value["accuracy"] * 100, 2)
                                 for name, value in conditions.items()}
                      for receiver, conditions in metrics.items()}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    parser.add_argument("--dataset", choices=("arc_challenge", "mmlu_pro", "all"), default="all")
    args = parser.parse_args()
    datasets = ("arc_challenge", "mmlu_pro") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets: summarize(args.mode, dataset)


if __name__ == "__main__":
    main()
