import argparse
import json
import random
from collections import defaultdict

from p2i_common import load_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Build pair-aligned true A/B mismatch controls")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    grouped = defaultdict(dict)
    for row in load_jsonl(args.data):
        grouped[row["pair_id"]][row["variant"]] = row
    pairs = [value for value in grouped.values() if {"base", "counterfactual"}.issubset(value)]
    rng = random.Random(args.seed)
    output = []
    for index, pair in enumerate(pairs):
        answers = {pair["base"]["answer"], pair["counterfactual"]["answer"]}
        candidates = list(range(len(pairs)))
        rng.shuffle(candidates)
        other_index = next(
            candidate for candidate in candidates
            if candidate != index and answers.isdisjoint({
                pairs[candidate]["base"]["answer"], pairs[candidate]["counterfactual"]["answer"]
            })
        )
        other = pairs[other_index]
        for variant in ("base", "counterfactual"):
            left = pair[variant]
            right = other[variant]
            row = dict(left)
            row.update(
                {
                    "id": left["id"] + "__true_mismatch",
                    "evidence_b": right["evidence_b"],
                    "answer": right["answer"],
                    "mismatch_source_pair_id": right["pair_id"],
                    "original_target_answer": left["answer"],
                    "control_kind": "current_A_with_unrelated_B",
                }
            )
            output.append(row)
    write_jsonl(args.out, output)
    with open(args.out + ".manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"status": "complete", "source": args.data, "pairs": len(pairs), "rows": len(output), "seed": args.seed},
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
