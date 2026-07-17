import argparse
import json
import random
from pathlib import Path

from audit_common import LazyPairCache, pair_id_hash


def greedy_coverage(cache, candidates, count, seed):
    rng = random.Random(seed)
    remaining = list(candidates)
    rng.shuffle(remaining)
    selected = []
    covered = set()
    while remaining and len(selected) < count:
        best_gain = max(
            len(
                {
                    cache.entries[index]["base_answer"],
                    cache.entries[index]["counterfactual_answer"],
                }
                - covered
            )
            for index in remaining
        )
        candidates_with_gain = [
            index
            for index in remaining
            if len(
                {
                    cache.entries[index]["base_answer"],
                    cache.entries[index]["counterfactual_answer"],
                }
                - covered
            )
            == best_gain
        ]
        chosen = rng.choice(candidates_with_gain)
        remaining.remove(chosen)
        selected.append(chosen)
        covered.update(
            {
                cache.entries[chosen]["base_answer"],
                cache.entries[chosen]["counterfactual_answer"],
            }
        )
    return sorted(selected), sorted(covered)


def records(cache, indices):
    return [
        {
            "index": index,
            "pair_id": cache.entries[index]["pair_id"],
            "base_answer": cache.entries[index]["base_answer"],
            "counterfactual_answer": cache.entries[index]["counterfactual_answer"],
        }
        for index in indices
    ]


def main():
    parser = argparse.ArgumentParser(description="Build the P2-E-C1 full Experiment-A split")
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-pairs", type=int, default=64)
    parser.add_argument("--validation-pairs", type=int, default=16)
    parser.add_argument("--test-pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    train_cache = LazyPairCache(args.train_index)
    test_cache = LazyPairCache(args.test_index)
    all_train = list(range(len(train_cache)))
    random.Random(args.seed).shuffle(all_train)
    experiment_a_validation_count = max(1, round(len(all_train) * 0.125))
    validation_pool = sorted(all_train[:experiment_a_validation_count])
    fitting_pool = sorted(all_train[experiment_a_validation_count:])
    train_indices, train_labels = greedy_coverage(
        train_cache, fitting_pool, args.train_pairs, args.seed + 1
    )
    validation_indices, validation_labels = greedy_coverage(
        train_cache, validation_pool, args.validation_pairs, args.seed + 2
    )
    test_indices, test_labels = greedy_coverage(
        test_cache, range(len(test_cache)), args.test_pairs, args.seed + 3
    )
    manifest = {
        "format_version": 1,
        "experiment": "P2-E-C1 calibrated functional readability audit",
        "seed": args.seed,
        "parent_split": {
            "algorithm": "Experiment A random.Random(seed), 12.5% pair-level validation",
            "full_train_pairs": len(train_cache),
            "full_test_pairs": len(test_cache),
        },
        "train": records(train_cache, train_indices),
        "validation": records(train_cache, validation_indices),
        "test": records(test_cache, test_indices),
        "label_coverage": {
            "train": train_labels,
            "validation": validation_labels,
            "test": test_labels,
        },
    }
    for split in ("train", "validation", "test"):
        manifest[f"{split}_pair_id_hash"] = pair_id_hash(
            [row["pair_id"] for row in manifest[split]]
        )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
