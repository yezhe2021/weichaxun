import argparse
import json
import random

from p2id_common import add_p2i_path, write_json

add_p2i_path()
from p2i_common import LazyPairCache


def main():
    parser = argparse.ArgumentParser(description="Create one fixed P2-I-D small-pair diagnostic subset")
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pairs", type=int, choices=(8, 16), default=8)
    parser.add_argument("--train-prefix", type=int, default=448)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    cache = LazyPairCache(args.index, capacity=1)
    if len(cache) < args.train_prefix:
        raise ValueError("Training cache is shorter than the fixed train prefix")
    indices = sorted(random.Random(args.seed).sample(range(args.train_prefix), args.pairs))
    rows = []
    for index in indices:
        pair = cache.load(index)
        rows.append(
            {
                "pair_index": index,
                "pair_id": pair["base"]["pair_id"],
                "base_answer": pair["base"]["answer"],
                "counterfactual_answer": pair["counterfactual"]["answer"],
            }
        )
    write_json(args.out, {"status": "complete", "seed": args.seed, "pairs": rows})


if __name__ == "__main__":
    main()
