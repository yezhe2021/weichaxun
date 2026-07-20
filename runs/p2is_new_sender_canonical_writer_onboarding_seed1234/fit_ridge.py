import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p2is_common import PairCache, load_aligned_pair, write_json


def fit(q4, old, field, target_field, pairs, ridge):
    count = 0; sum_x = torch.zeros(1024, dtype=torch.float64); sum_y = torch.zeros(256, dtype=torch.float64)
    xx = torch.zeros(1024, 1024, dtype=torch.float64); xy = torch.zeros(1024, 256, dtype=torch.float64)
    for index in tqdm(range(pairs), desc=f"ridge_{field}"):
        qpair, opair = load_aligned_pair(q4, old, index)
        for variant in ("base", "counterfactual"):
            x = qpair[variant][field].double(); y = opair[variant]["memory"][target_field].double()
            if x.shape[0] != y.shape[0]: raise RuntimeError("Token axes differ during ridge fitting")
            count += x.shape[0]; sum_x += x.sum(0); sum_y += y.sum(0); xx += x.T @ x; xy += x.T @ y
    mean_x, mean_y = sum_x / count, sum_y / count
    centered_xx = xx - count * torch.outer(mean_x, mean_x)
    centered_xy = xy - count * torch.outer(mean_x, mean_y)
    weight = torch.linalg.solve(centered_xx + float(ridge) * torch.eye(1024, dtype=torch.float64), centered_xy)
    bias = mean_y - mean_x @ weight
    return {"weight": weight.float(), "bias": bias.float(), "samples": count}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--q4-index", required=True); parser.add_argument("--old-index", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--train-pairs", type=int, default=448); parser.add_argument("--ridge", type=float, default=10.0)
    args = parser.parse_args(); q4, old = PairCache(args.q4_index, 2), PairCache(args.old_index, 2)
    if len(q4) != len(old) or args.train_pairs != 448: raise ValueError("Ridge requires aligned 512-pair caches and train prefix 448")
    state = {
        "key": fit(q4, old, "key_flat", "keys", args.train_pairs, args.ridge),
        "value": fit(q4, old, "value_flat", "values", args.train_pairs, args.ridge),
        "ridge": args.ridge, "train_pairs": args.train_pairs,
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True); torch.save(state, output)
    write_json(output.with_suffix(".json"), {"status": "complete", "ridge": args.ridge, "train_pairs": 448, "fit_split": "train_only_0_447"})


if __name__ == "__main__": main()
