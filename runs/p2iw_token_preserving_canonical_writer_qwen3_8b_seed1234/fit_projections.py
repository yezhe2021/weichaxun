import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p2iw_common import PairCache, resolve_device, seed_everything, write_json


def collect(cache, pairs, field, max_tokens, seed):
    chunks = []
    for index in tqdm(range(pairs), desc=f"collect_{field}", leave=False):
        pair = cache.load(index)
        chunks.extend([pair["base"][field], pair["counterfactual"][field]])
    matrix = torch.cat(chunks, dim=0).float()
    if len(matrix) > max_tokens:
        generator = torch.Generator().manual_seed(seed)
        matrix = matrix[torch.randperm(len(matrix), generator=generator)[:max_tokens]]
    return matrix


def fit(matrix, dim, device):
    matrix = matrix.to(device)
    mean = matrix.mean(0)
    centered = matrix - mean
    _, singular, components = torch.pca_lowrank(centered, q=dim, center=False, niter=3)
    scale = singular / max(1, matrix.shape[0] - 1) ** 0.5
    return {"mean": mean.cpu(), "components": components.cpu(), "scale": scale.cpu()}


def semiorthogonal(input_dim, output_dim, seed):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-pairs", type=int, default=448)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    cache = PairCache(args.index, capacity=2)
    if len(cache) < 512 or args.train_pairs != 448:
        raise ValueError("P2-I-W requires the fixed 448/64 split from 512 training pairs")
    states = {}
    for offset, (name, field) in enumerate((("key", "key_flat"), ("value", "value_flat"), ("hidden", "hidden"))):
        states[name] = fit(collect(cache, args.train_pairs, field, args.max_tokens, args.seed + offset), args.dim, device)
        torch.cuda.empty_cache() if device.type == "cuda" else None
    random_state = {
        "key": {"mean": torch.zeros(1024), "components": semiorthogonal(1024, args.dim, args.seed + 101), "scale": torch.ones(args.dim)},
        "value": {"mean": torch.zeros(1024), "components": semiorthogonal(1024, args.dim, args.seed + 102), "scale": torch.ones(args.dim)},
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pca": states, "random": random_state, "train_pairs": args.train_pairs, "seed": args.seed}, output)
    write_json(output.with_suffix(".json"), {
        "status": "complete", "train_pairs": args.train_pairs, "max_tokens": args.max_tokens,
        "dim": args.dim, "seed": args.seed, "fit_split": "train_prefix_only_0_447",
    })


if __name__ == "__main__":
    main()
