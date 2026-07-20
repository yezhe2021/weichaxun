import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3b_common import seed_everything, write_json


def load_index(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return path.parent, json.load(handle)


def fit_pca(samples, output_dim, device):
    matrix = torch.cat(samples, dim=0).float()
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    q = min(output_dim, centered.shape[0] - 1, centered.shape[1])
    if q < output_dim:
        raise RuntimeError(f"Only {q} PCA components available")
    _, _, vectors = torch.pca_lowrank(centered.to(device), q=output_dim, center=False, niter=3)
    return mean.cpu(), vectors.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--tokens-per-sample", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    root, index = load_index(args.cache)
    entries = index["entries"][: args.max_samples or None]
    key_atoms = [[] for _ in range(36)]
    value_atoms = [[] for _ in range(36)]
    generator = torch.Generator().manual_seed(args.seed)
    for entry in tqdm(entries, desc="p3b_collect_pca_atoms"):
        payload = torch.load(root / entry["file"], map_location="cpu", weights_only=False)
        memory = payload["modes"]["evidence_only"]
        length = memory["keys"].shape[1]
        count = min(args.tokens_per_sample, length)
        chosen = torch.randperm(length, generator=generator)[:count]
        for layer in range(36):
            key_atoms[layer].append(memory["keys"][layer].index_select(0, chosen))
            value_atoms[layer].append(memory["values"][layer].index_select(0, chosen))

    pca = {"key_mean": [], "value_mean": [], "key_projection": [], "value_projection": []}
    random_projection = {"key_projection": [], "value_projection": []}
    for layer in tqdm(range(36), desc="p3b_fit_layer_pca"):
        key_mean, key_projection = fit_pca(key_atoms[layer], args.output_dim, args.device)
        value_mean, value_projection = fit_pca(value_atoms[layer], args.output_dim, args.device)
        pca["key_mean"].append(key_mean)
        pca["value_mean"].append(value_mean)
        pca["key_projection"].append(key_projection)
        pca["value_projection"].append(value_projection)
        key_random = torch.randn(1024, args.output_dim, generator=generator) / args.output_dim ** 0.5
        value_random = torch.randn(1024, args.output_dim, generator=generator) / args.output_dim ** 0.5
        random_projection["key_projection"].append(F.normalize(key_random, dim=0))
        random_projection["value_projection"].append(F.normalize(value_random, dim=0))

    bundle = {
        "pca": {name: torch.stack(values) for name, values in pca.items()},
        "random": {name: torch.stack(values) for name, values in random_projection.items()},
        "config": {
            "input_dim": 1024,
            "output_dim": args.output_dim,
            "layers": 36,
            "samples": len(entries),
            "tokens_per_sample": args.tokens_per_sample,
            "seed": args.seed,
            "pca_is_layer_independent": True,
            "random_is_layer_independent": True,
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    write_json(output.with_suffix(".json"), {"status": "complete", **bundle["config"]})


if __name__ == "__main__":
    main()
