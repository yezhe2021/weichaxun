import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from p2id_common import add_p2i_path, resolve_device, state_sha256, write_json, write_jsonl

add_p2i_path()
from canonical_modules import CanonicalEvidenceWriter
from p2i_common import LazyPairCache, native_to


def effective_rank(tensor):
    singular = torch.linalg.svdvals(tensor.float())
    probability = singular / singular.sum().clamp_min(1e-9)
    return float(torch.exp(-(probability * probability.clamp_min(1e-9).log()).sum()).cpu())


def within_slot_cosine(tensor):
    normalized = F.normalize(tensor.float(), dim=-1)
    cosine = normalized @ normalized.T
    count = tensor.shape[0]
    return float(((cosine.sum() - cosine.diag().sum()) / (count * (count - 1))).cpu())


def entropy(values):
    probability = values.float().clamp_min(0)
    probability = probability / probability.sum().clamp_min(1e-9)
    return float((-(probability * probability.clamp_min(1e-9).log()).sum()).cpu())


@torch.inference_mode()
def process_split(writer, cache, indices, device, dtype, split):
    rows = []
    key_samples = []
    value_samples = []
    for pair_index in tqdm(indices, desc=f"slot_diagnostics_{split}"):
        pair = cache.load(pair_index)
        for variant in ("base", "counterfactual"):
            row = pair[variant]
            output = writer(
                native_to(row["memory"], device, dtype),
                output_dtype=torch.float32,
                return_diagnostics=True,
            )
            key = output["keys"].cpu()
            value = output["values"].cpu()
            diagnostics = output["diagnostics"]
            coverage = diagnostics["atom_coverage"].detach().cpu()
            coverage_entropy = entropy(coverage)
            normalized_entropy = coverage_entropy / math.log(max(2, coverage.numel()))
            rows.append(
                {
                    "split": split,
                    "pair_id": row["pair_id"],
                    "variant": variant,
                    "key_within_slot_cosine": within_slot_cosine(key),
                    "value_within_slot_cosine": within_slot_cosine(value),
                    "key_effective_rank": effective_rank(key),
                    "value_effective_rank": effective_rank(value),
                    "assignment_entropy": float(diagnostics["slot_entropy"].cpu()),
                    "slot_usage_entropy": entropy(diagnostics["slot_usage"].detach().cpu()),
                    "atom_coverage_entropy": coverage_entropy,
                    "atom_coverage_normalized_entropy": normalized_entropy,
                    "effective_covered_atoms": float(math.exp(coverage_entropy)),
                    "atoms": int(coverage.numel()),
                }
            )
            key_samples.append(key)
            value_samples.append(value)
    keys = torch.stack(key_samples)
    values = torch.stack(value_samples)
    same_slot_key_cosine = F.cosine_similarity(keys[:-1].float(), keys[1:].float(), dim=-1).mean()
    same_slot_value_cosine = F.cosine_similarity(values[:-1].float(), values[1:].float(), dim=-1).mean()
    aggregate = {
        "examples": len(rows),
        "key_cross_sample_variance": float(keys.float().var(dim=0, unbiased=False).mean()),
        "value_cross_sample_variance": float(values.float().var(dim=0, unbiased=False).mean()),
        "key_adjacent_sample_same_slot_cosine": float(same_slot_key_cosine),
        "value_adjacent_sample_same_slot_cosine": float(same_slot_value_cosine),
    }
    for field in (
        "key_within_slot_cosine",
        "value_within_slot_cosine",
        "key_effective_rank",
        "value_effective_rank",
        "assignment_entropy",
        "slot_usage_entropy",
        "atom_coverage_normalized_entropy",
        "effective_covered_atoms",
    ):
        aggregate[f"mean_{field}"] = float(np.mean([row[field] for row in rows]))
    return rows, aggregate


def main():
    parser = argparse.ArgumentParser(description="P2-I-D frozen Writer slot-collapse diagnostics")
    parser.add_argument("--writer-checkpoint", required=True)
    parser.add_argument("--native-train-index", required=True)
    parser.add_argument("--native-test-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-pairs", type=int, default=448)
    parser.add_argument("--validation-pairs", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    checkpoint = torch.load(args.writer_checkpoint, map_location="cpu", weights_only=False)
    interface = checkpoint["interface"]
    geometry = checkpoint["writer_geometry"]
    writer = CanonicalEvidenceWriter(
        geometry["sender_layers"], geometry["sender_heads"], geometry["sender_head_dim"],
        interface["slots"], interface["canonical_dim"], geometry["atom_dim"],
    ).to(device).eval()
    writer.load_state_dict(checkpoint["writer"])
    for parameter in writer.parameters():
        parameter.requires_grad_(False)
    writer_hash = state_sha256(writer.state_dict())
    if writer_hash != checkpoint["writer_sha256"]:
        raise RuntimeError("Writer checkpoint hash mismatch")
    train = LazyPairCache(args.native_train_index, capacity=2)
    test = LazyPairCache(args.native_test_index, capacity=2)
    validation_indices = range(args.train_pairs, args.train_pairs + args.validation_pairs)
    validation_rows, validation_summary = process_split(
        writer, train, validation_indices, device, dtype, "validation"
    )
    test_rows, test_summary = process_split(writer, test, range(len(test)), device, dtype, "test")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_slot_diagnostics.jsonl", validation_rows + test_rows)
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "writer_sha256": writer_hash,
            "validation": validation_summary,
            "test": test_summary,
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    main()
