import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from p2iw_common import PairCache, TokenCanonicalWriter, effective_rank, linear_cka, write_json, write_jsonl


def graph_agreement(left, right):
    left = F.normalize(left.float(), dim=-1) @ F.normalize(left.float(), dim=-1).T
    right = F.normalize(right.float(), dim=-1) @ F.normalize(right.float(), dim=-1).T
    return float(F.cosine_similarity(left.flatten(), right.flatten(), dim=0).cpu())


@torch.inference_mode()
def diagnose_split(writer, cache, indices, device):
    rows, pair_rows, pooled = [], [], []
    by_position = {}
    for pair_index in tqdm(indices, desc="diagnose_token_writer", leave=False):
        pair = cache.load(pair_index)
        outputs = {}
        for variant in ("base", "counterfactual"):
            row = pair[variant]
            output = writer(row["key_flat"].to(device), row["value_flat"].to(device))
            outputs[variant] = output
            pooled.append(output["shared"].mean(0).cpu())
            for position, vector in enumerate(output["shared"].cpu()):
                by_position.setdefault(position, []).append(vector)
            rows.append({
                "pair_id": row["pair_id"], "variant": variant, "tokens": output["keys"].shape[0],
                "key_effective_rank": effective_rank(output["keys"]),
                "value_effective_rank": effective_rank(output["values"]),
                "shared_effective_rank": effective_rank(output["shared"]),
                "kv_linear_cka": linear_cka(output["keys"], output["values"]),
                "kv_relation_graph_agreement": graph_agreement(output["keys"], output["values"]),
            })
        alignment = pair["_stable_alignment"].to(device)
        unchanged = 1.0 - F.cosine_similarity(
            outputs["base"]["shared"][alignment[:, 0]],
            outputs["counterfactual"]["shared"][alignment[:, 1]], dim=-1,
        ).mean()
        changed = {}
        for variant in ("base", "counterfactual"):
            changed[variant] = outputs[variant]["shared"][pair[variant]["answer_mask"].to(device)].mean(0)
        changed_distance = 1.0 - F.cosine_similarity(changed["base"], changed["counterfactual"], dim=0)
        pair_rows.append({
            "pair_id": pair["base"]["pair_id"],
            "unchanged_span_cosine_distance": float(unchanged.cpu()),
            "changed_city_span_cosine_distance": float(changed_distance.cpu()),
        })
    pooled = F.normalize(torch.stack(pooled).float(), dim=-1)
    pooled_cosine = pooled @ pooled.T
    offdiag = pooled_cosine[~torch.eye(len(pooled), dtype=torch.bool)]
    position_stats = []
    for position, vectors in by_position.items():
        if len(vectors) < 4:
            continue
        matrix = torch.stack(vectors).float()
        normalized = F.normalize(matrix, dim=-1)
        cosine = normalized @ normalized.T
        off = cosine[~torch.eye(len(matrix), dtype=torch.bool)]
        position_stats.append({
            "position": position, "samples": len(matrix),
            "mean_dimension_variance": float(matrix.var(0, unbiased=False).mean()),
            "mean_cross_sample_cosine": float(off.mean()),
        })
    summary = {
        "samples": len(rows),
        "mean_pooled_cross_sample_cosine": float(offdiag.mean()),
        "mean_key_effective_rank": float(np.mean([row["key_effective_rank"] for row in rows])),
        "mean_value_effective_rank": float(np.mean([row["value_effective_rank"] for row in rows])),
        "mean_shared_effective_rank": float(np.mean([row["shared_effective_rank"] for row in rows])),
        "mean_kv_linear_cka": float(np.mean([row["kv_linear_cka"] for row in rows])),
        "mean_kv_relation_graph_agreement": float(np.mean([row["kv_relation_graph_agreement"] for row in rows])),
        "mean_unchanged_span_distance": float(np.mean([row["unchanged_span_cosine_distance"] for row in pair_rows])),
        "mean_changed_city_span_distance": float(np.mean([row["changed_city_span_cosine_distance"] for row in pair_rows])),
        "mean_same_position_variance": float(np.mean([row["mean_dimension_variance"] for row in position_stats])),
        "mean_same_position_cross_sample_cosine": float(np.mean([row["mean_cross_sample_cosine"] for row in position_stats])),
        "collapsed_by_p2i_threshold": bool(float(offdiag.mean()) > 0.998),
    }
    return rows, pair_rows, position_stats, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--projections", required=True)
    parser.add_argument("--writer-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    projections = torch.load(args.projections, map_location="cpu", weights_only=False)["pca"]
    checkpoint = torch.load(args.writer_checkpoint, map_location="cpu", weights_only=False)
    writer = TokenCanonicalWriter(projections, **checkpoint["writer_config"]).to(device).eval()
    writer.load_state_dict(checkpoint["writer"])
    for parameter in writer.parameters():
        parameter.requires_grad_(False)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    result = {"status": "complete", "splits": {}}
    for name, cache, indices in (
        ("validation", PairCache(args.train_index), range(448, 512)),
        ("test", PairCache(args.test_index), range(64)),
    ):
        rows, pairs, positions, summary = diagnose_split(writer, cache, indices, device)
        write_jsonl(output / f"{name}_per_sample.jsonl", rows)
        write_jsonl(output / f"{name}_pair_distances.jsonl", pairs)
        write_jsonl(output / f"{name}_position_stats.jsonl", positions)
        result["splits"][name] = summary
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
