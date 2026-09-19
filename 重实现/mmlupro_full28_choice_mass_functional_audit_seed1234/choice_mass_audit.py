from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cache_store import cache_path, load_cache
from common import cuda, load_json, load_model, progress, read_jsonl, save_json, seed_all, write_jsonl
from receiver import trajectory
from writers import load_writer, make_writer


def distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=torch.float64)
    q = torch.quantile(tensor, quantiles)
    return {
        "count": len(values), "mean": tensor.mean().item(), "min": tensor.min().item(),
        "p10": q[0].item(), "p25": q[1].item(), "median": q[2].item(),
        "p75": q[3].item(), "p90": q[4].item(), "p95": q[5].item(),
        "max": tensor.max().item(),
    }


def functional_distances(native_logits: torch.Tensor, writer_logits: torch.Tensor,
                         choice_ids: torch.Tensor, temperature: float = 1.0) -> dict[str, float]:
    """Exact final-position distances. Gold labels are deliberately absent."""
    if temperature != 1.0:
        raise ValueError("the exact probability-chain decomposition is locked to temperature=1")
    native_logits, writer_logits = native_logits.float(), writer_logits.float()
    choice_ids = choice_ids.to(device=native_logits.device, dtype=torch.long)
    if native_logits.ndim != 1 or writer_logits.shape != native_logits.shape:
        raise ValueError("native/writer final logits must be equal one-dimensional vocab vectors")
    if choice_ids.ndim != 1 or choice_ids.numel() < 2 or choice_ids.unique().numel() != choice_ids.numel():
        raise ValueError("choice token IDs must be a unique one-dimensional vector")

    log_pn, log_pw = F.log_softmax(native_logits, dim=-1), F.log_softmax(writer_logits, dim=-1)
    pn = log_pn.exp()
    full_kl = torch.sum(pn * (log_pn - log_pw))
    native_choice, writer_choice = native_logits[choice_ids], writer_logits[choice_ids]
    log_qn, log_qw = F.log_softmax(native_choice, dim=-1), F.log_softmax(writer_choice, dim=-1)
    qn = log_qn.exp()
    choice_kl = torch.sum(qn * (log_qn - log_qw))

    log_mn = torch.logsumexp(native_choice, dim=-1) - torch.logsumexp(native_logits, dim=-1)
    log_mw = torch.logsumexp(writer_choice, dim=-1) - torch.logsumexp(writer_logits, dim=-1)
    mn, mw = log_mn.exp(), log_mw.exp()
    eps = torch.finfo(torch.float32).eps
    mn_safe, mw_safe = mn.clamp(eps, 1.0 - eps), mw.clamp(eps, 1.0 - eps)
    mass_kl = mn_safe * (mn_safe.log() - mw_safe.log()) + (1.0 - mn_safe) * (
        torch.log1p(-mn_safe) - torch.log1p(-mw_safe)
    )
    choice_contribution = mn * choice_kl
    # Compute the non-choice term independently from joint token contributions;
    # do not define it as a residual, otherwise the decomposition check is tautological.
    choice_joint_kl = torch.sum(pn[choice_ids] * (log_pn[choice_ids] - log_pw[choice_ids]))
    nonchoice_joint_kl = full_kl - choice_joint_kl
    nonchoice_mass_kl = (1.0 - mn_safe) * (
        torch.log1p(-mn_safe) - torch.log1p(-mw_safe)
    )
    nonchoice_contribution = nonchoice_joint_kl - nonchoice_mass_kl
    reconstructed = mass_kl + choice_contribution + nonchoice_contribution

    native_centered = native_choice - native_choice.mean()
    writer_centered = writer_choice - writer_choice.mean()
    ordered = torch.topk(native_choice, k=2)
    native_top1, native_top2 = int(ordered.indices[0]), int(ordered.indices[1])
    writer_top1 = int(writer_choice.argmax())
    native_margin = native_choice[native_top1] - native_choice[native_top2]
    writer_native_pair_margin = writer_choice[native_top1] - writer_choice[native_top2]
    entropy = -torch.sum(qn * log_qn)
    return {
        "full_vocab_kl": full_kl.item(), "choice_only_kl": choice_kl.item(),
        "centered_choice_logit_mse": F.mse_loss(writer_centered, native_centered).item(),
        "native_choice_mass": mn.item(), "writer_choice_mass": mw.item(),
        "choice_vs_nonchoice_mass_kl": mass_kl.item(),
        "weighted_choice_kl_contribution": choice_contribution.item(),
        "nonchoice_conditional_kl_contribution": nonchoice_contribution.item(),
        "decomposition_error": abs((full_kl - reconstructed).item()),
        "native_choice_entropy": entropy.item(),
        "native_choice_entropy_normalized": (entropy / math.log(choice_ids.numel())).item(),
        "native_top1_top2_logit_margin": native_margin.item(),
        "native_top1_top2_probability_margin": (qn[native_top1] - qn[native_top2]).item(),
        "writer_margin_on_native_top_pair": writer_native_pair_margin.item(),
        "native_top1_index": native_top1, "writer_top1_index": writer_top1,
        "native_top1_agreement": float(native_top1 == writer_top1),
        "native_top_pair_flipped": float(writer_native_pair_margin < 0),
    }


def margin_buckets(rows: list[dict[str, Any]], checkpoint: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["native"]["top1_top2_logit_margin"])
    cuts = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    output = []
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        begin = int(round(lower * len(ordered)))
        end = int(round(upper * len(ordered)))
        selected = ordered[begin:end]
        if not selected:
            continue
        metrics = [row["checkpoints"][checkpoint] for row in selected]
        output.append({
            "rank_interval": f"p{int(lower*100)}-p{int(upper*100)}", "count": len(selected),
            "native_margin": distribution_stats([row["native"]["top1_top2_logit_margin"] for row in selected]),
            "native_top1_agreement": sum(x["native_top1_agreement"] for x in metrics) / len(metrics),
            "native_top_pair_flip_rate": sum(x["native_top_pair_flipped"] for x in metrics) / len(metrics),
            "choice_only_kl": sum(x["choice_only_kl"] for x in metrics) / len(metrics),
            "centered_choice_logit_mse": sum(x["centered_choice_logit_mse"] for x in metrics) / len(metrics),
        })
    if sum(bucket["count"] for bucket in output) != len(rows):
        raise RuntimeError("margin bucket partition lost samples")
    return output


def aggregate_split(rows: list[dict[str, Any]], checkpoint_names: list[str]) -> dict[str, Any]:
    native_fields = ["choice_mass", "choice_entropy", "choice_entropy_normalized",
                     "top1_top2_logit_margin", "top1_top2_probability_margin"]
    result: dict[str, Any] = {
        "count": len(rows),
        "native": {field: distribution_stats([row["native"][field] for row in rows]) for field in native_fields},
        "checkpoints": {},
    }
    metric_fields = [
        "full_vocab_kl", "choice_only_kl", "centered_choice_logit_mse", "writer_choice_mass",
        "choice_vs_nonchoice_mass_kl", "weighted_choice_kl_contribution",
        "nonchoice_conditional_kl_contribution", "native_top1_agreement", "native_top_pair_flipped",
    ]
    for name in checkpoint_names:
        metrics = [row["checkpoints"][name] for row in rows]
        full_sum = sum(x["full_vocab_kl"] for x in metrics)
        checkpoint_summary = {
            field: distribution_stats([x[field] for x in metrics]) for field in metric_fields
        }
        checkpoint_summary["contribution_share_of_full_kl"] = {
            "choice_vs_nonchoice_mass": sum(x["choice_vs_nonchoice_mass_kl"] for x in metrics) / full_sum,
            "weighted_choice": sum(x["weighted_choice_kl_contribution"] for x in metrics) / full_sum,
            "nonchoice_conditional": sum(x["nonchoice_conditional_kl_contribution"] for x in metrics) / full_sum,
        }
        checkpoint_summary["margin_buckets"] = margin_buckets(rows, name)
        checkpoint_summary["accuracy_evaluation_only"] = {
            "native": sum(row["native"]["correct"] for row in rows) / len(rows),
            "writer": sum(row["checkpoints"][name]["correct"] for row in rows) / len(rows),
        }
        result["checkpoints"][name] = checkpoint_summary
    return result


def load_frozen_writer(cfg: dict[str, Any], path: str):
    writer = make_writer("full28", cfg).to(cuda()).eval()
    payload = load_writer(path, writer)
    writer.requires_grad_(False)
    if any(parameter.requires_grad for parameter in writer.parameters()):
        raise RuntimeError("audit Writer must be frozen")
    metadata = payload["writer_metadata"]
    if metadata["kind"] != "full28" or not metadata["per_target_layer_independent"]:
        raise RuntimeError("checkpoint is not the required independent Full-28 Writer")
    return writer, {key: value for key, value in payload.items() if key != "writer_state"}


@torch.inference_mode()
def audit(cfg: dict[str, Any], splits: list[str], limit: int | None, output_root: Path) -> None:
    seed_all(cfg["seed"])
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Receiver must be frozen")
    writers, metadata = {}, {}
    for name, path in cfg["checkpoints"].items():
        writers[name], metadata[name] = load_frozen_writer(cfg, path)
    all_summaries, protocol = {}, {
        "read_only": True, "gold_used_for_loss_or_selection": False,
        "condition": "final-position logits after Native4 or translated Full-28 KV plus identical question suffix",
        "temperature": cfg["temperature"], "checkpoints": metadata,
    }
    try:
        for split in splits:
            samples = read_jsonl(Path(cfg["manifest_dir"]) / f"{split}.jsonl")
            if limit is not None:
                samples = samples[:limit]
            rows = []
            # MMLU-Pro samples may expose fewer than ten options. Token IDs must
            # be stable for the same option count, not identical across counts.
            choice_ids_by_count: dict[int, tuple[int, ...]] = {}
            for index, sample in enumerate(samples, 1):
                choice_ids_cpu = torch.tensor(sample["label_token_ids"], dtype=torch.long)
                if choice_ids_cpu.numel() != sample["num_options"] or choice_ids_cpu.unique().numel() != choice_ids_cpu.numel():
                    raise RuntimeError(f"invalid choice token IDs for {sample['id']}")
                current_ids = tuple(choice_ids_cpu.tolist())
                option_count = int(sample["num_options"])
                expected_ids = choice_ids_by_count.setdefault(option_count, current_ids)
                if current_ids != expected_ids:
                    raise RuntimeError(f"choice token IDs changed for num_options={option_count}")
                source = load_cache(cache_path(cfg, "source17", split, sample["id"]), sample)
                target = load_cache(cache_path(cfg, "target4", split, sample["id"]), sample)
                source_k, source_v = source["pre_key"].to(cuda()), source["value"].to(cuda())
                target_k, target_v = target["pre_key"].to(cuda()), target["value"].to(cuda())
                native_output = trajectory(model, sample, "split_cache", target_k, target_v, output_hidden_states=False)
                native_logits = native_output.logits[-1].float()
                choice_ids = choice_ids_cpu.to(native_logits.device)
                exact_native_choice = native_logits.index_select(0, choice_ids)
                if not torch.equal(exact_native_choice, native_output.choice_logits):
                    raise RuntimeError("choice logits are not exact gathers from final full-vocab logits")
                native_q = F.softmax(exact_native_choice, dim=-1)
                ordered = torch.topk(exact_native_choice, 2)
                native_top1 = int(ordered.indices[0])
                native = {
                    "choice_mass": (F.log_softmax(native_logits, dim=-1)[choice_ids].exp().sum()).item(),
                    "choice_entropy": (-torch.sum(native_q * F.log_softmax(exact_native_choice, dim=-1))).item(),
                    "choice_entropy_normalized": (-torch.sum(native_q * F.log_softmax(exact_native_choice, dim=-1)) / math.log(choice_ids.numel())).item(),
                    "top1_top2_logit_margin": (ordered.values[0] - ordered.values[1]).item(),
                    "top1_top2_probability_margin": (native_q[ordered.indices[0]] - native_q[ordered.indices[1]]).item(),
                    "prediction_index": native_top1, "correct": float(native_top1 == sample["gold_index"]),
                }
                row = {"sample_id": sample["id"], "split": split, "category": sample["category"],
                       "gold_index_evaluation_only": sample["gold_index"], "native": native, "checkpoints": {}}
                for name, writer in writers.items():
                    predicted_k, predicted_v = writer(source_k, source_v)
                    writer_output = trajectory(model, sample, "split_cache", predicted_k, predicted_v, output_hidden_states=False)
                    metrics = functional_distances(native_logits, writer_output.logits[-1].float(), choice_ids, cfg["temperature"])
                    if metrics["decomposition_error"] > cfg["decomposition_tolerance"]:
                        raise RuntimeError(f"KL decomposition failed for {sample['id']}: {metrics['decomposition_error']}")
                    metrics["correct"] = float(metrics["writer_top1_index"] == sample["gold_index"])
                    row["checkpoints"][name] = metrics
                rows.append(row)
                progress(f"choice-mass audit {split}: {index}/{len(samples)}")
            write_jsonl(output_root / "per_sample" / f"{split}.jsonl", rows)
            all_summaries[split] = aggregate_split(rows, list(writers))
            all_summaries[split]["choice_token_ids_by_num_options"] = {
                str(count): list(ids) for count, ids in sorted(choice_ids_by_count.items())
            }
            save_json(output_root / "summary.partial.json", all_summaries)
        save_json(output_root / "summary.json", all_summaries)
        save_json(output_root / "protocol.json", protocol)
    finally:
        del writers, model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-subdir", default="formal")
    args = parser.parse_args()
    cfg = load_json(args.config)
    splits = args.splits or cfg["splits"]
    audit(cfg, splits, args.limit, Path(cfg["work_dir"]) / "artifacts" / args.output_subdir)


if __name__ == "__main__":
    main()
