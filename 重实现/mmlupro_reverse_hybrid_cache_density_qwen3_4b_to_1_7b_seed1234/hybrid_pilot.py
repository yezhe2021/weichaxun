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


PATTERNS = {
    "native": [],
    "half_interval_even": list(range(0, 28, 2)),
    "half_interval_odd": list(range(1, 28, 2)),
    "half_front": list(range(0, 14)),
    "half_back": list(range(14, 28)),
    "two_thirds_interval_wwn": [layer for layer in range(28) if layer % 3 != 2],
    "two_thirds_interval_wnw": [layer for layer in range(28) if layer % 3 != 1],
    "two_thirds_front": list(range(0, 19)),
    "two_thirds_back": list(range(9, 28)),
    "full_writer": list(range(28)),
}
GROUPS = {
    "half_14_of_28": ("half_interval_even", "half_interval_odd", "half_front", "half_back"),
    "two_thirds_19_of_28": (
        "two_thirds_interval_wwn", "two_thirds_interval_wnw",
        "two_thirds_front", "two_thirds_back",
    ),
}


def hybrid_cache(native_k: torch.Tensor, native_v: torch.Tensor,
                 writer_k: torch.Tensor, writer_v: torch.Tensor,
                 writer_layers: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    shapes = {tuple(x.shape) for x in (native_k, native_v, writer_k, writer_v)}
    target_layers = int(native_k.shape[0])
    if len(shapes) != 1 or native_k.ndim != 4 or target_layers != 28:
        raise ValueError(f"all cache tensors must have equal [28,T,H,D] shape; got {shapes}")
    if len(writer_layers) != len(set(writer_layers)) or any(layer < 0 or layer >= target_layers for layer in writer_layers):
        raise ValueError("Writer layer indices must be unique and within [0,27]")
    hybrid_k, hybrid_v = native_k.clone(), native_v.clone()
    if writer_layers:
        indices = torch.tensor(writer_layers, dtype=torch.long, device=native_k.device)
        hybrid_k.index_copy_(0, indices, writer_k.index_select(0, indices))
        hybrid_v.index_copy_(0, indices, writer_v.index_select(0, indices))
    selected = set(writer_layers)
    native_layers = [layer for layer in range(target_layers) if layer not in selected]
    if writer_layers:
        indices = torch.tensor(writer_layers, dtype=torch.long, device=native_k.device)
        if not torch.equal(hybrid_k.index_select(0, indices), writer_k.index_select(0, indices)):
            raise RuntimeError("hybrid K selected layers are not bitwise equal to Writer K")
        if not torch.equal(hybrid_v.index_select(0, indices), writer_v.index_select(0, indices)):
            raise RuntimeError("hybrid V selected layers are not bitwise equal to Writer V")
    if native_layers:
        indices = torch.tensor(native_layers, dtype=torch.long, device=native_k.device)
        if not torch.equal(hybrid_k.index_select(0, indices), native_k.index_select(0, indices)):
            raise RuntimeError("hybrid K retained layers are not bitwise equal to Native K")
        if not torch.equal(hybrid_v.index_select(0, indices), native_v.index_select(0, indices)):
            raise RuntimeError("hybrid V retained layers are not bitwise equal to Native V")
    return hybrid_k, hybrid_v


def distances(native_logits: torch.Tensor, condition_logits: torch.Tensor,
              choice_ids: torch.Tensor) -> dict[str, float | int]:
    native_logits, condition_logits = native_logits.float(), condition_logits.float()
    log_native = F.log_softmax(native_logits, dim=-1)
    log_condition = F.log_softmax(condition_logits, dim=-1)
    full_kl = torch.sum(log_native.exp() * (log_native - log_condition))
    native_choice = native_logits.index_select(0, choice_ids)
    condition_choice = condition_logits.index_select(0, choice_ids)
    log_native_choice = F.log_softmax(native_choice, dim=-1)
    log_condition_choice = F.log_softmax(condition_choice, dim=-1)
    choice_kl = torch.sum(log_native_choice.exp() * (log_native_choice - log_condition_choice))
    native_prediction, condition_prediction = int(native_choice.argmax()), int(condition_choice.argmax())
    return {
        "final_position_full_vocab_kl_vs_native": full_kl.item(),
        "choice_only_kl_vs_native": choice_kl.item(),
        "centered_choice_logit_mse": F.mse_loss(
            condition_choice - condition_choice.mean(), native_choice - native_choice.mean()
        ).item(),
        "native_choice_top1_agreement": float(condition_prediction == native_prediction),
        "prediction_index": condition_prediction,
    }


def stats(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    q = torch.quantile(tensor, torch.tensor([.10, .25, .50, .75, .90, .95], dtype=torch.float64))
    return {
        "count": len(values), "mean": tensor.mean().item(), "min": tensor.min().item(),
        "p10": q[0].item(), "p25": q[1].item(), "median": q[2].item(),
        "p75": q[3].item(), "p90": q[4].item(), "p95": q[5].item(), "max": tensor.max().item(),
    }


def margin_buckets(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["native"]["top1_top2_choice_logit_margin"])
    cuts = [0., .10, .25, .50, .75, .90, 1.]
    result = []
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        selected = ordered[round(lower * len(ordered)):round(upper * len(ordered))]
        if not selected:
            continue
        values = [row["conditions"][condition] for row in selected]
        result.append({
            "native_margin_rank_interval": f"p{int(lower*100)}-p{int(upper*100)}",
            "count": len(selected),
            "native_margin_mean": sum(row["native"]["top1_top2_choice_logit_margin"] for row in selected) / len(selected),
            "full_vocab_kl_mean": sum(x["final_position_full_vocab_kl_vs_native"] for x in values) / len(values),
            "choice_kl_mean": sum(x["choice_only_kl_vs_native"] for x in values) / len(values),
            "native_agreement": sum(x["native_choice_top1_agreement"] for x in values) / len(values),
            "accuracy_evaluation_only": sum(x["correct_evaluation_only"] for x in values) / len(values),
        })
    if sum(bucket["count"] for bucket in result) != len(rows):
        raise RuntimeError("margin buckets do not partition all samples")
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_fields = (
        "final_position_full_vocab_kl_vs_native", "choice_only_kl_vs_native",
        "centered_choice_logit_mse", "native_choice_top1_agreement",
    )
    conditions = {}
    for name in PATTERNS:
        selected = [row["conditions"][name] for row in rows]
        conditions[name] = {
            "writer_layer_count": len(PATTERNS[name]), "writer_layers": PATTERNS[name],
            **{field: stats([float(item[field]) for item in selected]) for field in metric_fields},
            "accuracy_evaluation_only": sum(item["correct_evaluation_only"] for item in selected) / len(selected),
            "margin_buckets": margin_buckets(rows, name),
        }
    native_accuracy = float(conditions["native"]["accuracy_evaluation_only"])
    full_accuracy = float(conditions["full_writer"]["accuracy_evaluation_only"])
    full_kl = float(conditions["full_writer"]["final_position_full_vocab_kl_vs_native"]["mean"])
    full_agreement = float(conditions["full_writer"]["native_choice_top1_agreement"]["mean"])
    groups = {}
    for group_name, names in GROUPS.items():
        means = {
            field: sum(float(conditions[name][field]["mean"]) for name in names) / len(names)
            for field in metric_fields
        }
        accuracy = sum(float(conditions[name]["accuracy_evaluation_only"]) for name in names) / len(names)
        groups[group_name] = {
            "conditions": list(names), "writer_layer_count": len(PATTERNS[names[0]]),
            "macro_mean": {**means, "accuracy_evaluation_only": accuracy},
            "strategy_spread": {
                field: max(float(conditions[name][field]["mean"]) for name in names)
                - min(float(conditions[name][field]["mean"]) for name in names)
                for field in metric_fields
            },
            "vs_full_writer": {
                "full_vocab_kl_reduction": full_kl - means["final_position_full_vocab_kl_vs_native"],
                "full_vocab_kl_gap_recovered_fraction": (
                    (full_kl - means["final_position_full_vocab_kl_vs_native"]) / full_kl if full_kl else None
                ),
                "native_agreement_gain": means["native_choice_top1_agreement"] - full_agreement,
                "native_agreement_gap_recovered_fraction": (
                    (means["native_choice_top1_agreement"] - full_agreement) / (1.0 - full_agreement)
                    if full_agreement < 1.0 else None
                ),
                "accuracy_gain": accuracy - full_accuracy,
                "native_accuracy_minus_group_accuracy": native_accuracy - accuracy,
            },
        }
    return {
        "sample_count": len(rows), "conditions": conditions,
        "density_groups": groups,
    }


@torch.inference_mode()
def run(cfg: dict[str, Any], limit: int | None, output_subdir: str) -> None:
    seed_all(cfg["seed"])
    samples = read_jsonl(Path(cfg["manifest_dir"]) / f"{cfg['split']}.jsonl")
    if limit is None and len(samples) != cfg["expected_samples"]:
        raise RuntimeError(f"formal sample count must be {cfg['expected_samples']}; got {len(samples)}")
    if limit is not None:
        samples = samples[:limit]
    model = load_model(cfg["model_1_7b"], cfg, frozen=True)
    writer = make_writer("full36", cfg).to(cuda()).eval()
    payload = load_writer(cfg["b1_checkpoint"], writer)
    writer.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()) or any(parameter.requires_grad for parameter in writer.parameters()):
        raise RuntimeError("pilot must keep Writer and Receiver frozen")
    if payload["writer_metadata"]["kind"] != "full36":
        raise RuntimeError("checkpoint is not a reverse Full-36 Writer")
    rows = []
    try:
        for index, sample in enumerate(samples, 1):
            source = load_cache(cache_path(cfg, cfg["sender_cache_family"], cfg["split"], sample["id"]), sample)
            target = load_cache(cache_path(cfg, cfg["receiver_cache_family"], cfg["split"], sample["id"]), sample)
            source_k, source_v = source["pre_key"].to(cuda()), source["value"].to(cuda())
            native_k, native_v = target["pre_key"].to(cuda()), target["value"].to(cuda())
            writer_k, writer_v = writer(source_k, source_v)
            native_output = trajectory(model, sample, "split_cache", native_k, native_v, output_hidden_states=False)
            native_logits = native_output.logits[-1].float()
            choice_ids = torch.tensor(sample["label_token_ids"], dtype=torch.long, device=native_logits.device)
            native_choice = native_logits.index_select(0, choice_ids)
            ordered = torch.topk(native_choice, 2)
            native_prediction = int(ordered.indices[0])
            row = {
                "sample_id": sample["id"], "category": sample["category"],
                "gold_index_evaluation_only": sample["gold_index"],
                "native": {"prediction_index": native_prediction,
                           "correct_evaluation_only": float(native_prediction == sample["gold_index"]),
                           "top1_top2_choice_logit_margin": (ordered.values[0] - ordered.values[1]).item()},
                "conditions": {},
            }
            for name, layers in PATTERNS.items():
                condition_k, condition_v = hybrid_cache(native_k, native_v, writer_k, writer_v, layers)
                if name == "native":
                    condition_logits = native_logits
                else:
                    output = trajectory(model, sample, "split_cache", condition_k, condition_v, output_hidden_states=False)
                    condition_logits = output.logits[-1].float()
                metrics = distances(native_logits, condition_logits, choice_ids)
                metrics["correct_evaluation_only"] = float(metrics["prediction_index"] == sample["gold_index"])
                row["conditions"][name] = metrics
            rows.append(row)
            progress(f"reverse hybrid density pilot: {index}/{len(samples)}")
        output = Path(cfg["work_dir"]) / "artifacts" / output_subdir
        write_jsonl(output / "per_sample.jsonl", rows)
        save_json(output / "summary.json", summarize(rows))
        save_json(output / "protocol.json", {
            "read_only": True, "gold_used_for_training_or_selection": False,
            "split": cfg["split"], "patterns": PATTERNS, "density_groups": GROUPS,
            "k_v_replaced_together": True, "cache_representation": "pre-RoPE K plus V",
            "identical_question_suffix_position_ids_attention_mask": True,
            "checkpoint": cfg["b1_checkpoint"], "writer_metadata": payload["writer_metadata"],
        })
    finally:
        del writer, model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-subdir", default="formal")
    args = parser.parse_args()
    run(load_json(args.config), args.limit, args.output_subdir)


if __name__ == "__main__":
    main()
