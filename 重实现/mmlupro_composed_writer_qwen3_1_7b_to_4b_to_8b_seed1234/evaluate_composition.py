from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cache_store import cache_path, load_cache
from common import cuda, load_json, load_model, progress, read_jsonl, save_json, seed_all, write_jsonl
from receiver import prediction, trajectory


def load_variant(spec: dict[str, Any]):
    module = importlib.import_module(spec["module"])
    variant_cfg = load_json(spec["config"])
    writer = module.make_writer(spec["kind"], variant_cfg).to(cuda()).eval()
    payload = module.load_writer(spec["checkpoint"], writer)
    metadata = {key: value for key, value in payload.items() if key != "writer_state"}
    return writer, metadata


def kl(teacher: torch.Tensor, student: torch.Tensor) -> float:
    return F.kl_div(
        F.log_softmax(student.float(), dim=-1),
        F.log_softmax(teacher.float(), dim=-1),
        reduction="sum", log_target=True,
    ).item()


def tensor_metrics(predicted: torch.Tensor, target: torch.Tensor):
    p, t = predicted.float(), target.float()
    per_layer_nmse = ((p - t).square().mean(dim=(1, 2, 3)) / t.square().mean(dim=(1, 2, 3)).clamp_min(1e-12))
    per_layer_cosine = F.cosine_similarity(p.flatten(1), t.flatten(1), dim=1)
    return {
        "nmse": per_layer_nmse.mean().item(),
        "cosine": per_layer_cosine.mean().item(),
        "per_layer_nmse": per_layer_nmse.cpu().tolist(),
        "per_layer_cosine": per_layer_cosine.cpu().tolist(),
    }


def validate_shape(tensor: torch.Tensor, layers: int, tokens: int, cfg: dict[str, Any], label: str):
    expected = (layers, tokens, cfg["num_kv_heads"], cfg["head_dim"])
    if tuple(tensor.shape) != expected:
        raise RuntimeError(f"{label} shape {tuple(tensor.shape)} != {expected}")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scope", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    samples = read_jsonl(Path(cfg["manifest_dir"]) / "test.jsonl")
    if args.limit is not None:
        samples = samples[:args.limit]
    elif args.scope == "smoke":
        samples = samples[:2]
    for spec in cfg["first_hop"] + cfg["second_hop"]:
        if not Path(spec["config"]).is_file() or not Path(spec["checkpoint"]).is_file():
            raise RuntimeError(f"missing Writer asset: {spec}")

    model = load_model(cfg["receiver_model"], cfg, frozen=True)
    teacher = {}
    rows, checkpoint_metadata = [], {}
    per_layer = {}
    try:
        for index, sample in enumerate(samples, 1):
            native8 = load_cache(cache_path(cfg, "source8", "test", sample["id"]), sample)
            native8_k, native8_v = native8["pre_key"].to(cuda()), native8["value"].to(cuda())
            output = trajectory(model, sample, pre_key=native8_k, value=native8_v, output_hidden_states=False)
            pred_index, pred_label = prediction(output.choice_logits)
            teacher[sample["id"]] = {
                "final_logits": output.logits[-1].half().cpu(),
                "choice_logits": output.choice_logits.half().cpu(),
                "prediction_index": pred_index,
                "prediction_label": pred_label,
            }
            rows.append({
                "sample_id": sample["id"], "condition": "native8",
                "gold_index": sample["gold_index"], "gold_label": sample["gold_label"],
                "prediction_index": pred_index, "prediction_label": pred_label,
                "accuracy": float(pred_index == sample["gold_index"]), "native_agreement": 1.0,
                "final_full_vocab_kl": 0.0, "choice_only_kl": 0.0,
                "choice_logit_mse_centered": 0.0,
            })
            progress(f"Native8 teacher: {index}/{len(samples)}")

        def evaluate_condition(condition: str, second, first=None):
            layer_sums = {name: torch.zeros(36) for name in ("k_nmse", "k_cosine", "v_nmse", "v_cosine")}
            intermediate_sums = {name: 0.0 for name in ("k_nmse", "k_cosine", "v_nmse", "v_cosine")}
            for index, sample in enumerate(samples, 1):
                native4 = load_cache(cache_path(cfg, "target4", "test", sample["id"]), sample)
                native8 = load_cache(cache_path(cfg, "source8", "test", sample["id"]), sample)
                native4_k, native4_v = native4["pre_key"].to(cuda()), native4["value"].to(cuda())
                native8_k, native8_v = native8["pre_key"].to(cuda()), native8["value"].to(cuda())
                if first is None:
                    middle_k, middle_v = native4_k, native4_v
                else:
                    native17 = load_cache(cache_path(cfg, "source17", "test", sample["id"]), sample)
                    source_k, source_v = native17["pre_key"].to(cuda()), native17["value"].to(cuda())
                    middle_k, middle_v = first(source_k, source_v)
                    validate_shape(middle_k, 36, sample["context_length"], cfg, "translated4 K")
                    for prefix, predicted, target in (("k", middle_k, native4_k), ("v", middle_v, native4_v)):
                        metric = tensor_metrics(predicted, target)
                        intermediate_sums[prefix + "_nmse"] += metric["nmse"]
                        intermediate_sums[prefix + "_cosine"] += metric["cosine"]
                predicted_k, predicted_v = second(middle_k, middle_v)
                validate_shape(predicted_k, 36, sample["context_length"], cfg, "translated8 K")
                final_metrics = {}
                for prefix, predicted, target in (("k", predicted_k, native8_k), ("v", predicted_v, native8_v)):
                    metric = tensor_metrics(predicted, target)
                    final_metrics[prefix + "_nmse"] = metric["nmse"]
                    final_metrics[prefix + "_cosine"] = metric["cosine"]
                    layer_sums[prefix + "_nmse"] += torch.tensor(metric["per_layer_nmse"])
                    layer_sums[prefix + "_cosine"] += torch.tensor(metric["per_layer_cosine"])
                output = trajectory(model, sample, pre_key=predicted_k, value=predicted_v, output_hidden_states=False)
                pred_index, pred_label = prediction(output.choice_logits)
                native = teacher[sample["id"]]
                teacher_final = native["final_logits"].to(cuda())
                teacher_choice = native["choice_logits"].to(cuda())
                rows.append({
                    "sample_id": sample["id"], "condition": condition,
                    "gold_index": sample["gold_index"], "gold_label": sample["gold_label"],
                    "prediction_index": pred_index, "prediction_label": pred_label,
                    "native_prediction_index": native["prediction_index"],
                    "accuracy": float(pred_index == sample["gold_index"]),
                    "native_agreement": float(pred_index == native["prediction_index"]),
                    "final_full_vocab_kl": kl(teacher_final, output.logits[-1]),
                    "choice_only_kl": kl(teacher_choice, output.choice_logits),
                    "choice_logit_mse_centered": F.mse_loss(
                        output.choice_logits.float() - output.choice_logits.float().mean(),
                        teacher_choice.float() - teacher_choice.float().mean(),
                    ).item(),
                    **{"final_" + key: value for key, value in final_metrics.items()},
                })
                progress(f"{condition}: {index}/{len(samples)}")
            per_layer[condition] = {name: (values / len(samples)).tolist() for name, values in layer_sums.items()}
            if first is not None:
                per_layer[condition]["intermediate_mean"] = {
                    name: value / len(samples) for name, value in intermediate_sums.items()
                }

        for second_spec in cfg["second_hop"]:
            second, second_meta = load_variant(second_spec)
            checkpoint_metadata[second_spec["name"]] = second_meta
            try:
                evaluate_condition("native4_to8__" + second_spec["name"], second)
                for first_spec in cfg["first_hop"]:
                    first, first_meta = load_variant(first_spec)
                    checkpoint_metadata[first_spec["name"]] = first_meta
                    try:
                        evaluate_condition(
                            "chain__" + first_spec["name"] + "__" + second_spec["name"],
                            second, first,
                        )
                    finally:
                        del first
                        torch.cuda.empty_cache()
            finally:
                del second
                torch.cuda.empty_cache()

        for index, sample in enumerate(samples, 1):
            native8 = load_cache(cache_path(cfg, "source8", "test", sample["id"]), sample)
            zero_k = torch.zeros_like(native8["pre_key"], device=cuda())
            zero_v = torch.zeros_like(native8["value"], device=cuda())
            output = trajectory(model, sample, pre_key=zero_k, value=zero_v, output_hidden_states=False)
            pred_index, pred_label = prediction(output.choice_logits)
            native = teacher[sample["id"]]
            rows.append({
                "sample_id": sample["id"], "condition": "zero8",
                "gold_index": sample["gold_index"], "gold_label": sample["gold_label"],
                "prediction_index": pred_index, "prediction_label": pred_label,
                "native_prediction_index": native["prediction_index"],
                "accuracy": float(pred_index == sample["gold_index"]),
                "native_agreement": float(pred_index == native["prediction_index"]),
                "final_full_vocab_kl": kl(native["final_logits"].to(cuda()), output.logits[-1]),
                "choice_only_kl": kl(native["choice_logits"].to(cuda()), output.choice_logits),
                "choice_logit_mse_centered": F.mse_loss(
                    output.choice_logits.float() - output.choice_logits.float().mean(),
                    native["choice_logits"].to(cuda()).float() - native["choice_logits"].to(cuda()).float().mean(),
                ).item(),
            })
            progress(f"zero8: {index}/{len(samples)}")
    finally:
        del model
        torch.cuda.empty_cache()

    conditions = {}
    metric_names = ("accuracy", "native_agreement", "final_full_vocab_kl", "choice_only_kl", "choice_logit_mse_centered")
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        conditions[condition] = {"count": len(selected), **{
            metric: sum(row[metric] for row in selected) / len(selected) for metric in metric_names
        }}
        for metric in ("final_k_nmse", "final_k_cosine", "final_v_nmse", "final_v_cosine"):
            if metric in selected[0]:
                conditions[condition][metric] = sum(row[metric] for row in selected) / len(selected)
    composition_penalties = {}
    for second in cfg["second_hop"]:
        baseline = conditions["native4_to8__" + second["name"]]
        for first in cfg["first_hop"]:
            name = "chain__" + first["name"] + "__" + second["name"]
            chain = conditions[name]
            composition_penalties[name] = {
                "delta_accuracy": chain["accuracy"] - baseline["accuracy"],
                "delta_native_agreement": chain["native_agreement"] - baseline["native_agreement"],
                "delta_final_full_vocab_kl": chain["final_full_vocab_kl"] - baseline["final_full_vocab_kl"],
                "kl_amplification_ratio": chain["final_full_vocab_kl"] / max(baseline["final_full_vocab_kl"], 1e-12),
            }
    root = Path(cfg["work_dir"]) / "artifacts" / args.scope
    write_jsonl(root / "per_sample.jsonl", rows)
    save_json(root / "per_layer_metrics.json", per_layer)
    save_json(root / "summary.json", {
        "scope": args.scope, "sample_count": len(samples), "conditions": conditions,
        "composition_penalties": composition_penalties,
        "checkpoint_metadata": checkpoint_metadata,
        "writer_training_performed": False,
        "intermediate_kv_persisted": False,
        "gold_used_only_for_evaluation": True,
    })


if __name__ == "__main__":
    main()
