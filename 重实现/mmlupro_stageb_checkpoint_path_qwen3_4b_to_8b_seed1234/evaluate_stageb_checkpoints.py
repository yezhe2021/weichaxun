from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from cache_store import cache_path, load_cache
from common import cuda, load_json, load_model, progress, read_jsonl, save_json, seed_all, write_jsonl
from receiver import prediction, trajectory
from writers import load_writer, make_writer


def kl(teacher: torch.Tensor, student: torch.Tensor) -> float:
    return F.kl_div(
        F.log_softmax(student.float(), dim=-1),
        F.log_softmax(teacher.float(), dim=-1),
        reduction="sum", log_target=True,
    ).item()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    root = Path(cfg["work_dir"])
    stage_b = root / "checkpoints/quick/full36_head8/stage_b_final"
    checkpoints = {
        "stage_a": Path(cfg["stage_a_checkpoint"]),
        "stage_b_lowest_validation": stage_b / "lowest_stage_b.pt",
        "stage_b_last": stage_b / "last.pt",
    }
    writers, metadata = {}, {}
    for name, path in checkpoints.items():
        writer = make_writer(cfg["writer_kind"], cfg).to(cuda()).eval()
        payload = load_writer(str(path), writer)
        writers[name] = writer
        metadata[name] = {key: value for key, value in payload.items() if key != "writer_state"}
    model = load_model(cfg["receiver_model"], cfg, frozen=True)
    samples = read_jsonl(Path(cfg["manifest_dir"]) / "test.jsonl")
    rows = []
    try:
        for index, sample in enumerate(samples, 1):
            source = load_cache(cache_path(cfg, cfg["sender_cache_family"], "test", sample["id"]), sample)
            target = load_cache(cache_path(cfg, cfg["receiver_cache_family"], "test", sample["id"]), sample)
            source_k, source_v = source["pre_key"].to(cuda()), source["value"].to(cuda())
            target_k, target_v = target["pre_key"].to(cuda()), target["value"].to(cuda())
            teacher = trajectory(model, sample, pre_key=target_k, value=target_v, output_hidden_states=False)
            teacher_index, teacher_label = prediction(teacher.choice_logits)
            native_accuracy = float(teacher_index == sample["gold_index"])
            for condition, writer in writers.items():
                pred_k, pred_v = writer(source_k, source_v)
                student = trajectory(model, sample, pre_key=pred_k, value=pred_v, output_hidden_states=False)
                student_index, student_label = prediction(student.choice_logits)
                rows.append({
                    "sample_id": sample["id"], "category": sample["category"],
                    "condition": condition,
                    "gold_index": sample["gold_index"], "gold_label": sample["gold_label"],
                    "native_prediction_index": teacher_index,
                    "native_prediction_label": teacher_label,
                    "prediction_index": student_index, "prediction_label": student_label,
                    "native_accuracy": native_accuracy,
                    "accuracy": float(student_index == sample["gold_index"]),
                    "native_agreement": float(student_index == teacher_index),
                    "final_full_vocab_kl": kl(teacher.logits[-1], student.logits[-1]),
                    "choice_only_kl": kl(teacher.choice_logits, student.choice_logits),
                    "choice_logit_mse_centered": F.mse_loss(
                        student.choice_logits.float() - student.choice_logits.float().mean(),
                        teacher.choice_logits.float() - teacher.choice_logits.float().mean(),
                    ).item(),
                })
            progress(f"checkpoint-path evaluation: {index}/{len(samples)}")
    finally:
        del model
        for writer in writers.values():
            del writer
        torch.cuda.empty_cache()
    conditions = {}
    for name in checkpoints:
        selected = [row for row in rows if row["condition"] == name]
        conditions[name] = {
            "count": len(selected),
            **{
                key: sum(row[key] for row in selected) / len(selected)
                for key in (
                    "accuracy", "native_accuracy", "native_agreement",
                    "final_full_vocab_kl", "choice_only_kl", "choice_logit_mse_centered",
                )
            },
            "checkpoint_metadata": metadata[name],
        }
    output = root / "artifacts/checkpoint_path_evaluation"
    write_jsonl(output / "per_sample.jsonl", rows)
    save_json(output / "summary.json", {
        "conditions": conditions,
        "comparison": {
            "lowest_minus_last_accuracy": conditions["stage_b_lowest_validation"]["accuracy"] - conditions["stage_b_last"]["accuracy"],
            "lowest_minus_last_native_agreement": conditions["stage_b_lowest_validation"]["native_agreement"] - conditions["stage_b_last"]["native_agreement"],
            "lowest_minus_last_final_full_vocab_kl": conditions["stage_b_lowest_validation"]["final_full_vocab_kl"] - conditions["stage_b_last"]["final_full_vocab_kl"],
        },
        "selection_uses_validation_only": True,
        "test_set_used_once_after_checkpoint_selection": True,
        "gold_used_only_for_evaluation": True,
    })


if __name__ == "__main__":
    main()
