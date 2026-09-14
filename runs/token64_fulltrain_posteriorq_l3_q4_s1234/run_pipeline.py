from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BOOTSTRAP_CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
BASELINE_ROOT = Path(BOOTSTRAP_CONFIG["comparison_32_root"])
sys.path.insert(1, str(BASELINE_ROOT))

import common

common.ROOT = ROOT
from common import configuration, load_model, run_root, save_json, seed_all
from data import cache_llama, cache_qwen_and_pairs, load_pair, load_source, manifest_rows, prepare_manifests
from experiment import (
    evaluate_native_baselines,
    evaluate_translated,
    load_checkpoint,
    prepare_oracle_teachers,
    select_stage_a,
    train_stage_a,
    train_stage_b,
)


STAGES = (
    "prepare", "cache_llama", "cache_qwen", "phase0_audit", "teachers_baselines",
    "stage_a", "select_stage_a", "stage_b", "evaluate",
)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def paired_counts(name_a, correctness_a, name_b, correctness_b):
    if set(correctness_a) != set(correctness_b):
        raise RuntimeError(f"Paired ID mismatch: {name_a} vs {name_b}")
    ids = sorted(correctness_a, key=str)
    groups = {"both_correct": [], "only_a_correct": [], "only_b_correct": [], "both_wrong": []}
    diffs = []
    for sample_id in ids:
        a, b = bool(correctness_a[sample_id]), bool(correctness_b[sample_id])
        diffs.append(int(a) - int(b))
        if a and b:
            groups["both_correct"].append(sample_id)
        elif a:
            groups["only_a_correct"].append(sample_id)
        elif b:
            groups["only_b_correct"].append(sample_id)
        else:
            groups["both_wrong"].append(sample_id)
    discordant = len(groups["only_a_correct"]) + len(groups["only_b_correct"])
    if discordant:
        smaller = min(len(groups["only_a_correct"]), len(groups["only_b_correct"]))
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
        mcnemar_exact_p = min(1.0, 2 * tail)
    else:
        mcnemar_exact_p = 1.0
    mean = sum(diffs) / len(diffs)
    variance = sum((value - mean) ** 2 for value in diffs) / max(len(diffs) - 1, 1)
    half_width = 1.96 * math.sqrt(variance / len(diffs))
    return {
        "a": name_a,
        "b": name_b,
        "samples": len(ids),
        "a_correct": sum(correctness_a.values()),
        "b_correct": sum(correctness_b.values()),
        "both_correct": len(groups["both_correct"]),
        "only_a_correct": len(groups["only_a_correct"]),
        "only_b_correct": len(groups["only_b_correct"]),
        "both_wrong": len(groups["both_wrong"]),
        "prediction_accuracy_difference_a_minus_b": mean,
        "difference_95pct_normal_ci": [mean - half_width, mean + half_width],
        "mcnemar_exact_two_sided_p": mcnemar_exact_p,
        "sample_ids": groups,
    }


def correctness_from_records(records, condition):
    return {row["id"]: int(row["conditions"][condition]["accuracy"]) for row in records}


def phase0_audit(cfg):
    rows = manifest_rows(cfg, "test")[:min(cfg["audit_samples"], cfg["test_samples"])]
    records = []
    for row in rows:
        source, pair = load_source(cfg, "test", row), load_pair(cfg, "test", row)
        anchors = pair["metadata"]["anchors"]
        source_indices = source["metadata"]["selected_indices"]
        target_indices = pair["metadata"]["target_indices"][1:]
        if len(anchors) != cfg["regions"] or len(source_indices) != cfg["regions"] or len(target_indices) != cfg["regions"]:
            raise RuntimeError(f"64-token audit count failure for {row['id']}")
        records.append({
            "id": row["id"],
            "selected_slots": len(anchors),
            "unique_source_tokens": len(set(source_indices)),
            "unique_target_tokens": len(set(target_indices)),
            "selected_attention_mass": source["metadata"]["selected_attention_mass"],
        })
    audit = {
        "passed": True,
        "sample_count": len(records),
        "requested_slots": cfg["regions"],
        "mean_unique_source_tokens": sum(row["unique_source_tokens"] for row in records) / len(records),
        "mean_unique_target_tokens": sum(row["unique_target_tokens"] for row in records) / len(records),
        "mean_selected_attention_mass": sum(row["selected_attention_mass"] for row in records) / len(records),
        "records": records,
    }
    save_json(run_root(cfg) / "audit" / "token64_selection_audit.json", audit)
    print(f"TOKEN64 PHASE-0 AUDIT PASSED: {audit}", flush=True)


def append_results(cfg, comparison):
    document = Path(cfg["shared_results_document"])
    tag = ROOT.name
    begin, end = f"<!-- BEGIN {tag} -->", f"<!-- END {tag} -->"
    table = comparison["main_table"]
    section = "\n".join([
        begin,
        f"## {tag}",
        "",
        "Machine-generated after successful completion. Normal-budget 64-token Translator training.",
        "",
        "| Condition | Accuracy | Oracle agreement | Choice KL |",
        "|---|---:|---:|---:|",
        *[
            f"| {name} | {100 * value['accuracy']:.2f}% | {100 * value.get('oracle_agreement', 0):.2f}% | {value.get('oracle_choice_kl', 0):.6f} |"
            for name, value in table.items()
        ],
        "",
        f"Training: {cfg['train_samples']} train, Stage A {cfg['stage_a_steps']} steps, Stage B {cfg['stage_b_steps']} steps.",
        end,
        "",
    ])
    current = document.read_text(encoding="utf-8") if document.exists() else "# Experiment Results\n\n"
    if begin in current and end in current:
        left, tail = current.split(begin, 1)
        _, right = tail.split(end, 1)
        current = left.rstrip() + "\n\n" + section + right.lstrip("\n")
    else:
        current = current.rstrip() + "\n\n" + section
    document.write_text(current, encoding="utf-8")


def finalize(cfg):
    model = load_model(cfg, "qwen")
    try:
        baselines_path = run_root(cfg) / "evaluation" / "baselines" / "summary.json"
        baselines = json.loads(baselines_path.read_text(encoding="utf-8")) if baselines_path.exists() else evaluate_native_baselines(cfg, model)
        training = json.loads((run_root(cfg) / "stage_b" / "training_summary.json").read_text(encoding="utf-8"))
        translated = {}
        for alias in ("best_accuracy", "best_choice_kl"):
            module, payload = load_checkpoint(cfg, training[alias]["path"])
            translated[alias] = {"checkpoint_step": payload["step"], **evaluate_translated(cfg, model, module, alias)}
            del module
            import torch
            torch.cuda.empty_cache()
    finally:
        del model
        import torch
        torch.cuda.empty_cache()

    primary = translated["best_accuracy"]["functional"]
    main_table = {
        "Qwen Full Native": baselines["qwen_full_native"],
        "Qwen Self-Selected64 Native": baselines["qwen_self_selected32_native"],
        "Llama-selected64 -> Qwen Native Oracle": baselines["llama_selected_qwen_native_oracle"],
        "Translated64 Full28-Diagonal": primary["translated"],
        "Shuffle64": primary["shuffle"],
        "Zero64": baselines["zero"],
        "No-memory": baselines["no_memory"],
    }

    new_translated_records = load_jsonl(
        run_root(cfg) / "evaluation" / "translated" / "best_accuracy" / "per_sample_metrics.jsonl"
    )
    new_baseline_records = load_jsonl(run_root(cfg) / "evaluation" / "baselines" / "per_sample_metrics.jsonl")
    old_root = Path(cfg["comparison_32_root"]) / "runs" / "study" / "evaluation"
    old_translated_records = load_jsonl(old_root / "translated" / "best_accuracy" / "per_sample_metrics.jsonl")
    old_baseline_records = load_jsonl(old_root / "baselines" / "per_sample_metrics.jsonl")
    current_ids = {row["id"] for row in new_translated_records}
    old_translated_records = [row for row in old_translated_records if row["id"] in current_ids]
    old_baseline_records = [row for row in old_baseline_records if row["id"] in current_ids]
    comparisons = {
        "translated64_vs_native64_oracle": paired_counts(
            "Translated64",
            correctness_from_records(new_translated_records, "translated"),
            "Native64Oracle",
            correctness_from_records(new_baseline_records, "llama_selected_qwen_native_oracle"),
        ),
        "translated64_vs_translated32": paired_counts(
            "Translated64",
            correctness_from_records(new_translated_records, "translated"),
            "Translated32",
            correctness_from_records(old_translated_records, "translated"),
        ),
        "native64_oracle_vs_native32_oracle": paired_counts(
            "Native64Oracle",
            correctness_from_records(new_baseline_records, "llama_selected_qwen_native_oracle"),
            "Native32Oracle",
            correctness_from_records(old_baseline_records, "llama_selected_qwen_native_oracle"),
        ),
    }
    save_json(run_root(cfg) / "results" / "paired_comparisons.json", comparisons)
    comparison = {
        "experiment": ROOT.name,
        "status": "completed",
        "pilot": False,
        "training_budget": {
            "train_samples": cfg["train_samples"],
            "validation_samples": cfg["validation_samples"],
            "test_samples": cfg["test_samples"],
            "stage_a_steps": cfg["stage_a_steps"],
            "stage_b_steps": cfg["stage_b_steps"],
        },
        "selection": "64 option-only normalized raw-text regions",
        "sender_prompt": "Question -> Options -> repeated Question",
        "receiver": "native Question + translated64 Options KV + native Answer:",
        "checkpoint_selection": "validation only; primary=best validation accuracy",
        "main_table": main_table,
        "translated_checkpoints": translated,
        "paired_comparisons_file": "paired_comparisons.json",
    }
    save_json(run_root(cfg) / "results" / "comparison.json", comparison)
    append_results(cfg, comparison)
    print("TOKEN64 FULL-TRAIN EXPERIMENT COMPLETED", flush=True)


def execute_stage(cfg, stage):
    if stage == "prepare":
        prepare_manifests(cfg)
    elif stage == "cache_llama":
        cache_llama(cfg)
    elif stage == "cache_qwen":
        cache_qwen_and_pairs(cfg)
    elif stage == "phase0_audit":
        phase0_audit(cfg)
    elif stage == "teachers_baselines":
        model = load_model(cfg, "qwen")
        try:
            prepare_oracle_teachers(cfg, model)
            evaluate_native_baselines(cfg, model)
        finally:
            import torch
            del model
            torch.cuda.empty_cache()
    elif stage == "stage_a":
        train_stage_a(cfg)
    elif stage == "select_stage_a":
        model = load_model(cfg, "qwen")
        try:
            prepare_oracle_teachers(cfg, model)
            select_stage_a(cfg, model)
        finally:
            import torch
            del model
            torch.cuda.empty_cache()
    elif stage == "stage_b":
        model = load_model(cfg, "qwen")
        try:
            train_stage_b(cfg, model)
        finally:
            import torch
            del model
            torch.cuda.empty_cache()
    elif stage == "evaluate":
        finalize(cfg)
    else:
        raise ValueError(stage)


def done_path(cfg, stage):
    return run_root(cfg) / "stage_status" / f"{stage}.json"


def run_stage(cfg, stage):
    save_json(run_root(cfg) / "status.json", {"status": "running", "stage": stage, "pid": os.getpid()})
    try:
        execute_stage(cfg, stage)
        save_json(done_path(cfg, stage), {"signature": cfg["signature"], "status": "completed", "stage": stage})
        save_json(run_root(cfg) / "status.json", {
            "status": "completed", "stage": stage, "pid": os.getpid()
        })
    except BaseException as error:
        save_json(run_root(cfg) / "status.json", {
            "status": "failed", "stage": stage,
            "error": f"{type(error).__name__}: {error}", "pid": os.getpid(),
        })
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    parser.add_argument("action", choices=("all", *STAGES), default="all", nargs="?")
    args = parser.parse_args()
    cfg = configuration(args.mode)
    seed_all(cfg["seed"])
    root = run_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "run_config.json", cfg)
    if args.action != "all":
        run_stage(cfg, args.action)
        return
    lock = (root / "pipeline.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Token64 training pipeline already running")
    (root / "pipeline.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    for stage in STAGES:
        print(f"START stage: {stage}", flush=True)
        subprocess.run([sys.executable, "-u", __file__, "--mode", args.mode, stage], check=True)
        print(f"DONE stage: {stage}", flush=True)
    save_json(root / "status.json", {"status": "completed", "stage": "all", "pid": os.getpid()})
    print("ALL TOKEN64 FULL-TRAIN STAGES COMPLETED", flush=True)


if __name__ == "__main__":
    main()
