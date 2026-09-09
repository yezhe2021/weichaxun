from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import torch

from data import capture_native, cuda, load_json, load_model, normalize_answer, progress, render, save_json, shuffle_derangement, stratified, token_f1, tokenizer
from experiment import checkpoint_for, evaluate_condition, new_writer


def output_root(cfg):
    return Path(cfg["work_dir"]) / "artifacts" / "eval128"


def prepare_manifest(cfg):
    destination = output_root(cfg) / "manifest.json"
    if destination.exists():
        rows = load_json(destination)
        if len(rows) != 128: raise RuntimeError("existing eval128 manifest does not contain 128 samples")
        return rows
    base = load_json(Path(cfg["work_dir"]) / "artifacts" / "manifest.json")
    original = list(base["test"])
    if len(original) != 64: raise RuntimeError("frozen test set must contain 64 samples")
    excluded = {row["id"] for split in base.values() for row in split}
    raw = load_json(cfg["hotpot_dev"])
    eligible = [row for row in raw if row["_id"] not in excluded and row.get("type") in ("bridge", "comparison")]
    additional_raw = stratified(eligible, 64, random.Random(cfg["seed"] + 128))
    tok4, tok17 = tokenizer(cfg["model_4b"]), tokenizer(cfg["model_1_7b"])
    additional = []
    for raw_row in additional_raw:
        four, small = render(tok4, raw_row, True), render(tok17, raw_row, True)
        if four["full_input_ids"] != small["full_input_ids"] or four["context_input_ids"] != small["context_input_ids"]:
            raise RuntimeError(f"4B/1.7B token mismatch: {raw_row['_id']}")
        qonly = render(tok4, raw_row, False)
        answer_ids = tok4(raw_row["answer"], add_special_tokens=False).input_ids[:cfg["max_answer_tokens"]]
        answer_ids = (answer_ids or [tok4.eos_token_id]) + [tok4.eos_token_id]
        additional.append({
            "id": raw_row["_id"], "type": raw_row["type"], "answer": raw_row["answer"], "question": raw_row["question"],
            **four, **qonly, "answer_token_ids": answer_ids, "context_length": len(four["context_input_ids"]),
        })
    rows = original + additional
    mapping = shuffle_derangement(rows)
    for row in rows: row["shuffle_id"] = mapping[row["id"]]
    counts = Counter(row["type"] for row in rows)
    if len(rows) != 128 or counts["bridge"] != 64 or counts["comparison"] != 64:
        raise RuntimeError(f"eval128 balance failure: {counts}")
    save_json(destination, rows)
    save_json(output_root(cfg) / "manifest_summary.json", {
        "count": 128, "type_counts": dict(counts), "original_test64_is_prefix": [x["id"] for x in rows[:64]] == [x["id"] for x in original],
        "new_samples": 64, "selection_seed": cfg["seed"] + 128, "training_and_validation_unchanged": True,
    })
    return rows


class Eval128Store:
    def __init__(self, cfg):
        self.cfg, self.cache = cfg, {}

    def source(self, split, sample_id):
        key = sample_id
        if key not in self.cache:
            new = Path(self.cfg["work_dir"]) / "cache" / "eval128" / "source1_7" / f"{sample_id}.pt"
            old = Path(self.cfg["work_dir"]) / "cache" / "source1_7" / "test" / f"{sample_id}.pt"
            path = new if new.exists() else old
            if not path.exists(): raise FileNotFoundError(path)
            if len(self.cache) > 3: self.cache.clear()
            self.cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.cache[key]


@torch.no_grad()
def build_source_cache(cfg):
    rows = prepare_manifest(cfg); model = load_model(cfg["model_1_7b"], cfg)
    destination = Path(cfg["work_dir"]) / "cache" / "eval128" / "source1_7"; destination.mkdir(parents=True, exist_ok=True)
    old_root = Path(cfg["work_dir"]) / "cache" / "source1_7" / "test"
    try:
        for index, sample in enumerate(rows, 1):
            if (old_root / f"{sample['id']}.pt").exists() or (destination / f"{sample['id']}.pt").exists():
                continue
            key, value = capture_native(model, sample["context_input_ids"], cfg["source_layers"])
            target = destination / f"{sample['id']}.pt"; temporary = target.with_suffix(".tmp")
            torch.save({"id": sample["id"], "pre_key": key, "value": value, "context_length": len(sample["context_input_ids"])}, temporary)
            temporary.replace(target)
            progress(f"eval128: source cache {index}/128")
    finally:
        del model; torch.cuda.empty_cache()
    store = Eval128Store(cfg)
    for sample in rows:
        record = store.source("test", sample["id"])
        if record["pre_key"].shape[0] != cfg["source_layers"]: raise RuntimeError("source cache layer mismatch")
    save_json(output_root(cfg) / "source_cache_complete.json", {"completed": True, "count": 128, "source_layers": cfg["source_layers"]})


@torch.no_grad()
def generate_text(model, tok, ids, cfg):
    tensor = torch.tensor([ids], dtype=torch.long, device=cuda())
    positions = torch.arange(len(ids), device=cuda()).unsqueeze(0)
    output = model(input_ids=tensor, attention_mask=torch.ones_like(tensor), position_ids=positions, use_cache=True)
    past, token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True)
    generated = []
    for position in range(len(ids), len(ids) + cfg["max_new_tokens"]):
        value = int(token.item())
        if value == tok.eos_token_id: break
        generated.append(value)
        output = model(input_ids=token, attention_mask=torch.ones(1, past.get_seq_length() + 1, dtype=torch.long, device=cuda()), position_ids=torch.tensor([[position]], device=cuda()), past_key_values=past, use_cache=True)
        past, token = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True)
    return tok.decode(generated, skip_special_tokens=True).strip()


def metric_row(sample, condition, prediction):
    return {"sample_id": sample["id"], "type": sample["type"], "condition": condition, "answer": sample["answer"], "prediction": prediction, "em": float(normalize_answer(prediction) == normalize_answer(sample["answer"])), "f1": token_f1(prediction, sample["answer"])}


@torch.no_grad()
def baseline(cfg, model_key):
    rows = prepare_manifest(cfg); model_path = cfg[model_key]; model = load_model(model_path, cfg); tok = tokenizer(model_path); output = []
    try:
        for index, sample in enumerate(rows, 1):
            if model_key == "model_4b":
                output.append(metric_row(sample, "qwen3_4b_question_only", generate_text(model, tok, sample["question_only_ids"], cfg)))
                output.append(metric_row(sample, "qwen3_4b_full_context", generate_text(model, tok, sample["full_input_ids"], cfg)))
            else:
                output.append(metric_row(sample, "qwen3_1_7b_full_context", generate_text(model, tok, sample["full_input_ids"], cfg)))
            if index % 8 == 0: progress(f"eval128: {model_key} baseline {index}/128")
    finally:
        del model; torch.cuda.empty_cache()
    save_json(output_root(cfg) / f"baseline_{model_key}.json", output)


@torch.no_grad()
def translated(cfg, condition):
    rows = prepare_manifest(cfg); store = Eval128Store(cfg)
    model = load_model(cfg["model_4b"], cfg); tok = tokenizer(cfg["model_4b"])
    kind, checkpoint = checkpoint_for(cfg, "development", condition)
    writer = new_writer(cfg, "development", kind, checkpoint=checkpoint)
    try:
        output = evaluate_condition(cfg, model, tok, store, rows, condition + "_correct", writer)
        output += evaluate_condition(cfg, model, tok, store, rows, condition + "_shuffled", writer, True)
    finally:
        del model, writer; torch.cuda.empty_cache()
    save_json(output_root(cfg) / f"{condition}.json", output)
    progress(f"eval128: {condition} completed")


def summarize(rows):
    result = {}
    for condition in sorted({x["condition"] for x in rows}):
        values = [x for x in rows if x["condition"] == condition]
        result[condition] = {"count": len(values), "em": sum(x["em"] for x in values) / len(values), "f1": sum(x["f1"] for x in values) / len(values)}
    return result


def finalize(cfg):
    names = ["partial28_skip_f0", "partial28_skip_ce", "partial28_repeat_f0", "partial28_repeat_ce", "repeat_continued_f0", "repeat_continued_ce", "learnable_matrix_f0", "learnable_matrix_ce"]
    rows = load_json(output_root(cfg) / "baseline_model_4b.json") + load_json(output_root(cfg) / "baseline_model_1_7b.json")
    for name in names: rows += load_json(output_root(cfg) / f"{name}.json")
    summary = summarize(rows); question_f1 = summary["qwen3_4b_question_only"]["f1"]
    comparisons = {}
    for name in names:
        correct, shuffled = summary[name + "_correct"]["f1"], summary[name + "_shuffled"]["f1"]
        comparisons[name] = {"delta_memory": correct - question_f1, "delta_shuffle": correct - shuffled, "beats_question_only": correct > question_f1, "beats_shuffled": correct > shuffled}
    comparisons["delta_matrix_ce"] = summary["learnable_matrix_ce_correct"]["f1"] - summary["repeat_continued_ce_correct"]["f1"]
    summary["comparisons"] = comparisons
    save_json(output_root(cfg) / "per_sample.json", rows); save_json(output_root(cfg) / "summary.json", summary)
    save_json(output_root(cfg) / "completion.json", {"completed": True, "test_samples": 128, "trained_checkpoints_unchanged": True})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("action"); args = parser.parse_args(); cfg = load_json(args.config)
    names = ["partial28_skip_f0", "partial28_skip_ce", "partial28_repeat_f0", "partial28_repeat_ce", "repeat_continued_f0", "repeat_continued_ce", "learnable_matrix_f0", "learnable_matrix_ce"]
    if args.action == "prepare": prepare_manifest(cfg)
    elif args.action == "cache": build_source_cache(cfg)
    elif args.action == "baseline4": baseline(cfg, "model_4b")
    elif args.action == "baseline17": baseline(cfg, "model_1_7b")
    elif args.action == "finalize": finalize(cfg)
    elif args.action in names: translated(cfg, args.action)
    else: raise ValueError(args.action)


if __name__ == "__main__": main()
