from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import torch

from v2_common import (
    Stores, cuda, load_json, load_reader, load_scales, load_writer, progress,
    rows_for, save_json, validate_rep,
)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aggregate(rows):
    output = {}
    for condition in sorted({x["condition"] for x in rows}):
        selected = [x for x in rows if x["condition"] == condition]
        record = {
            key: sum(float(x[key]) for x in selected) / len(selected)
            for key in ("em", "token_f1", "nll")
        }
        for kind in ("bridge", "comparison"):
            subset = [x for x in selected if x["type"] == kind]
            record[f"{kind}_f1"] = (
                sum(float(x["token_f1"]) for x in subset) / len(subset)
                if subset else None
            )
        record["count"] = len(selected)
        output[condition] = record
    return output


def load_old_q35(cfg, mode):
    module = import_file(
        "old_q35_writer",
        Path(cfg["previous_q35_dir"]) / "writer.py",
    )
    old_cfg = load_json(Path(cfg["previous_q35_dir"]) / "config.json")
    writer = module.Qwen35FullAttentionWriter(old_cfg, "s2").to(cuda()).eval()
    checkpoint = Path(cfg["previous_q35_dir"]) / "artifacts" / mode / "s2" / "stage_b" / "best.pt"
    writer.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["writer"]
    )
    return writer


def load_writer8(cfg, mode):
    module = import_file(
        "reference_writer8", Path(cfg["writer8_dir"]) / "writer.py"
    )
    source_mode = "smoke" if mode == "smoke" else "development"
    stats = torch.load(
        Path(cfg["writer8_dir"]) / "artifacts" / source_mode / "protocol" / "fixed_scales.pt",
        map_location="cpu", weights_only=False,
    )
    source_cfg = load_json(Path(cfg["writer8_dir"]) / "config.json")
    writer = module.Native4ChannelWriter(source_cfg, stats, "full").to(cuda()).eval()
    checkpoint = Path(cfg["writer8_dir"]) / "artifacts" / source_mode / "full" / "stage_b" / "best.pt"
    writer.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["writer"]
    )
    return writer


def load_variants(cfg, mode):
    root = Path(cfg["work_dir"]) / "artifacts" / mode
    v0 = load_writer(cfg, mode, "v0").eval()
    v1 = load_writer(cfg, mode, "v1")
    v1.load_state_dict(torch.load(
        root / "a1" / "best.pt", map_location="cpu", weights_only=False
    )["writer"])
    v2a = load_writer(cfg, mode, "v2", root / "a2" / "best.pt")
    v2b = load_writer(cfg, mode, "v2", root / "b" / "best.pt")
    return {k: v.eval() for k, v in {
        "q35_v0_scaled": v0, "q35_v1_a1": v1,
        "q35_v2_a2": v2a, "q35_v2_b": v2b,
    }.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    rows = rows_for(cfg, args.mode)
    store = Stores(cfg, args.mode, rows)
    r1, reader, tok = load_reader(cfg, args.mode)
    variants = load_variants(cfg, args.mode)
    variants["q35_old_s2"] = load_old_q35(cfg, args.mode)
    writer8 = load_writer8(cfg, args.mode)
    conditions = [
        ("question_only", "none", "correct", True),
        ("native4_lora_off", "4b", "correct", False),
        ("native4_lora_on", "4b", "correct", True),
        ("native4_shuffled", "4b", "shuffled", True),
        ("previous_writer8", "8b_writer", "correct", True),
        ("q35_old_s2", "q35_old_s2", "correct", True),
        ("q35_v0_scaled", "q35_v0_scaled", "correct", True),
        ("q35_v1_a1", "q35_v1_a1", "correct", True),
        ("q35_v2_a2", "q35_v2_a2", "correct", True),
        ("q35_v2_b", "q35_v2_b", "correct", True),
        ("q35_v2_b_shuffled", "q35_v2_b", "shuffled", True),
        ("q35_v2_b_zero", "q35_v2_b", "zero", True),
    ]
    output = []
    for condition, family, kind, lora in conditions:
        r1.set_lora(reader, lora)
        progress(f"{args.mode}: evaluate {condition}")
        for sample in rows["test"]:
            key = value = mask = None
            compact = family == "none"
            if family == "4b":
                key, value, mask = store.memory("test", "4b", sample, kind)
                key, value = key.to(cuda()), value.to(cuda())
            elif family == "8b_writer":
                source_k, source_v, mask = store.memory("test", "8b", sample, kind)
                key, value = writer8(source_k.to(cuda()), source_v.to(cuda()))
            elif family in variants:
                source_k, source_v, mask = store.memory("test", "q35", sample, kind)
                key, value = variants[family](source_k.to(cuda()), source_v.to(cuda()))
                if kind == "zero":
                    key, value = torch.zeros_like(key), torch.zeros_like(value)
            with torch.no_grad():
                prediction = r1.greedy_generate(
                    cfg, reader, tok, sample,
                    None if key is None else key.half(),
                    None if value is None else value.half(),
                    mask, compact_positions=compact,
                )
                nll = r1.answer_loss(
                    cfg, reader, tok, sample,
                    None if key is None else key.half(),
                    None if value is None else value.half(),
                    mask, compact_question_positions=compact,
                ).item()
            output.append({
                "sample_id": sample["id"], "type": sample.get("type", "unknown"),
                "condition": condition, "answer": sample["answer"],
                "prediction": prediction,
                "em": float(r1.normalize_answer(prediction) == r1.normalize_answer(sample["answer"])),
                "token_f1": r1.token_f1(prediction, sample["answer"]),
                "nll": nll, "manual_c_p_w": "",
            })
    summary = aggregate(output)
    for key in ("q35_v0_scaled", "q35_v1_a1", "q35_v2_a2", "q35_v2_b"):
        stage = "a1" if key == "q35_v1_a1" else "a2"
        summary[f"{key}_per_layer"] = validate_rep(
            cfg, variants[key], store, rows["test"], stage, split="test"
        )
    correct, shuffled = summary["q35_v2_b"], summary["q35_v2_b_shuffled"]
    summary["q35_v2_b_dependence"] = {
        "correct_shuffled_em_gap": correct["em"] - shuffled["em"],
        "correct_shuffled_f1_gap": correct["token_f1"] - shuffled["token_f1"],
        "correct_shuffled_nll_gap": shuffled["nll"] - correct["nll"],
        "diagnostic_only": True,
    }
    root = Path(cfg["work_dir"]) / "artifacts" / args.mode / "evaluation"
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "summary.json", summary)
    save_json(root / "per_sample.json", output)
    with (root / "manual_c_p_w.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader(); writer.writerows(output)
    save_json(root / "completion.json", {
        "completed": True, "hard_gate": None,
        "stage_b_only_answer_cross_entropy": True,
        "zero_control_forced_after_writer": True,
    })


if __name__ == "__main__":
    main()
