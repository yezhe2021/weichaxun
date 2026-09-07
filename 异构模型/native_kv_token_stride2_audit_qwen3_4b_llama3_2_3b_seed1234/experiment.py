from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, DynamicCache


ROOT = Path(__file__).resolve().parent


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(payload):
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope_at_positions(model, key, positions):
    if key.shape[1] != positions.numel():
        raise ValueError("K/position length mismatch")
    position_ids = positions.to(key.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(key[0].unsqueeze(0), position_ids)
    cos = cos[0][None, :, None, :]
    sin = sin[0][None, :, None, :]
    return key * cos + rotate_half(key) * sin


def make_cache(model, key, value, positions):
    if key.shape != value.shape:
        raise ValueError("K/V shape mismatch")
    if positions.ndim != 1 or positions.numel() != key.shape[1]:
        raise ValueError("Invalid retained positions")
    dtype = next(model.parameters()).dtype
    key, value = key.to(dtype), value.to(dtype)
    rotated_key = apply_rope_at_positions(model, key, positions)
    layers = [
        (rotated_key[layer].permute(1, 0, 2).unsqueeze(0),
         value[layer].permute(1, 0, 2).unsqueeze(0))
        for layer in range(key.shape[0])
    ]
    return DynamicCache(ddp_cache_data=layers, config=model.config)


@torch.no_grad()
def final_logits(model, suffix_ids, key, value, retained_positions, original_prefix_length):
    cache = make_cache(model, key, value, retained_positions)
    cached_tokens = int(cache.get_seq_length())
    suffix = torch.tensor([suffix_ids], device="cuda", dtype=torch.long)
    # The mask is indexed by physical cache entries; RoPE position_ids retain the
    # original timeline, so deleted context positions are gaps rather than a reindexing.
    output = model.model(
        input_ids=suffix,
        attention_mask=torch.ones(1, cached_tokens + len(suffix_ids), dtype=torch.long, device="cuda"),
        position_ids=torch.arange(original_prefix_length,
                                  original_prefix_length + len(suffix_ids),
                                  device="cuda").unsqueeze(0),
        past_key_values=cache,
        use_cache=False,
    )
    return model.lm_head(output.last_hidden_state[:, -1])[0].float()


def stride_indices(length, offset):
    if offset not in (0, 1):
        raise ValueError("offset must be 0 or 1")
    return torch.arange(offset, length, 2, dtype=torch.long)


def load_rows(cfg, count):
    source = Path(cfg["source_root"])
    manifest = read_json(source / "manifests" / f"{cfg['split']}.json")
    source_cfg = read_json(source / "run_config.json")
    if manifest["signature"] != source_cfg["signature"]:
        raise RuntimeError("Source manifest signature mismatch")
    if count > len(manifest["rows"]):
        raise RuntimeError(f"Requested {count} rows, only {len(manifest['rows'])} available")
    return manifest["rows"][:count], source_cfg["signature"]


def load_native(cfg, family, row, signature):
    path = Path(cfg["source_root"]) / "cache" / family / cfg["split"] / f"{row['id']}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    encoded = row["encoded"][family]
    if payload["signature"] != signature or payload["tokens_hash"] != digest(encoded):
        raise RuntimeError(f"Native cache mismatch: {family}/{row['id']}")
    return payload


def load_model(cfg, family):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["models"][family], local_files_only=True, dtype=torch.float16,
        attn_implementation=cfg["attention_implementation"],
    ).to("cuda").eval().requires_grad_(False)
    expected = {"qwen": (36, 8, 128), "llama": (28, 8, 128)}[family]
    observed = (model.config.num_hidden_layers, model.config.num_key_value_heads, model.config.head_dim)
    if observed != expected:
        raise RuntimeError(f"{family} architecture mismatch: {observed}")
    return model


def full_kl(student, teacher, temperature):
    student_log = F.log_softmax(student.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher.float() / temperature, dim=-1)
    return (F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) * temperature ** 2).item()


def choice_kl(student, teacher, choice_ids, temperature):
    index = torch.tensor(choice_ids, device=student.device)
    student_log = F.log_softmax(student[index].float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher[index].float() / temperature, dim=-1)
    return (F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) * temperature ** 2).item()


def summarize(records):
    fields = ("full_vocab_kl", "choice_kl", "native_agreement", "accuracy", "retained_fraction")
    result = {field: sum(record[field] for record in records) / len(records) for field in fields}
    result["samples"] = len(records)
    result["mean_retained_tokens"] = sum(record["retained_tokens"] for record in records) / len(records)
    result["mean_original_tokens"] = sum(record["original_tokens"] for record in records) / len(records)
    return result


@torch.no_grad()
def evaluate_family(cfg, family, rows, signature, output_root):
    model = load_model(cfg, family)
    conditions = {"native_full": [], "keep_even": [], "keep_odd": []}
    per_sample_path = output_root / f"{family}_per_sample.jsonl"
    started = time.monotonic()
    with per_sample_path.open("w", encoding="utf-8") as stream:
        try:
            for number, row in enumerate(rows, 1):
                payload = load_native(cfg, family, row, signature)
                encoded = row["encoded"][family]
                teacher = payload["native_logits"].cuda().float()
                choice_ids = encoded["choice_ids"]
                native_prediction = int(teacher[choice_ids].argmax())
                original_tokens = int(payload["k"].shape[1])
                native_record = {
                    "full_vocab_kl": 0.0,
                    "choice_kl": 0.0,
                    "native_agreement": 1.0,
                    "accuracy": float(native_prediction == row["gold_index"]),
                    "retained_tokens": original_tokens,
                    "original_tokens": original_tokens,
                    "retained_fraction": 1.0,
                }
                conditions["native_full"].append(native_record)
                sample = {"id": row["id"], "family": family, "native_full": native_record}
                for offset, name in ((0, "keep_even"), (1, "keep_odd")):
                    indices = stride_indices(original_tokens, offset)
                    student = final_logits(
                        model,
                        encoded["suffix"],
                        payload["k"][:, indices].cuda(),
                        payload["v"][:, indices].cuda(),
                        indices.cuda(),
                        original_tokens,
                    )
                    prediction = int(student[choice_ids].argmax())
                    record = {
                        "full_vocab_kl": full_kl(student, teacher, cfg["temperature"]),
                        "choice_kl": choice_kl(student, teacher, choice_ids, cfg["temperature"]),
                        "native_agreement": float(prediction == native_prediction),
                        "accuracy": float(prediction == row["gold_index"]),
                        "retained_tokens": int(indices.numel()),
                        "original_tokens": original_tokens,
                        "retained_fraction": indices.numel() / original_tokens,
                    }
                    conditions[name].append(record)
                    sample[name] = record
                stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if number % 8 == 0 or number == len(rows):
                    log(f"{family} evaluation {number}/{len(rows)}")
        finally:
            del model
            torch.cuda.empty_cache()
    return {
        "family": family,
        "split": cfg["split"],
        "conditions": {name: summarize(records) for name, records in conditions.items()},
        "seconds": time.monotonic() - started,
        "position_protocol": "retained K uses original token positions; suffix begins at original full-context length",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--output", default="formal")
    args = parser.parse_args()
    cfg = read_json(args.config)
    samples = args.samples or cfg["samples"]
    seed_all(cfg["seed"])
    output_root = ROOT / "runs" / args.output
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(output_root / "run_config.json", {**cfg, "effective_samples": samples})
    rows, signature = load_rows(cfg, samples)
    results = {}
    for family in ("qwen", "llama"):
        results[family] = evaluate_family(cfg, family, rows, signature, output_root)
        save_json(output_root / "results.json", results)
    save_json(output_root / "summary.json", {
        "experiment": "native_kv_token_stride2_audit",
        "question": "How much native-cache function remains after deleting every other context token?",
        "training": "none",
        "results": results,
    })
    save_json(output_root / "status.json", {"status": "completed", "samples": samples})
    log("ALL NATIVE KV STRIDE-2 AUDITS COMPLETED")


if __name__ == "__main__":
    main()
