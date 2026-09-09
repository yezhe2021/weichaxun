from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, DynamicCache

from modules import SoftLocalSlotCompressor


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("native_full", "slot16_original", "slot16_shifted_control", "token0_plus_slot16")


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


def apply_rope(model, key, positions):
    if key.shape[1] != positions.numel():
        raise ValueError("K/position length mismatch")
    position_ids = positions.to(key.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(key[0].unsqueeze(0), position_ids)
    cos, sin = cos[0][None, :, None, :], sin[0][None, :, None, :]
    return key * cos + rotate_half(key) * sin


def make_cache(model, key, value, positions):
    if key.shape != value.shape:
        raise ValueError("K/V shape mismatch")
    dtype = next(model.parameters()).dtype
    key, value = key.to(dtype), value.to(dtype)
    rotated = apply_rope(model, key, positions)
    layers = [
        (rotated[layer].permute(1, 0, 2).unsqueeze(0),
         value[layer].permute(1, 0, 2).unsqueeze(0))
        for layer in range(key.shape[0])
    ]
    return DynamicCache(ddp_cache_data=layers, config=model.config)


@torch.no_grad()
def final_logits(model, suffix_ids, key, value, positions, suffix_start):
    cache = make_cache(model, key, value, positions)
    cached_tokens = int(cache.get_seq_length())
    suffix = torch.tensor([suffix_ids], device="cuda", dtype=torch.long)
    output = model.model(
        input_ids=suffix,
        attention_mask=torch.ones(1, cached_tokens + len(suffix_ids), dtype=torch.long, device="cuda"),
        position_ids=torch.arange(suffix_start, suffix_start + len(suffix_ids), device="cuda").unsqueeze(0),
        past_key_values=cache,
        use_cache=False,
    )
    return model.lm_head(output.last_hidden_state[:, -1])[0].float()


def load_rows(cfg, count):
    source = Path(cfg["native_source_root"])
    manifest = read_json(source / "manifests" / f"{cfg['split']}.json")
    source_cfg = read_json(source / "run_config.json")
    if manifest["signature"] != source_cfg["signature"]:
        raise RuntimeError("Source manifest signature mismatch")
    if count > len(manifest["rows"]):
        raise RuntimeError(f"Requested {count} rows, only {len(manifest['rows'])} available")
    return manifest["rows"][:count], source_cfg["signature"]


def load_native(cfg, family, row, signature):
    path = Path(cfg["native_source_root"]) / "cache" / family / cfg["split"] / f"{row['id']}.pt"
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


def load_slot_module(cfg, family):
    layers = 36 if family == "qwen" else 28
    module = SoftLocalSlotCompressor(
        layers, slots=cfg["slots"], locality_strength=cfg["locality_strength"]
    ).cuda().eval()
    checkpoint = (Path(cfg["slot_checkpoint_root"]) / "checkpoints" / family /
                  f"lambda_{cfg['locality_strength']:g}" / "best.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["family"] != family or payload["slots"] != cfg["slots"]:
        raise RuntimeError(f"Slot checkpoint protocol mismatch: {checkpoint}")
    module.load_state_dict(payload["state"])
    return module, {"path": str(checkpoint), "step": payload["step"],
                    "validation_full_kl": payload["validation_full_kl"]}


def construct_condition(condition, native_key, native_value, slot_key, slot_value):
    device = slot_key.device
    slots = slot_key.shape[1]
    if condition == "slot16_original":
        return slot_key, slot_value, torch.arange(slots, device=device), slots
    if condition == "slot16_shifted_control":
        return slot_key, slot_value, torch.arange(1, slots + 1, device=device), slots + 1
    if condition == "token0_plus_slot16":
        key = torch.cat((native_key[:, :1], slot_key), dim=1)
        value = torch.cat((native_value[:, :1], slot_value), dim=1)
        return key, value, torch.arange(slots + 1, device=device), slots + 1
    raise ValueError(f"Unsupported constructed condition: {condition}")


def distribution_kl(student, teacher, indices=None, temperature=1.0):
    if indices is not None:
        index = torch.tensor(indices, device=student.device)
        student, teacher = student[index], teacher[index]
    student_log = F.log_softmax(student.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher.float() / temperature, dim=-1)
    return (F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) * temperature ** 2).item()


def aggregate(records):
    names = ("full_vocab_kl", "choice_kl", "native_agreement", "accuracy", "cache_tokens")
    return {**{name: sum(row[name] for row in records) / len(records) for name in names},
            "samples": len(records)}


@torch.no_grad()
def evaluate_family(cfg, family, rows, signature, output_root):
    model = load_model(cfg, family)
    module, checkpoint = load_slot_module(cfg, family)
    records = {condition: [] for condition in CONDITIONS}
    started = time.monotonic()
    output_path = output_root / f"{family}_per_sample.jsonl"
    with output_path.open("w", encoding="utf-8") as stream:
        try:
            for number, row in enumerate(rows, 1):
                payload = load_native(cfg, family, row, signature)
                native_key, native_value = payload["k"].cuda(), payload["v"].cuda()
                slot_key, slot_value = module(native_key, native_value)
                teacher = payload["native_logits"].cuda().float()
                encoded = row["encoded"][family]
                choices = encoded["choice_ids"]
                native_prediction = int(teacher[choices].argmax())
                native_record = {"full_vocab_kl": 0.0, "choice_kl": 0.0,
                                 "native_agreement": 1.0,
                                 "accuracy": float(native_prediction == row["gold_index"]),
                                 "cache_tokens": int(native_key.shape[1])}
                records["native_full"].append(native_record)
                sample = {"id": row["id"], "family": family, "native_full": native_record}
                for condition in CONDITIONS[1:]:
                    key, value, positions, suffix_start = construct_condition(
                        condition, native_key, native_value, slot_key, slot_value
                    )
                    student = final_logits(model, encoded["suffix"], key, value, positions, suffix_start)
                    prediction = int(student[choices].argmax())
                    record = {
                        "full_vocab_kl": distribution_kl(student, teacher, temperature=cfg["temperature"]),
                        "choice_kl": distribution_kl(student, teacher, choices, cfg["temperature"]),
                        "native_agreement": float(prediction == native_prediction),
                        "accuracy": float(prediction == row["gold_index"]),
                        "cache_tokens": int(key.shape[1]),
                    }
                    records[condition].append(record)
                    sample[condition] = record
                stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if number % 8 == 0 or number == len(rows):
                    log(f"{family} evaluation {number}/{len(rows)}")
        finally:
            del module, model
            torch.cuda.empty_cache()
    return {
        "family": family,
        "checkpoint": checkpoint,
        "conditions": {name: aggregate(values) for name, values in records.items()},
        "clean_token0_contrast": "token0_plus_slot16 minus slot16_shifted_control",
        "seconds": time.monotonic() - started,
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
        "experiment": "selfslot16_plus_native_token0_audit",
        "training": "none; reuse existing best 16-slot checkpoints",
        "results": results,
    })
    save_json(output_root / "status.json", {"status": "completed", "samples": samples})
    log("ALL TOKEN0 PLUS 16-SLOT AUDITS COMPLETED")


if __name__ == "__main__":
    main()
