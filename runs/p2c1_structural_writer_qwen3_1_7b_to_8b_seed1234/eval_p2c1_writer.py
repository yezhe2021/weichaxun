import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2a_common import (
    NativeKVExternalReader,
    extract_answer,
    full_text_prefixed_prompt,
    memory_to,
    mismatched_memory,
    normalize_answer,
    parse_dtype,
    resolve_device,
    student_prefixed_prompt,
    summarize_diagnostics,
    write_jsonl,
    zero_memory,
)
from p2c1_writer import HeterogeneousNativeKVWriter, shape_only_memory


class LazyPairCache:
    def __init__(self, index_path, capacity=2):
        with open(index_path, encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.root = Path(index_path).parent
        self.entries = self.index["pair_files"]
        self.capacity = capacity
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index in self.loaded:
            self.loaded.move_to_end(index)
            return self.loaded[index]
        payload = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
        pair = {example["variant"]: example for example in payload["examples"]}
        self.loaded[index] = pair
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
        return pair


def compatible_negative(cache, index, candidate):
    if index == candidate:
        return False
    left = cache.entries[index]
    right = cache.entries[candidate]
    return {left["base_answer"], left["counterfactual_answer"]}.isdisjoint(
        {right["base_answer"], right["counterfactual_answer"]}
    )


def fixed_negative(cache, index):
    for offset in range(1, len(cache)):
        candidate = (index + offset) % len(cache)
        if compatible_negative(cache, index, candidate):
            return candidate
    raise RuntimeError(f"No compatible negative for pair {index}")


@torch.inference_mode()
def generate(receiver, tokenizer, reader, prompt, memory, max_new_tokens, device, enable_reader):
    current = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    past = None
    token_ids = []
    diagnostics = {}
    eos_ids = tokenizer.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    eos_reached = False
    for _ in range(max_new_tokens):
        if enable_reader:
            with reader.inject(receiver, memory, diagnostics):
                output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        else:
            output = receiver(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        token = int(output.logits[:, -1, :].argmax(dim=-1).item())
        token_ids.append(token)
        past = output.past_key_values
        if token in eos_ids:
            eos_reached = True
            break
        current = torch.tensor([[token]], dtype=torch.long, device=device)
    return {
        "token_ids": token_ids,
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "eos_reached": eos_reached,
        "diagnostics": summarize_diagnostics(diagnostics),
    }


@torch.inference_mode()
def capture_prompt(receiver, tokenizer, reader, row, memory, device):
    encoded = tokenizer(
        student_prefixed_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False
    )
    diagnostics = {
        "_capture_training_tensors": True,
        "_capture_query_index": int(encoded.input_ids.shape[1] - 1),
    }
    with reader.inject(receiver, memory, diagnostics):
        receiver(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
    return diagnostics


def route_readout_rows(pair_id, variant, writer_diag, native_diag, token_aligned):
    rows = []
    for layer in sorted(int(key) for key in writer_diag if key.isdigit()):
        writer_slot = writer_diag[str(layer)]
        native_slot = native_diag[str(layer)]
        writer_route = writer_slot["route_tensor"].float()
        native_route = native_slot["route_tensor"].float()
        route_kl = None
        if token_aligned and writer_route.shape == native_route.shape:
            route_kl = float(
                (F.kl_div(writer_route.clamp_min(1e-8).log(), native_route, reduction="batchmean")
                 / writer_route.shape[1]).cpu()
            )
        writer_readout = writer_slot["readout_tensor"].float()
        native_readout = native_slot["readout_tensor"].float()
        rows.append(
            {
                "pair_id": pair_id,
                "variant": variant,
                "layer": layer,
                "token_aligned": token_aligned,
                "route_kl": route_kl,
                "writer_route_entropy": float(writer_slot["route_entropy_tensor"].cpu()),
                "native_route_entropy": float(native_slot["route_entropy_tensor"].cpu()),
                "writer_target_attention_mass": float(writer_slot.get("target_mass_tensor", torch.tensor(0.0)).cpu()),
                "native_target_attention_mass": float(native_slot.get("target_mass_tensor", torch.tensor(0.0)).cpu()),
                "readout_cosine": float(
                    F.cosine_similarity(writer_readout, native_readout, dim=-1).mean().cpu()
                ),
                "readout_normalized_l2": float(
                    ((writer_readout - native_readout).norm()
                     / (0.5 * (writer_readout.norm() + native_readout.norm()) + 1e-8)).cpu()
                ),
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_summary(records):
    rows = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "metric_role": selected[0]["metric_role"],
                "n": len(selected),
                "target_em": float(np.mean([row["target_em"] for row in selected])),
                "memory_answer_hit_rate": (
                    float(np.mean([row["memory_answer_hit"] for row in selected]))
                    if any(row["memory_answer"] is not None for row in selected)
                    else None
                ),
                "eos_rate": float(np.mean([row["eos_reached"] for row in selected])),
                "answer_found_rate": float(np.mean([row["answer_found"] for row in selected])),
                "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in selected])),
            }
        )
    return rows


def paired_consistency(records, left, right):
    left_rows = {row["pair_id"]: row for row in records if row["condition"] == left}
    right_rows = {row["pair_id"]: row for row in records if row["condition"] == right}
    common = left_rows.keys() & right_rows.keys()
    return float(np.mean([left_rows[key]["target_em"] * right_rows[key]["target_em"] for key in common]))


def source_em(summary, base, counterfactual):
    values = {row["condition"]: row["target_em"] for row in summary}
    return 0.5 * (values[base] + values[counterfactual])


def recovery(writer_value, raw_value, native_value):
    denominator = native_value - raw_value
    if abs(denominator) < 1e-8:
        return None
    return (writer_value - raw_value) / denominator


def paired_gap_bootstrap(records, writer_left, writer_right, raw_left, raw_right, seed=1234, samples=10000):
    by_condition = {
        condition: {row["pair_id"]: row for row in records if row["condition"] == condition}
        for condition in (writer_left, writer_right, raw_left, raw_right)
    }
    pair_ids = sorted(set.intersection(*(set(rows) for rows in by_condition.values())))
    writer_success = np.array(
        [
            by_condition[writer_left][pair_id]["target_em"]
            * by_condition[writer_right][pair_id]["target_em"]
            for pair_id in pair_ids
        ],
        dtype=np.float64,
    )
    raw_success = np.array(
        [
            by_condition[raw_left][pair_id]["target_em"]
            * by_condition[raw_right][pair_id]["target_em"]
            for pair_id in pair_ids
        ],
        dtype=np.float64,
    )
    differences = writer_success - raw_success
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pair_ids), size=(samples, len(pair_ids)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        "mean_gap": float(differences.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the P2-B heterogeneous Writer")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--writer-checkpoint", required=True)
    parser.add_argument("--sender-index", required=True)
    parser.add_argument("--teacher-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    sender_cache = LazyPairCache(args.sender_index)
    teacher_cache = LazyPairCache(args.teacher_index)
    pair_count = min(len(sender_cache), args.max_pairs) if args.max_pairs > 0 else len(sender_cache)
    if [entry["pair_id"] for entry in sender_cache.entries] != [entry["pair_id"] for entry in teacher_cache.entries]:
        raise ValueError("Sender and teacher test caches are not pair-aligned")
    allowed_answers = sorted(
        {answer for entry in teacher_cache.entries for answer in (entry["base_answer"], entry["counterfactual_answer"])}
    )

    tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    reader_checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=False)
    reader_args = reader_checkpoint["args"]
    reader = NativeKVExternalReader(
        receiver,
        max_gate=float(reader_args["max_gate"]),
        gate_init=float(reader_args["gate_init"]),
        query_rank=int(reader_args["query_rank"]),
        output_rank=int(reader_args["output_rank"]),
    ).to(device).eval()
    reader.load_state_dict(reader_checkpoint["adapter"])

    writer_checkpoint = torch.load(args.writer_checkpoint, map_location="cpu", weights_only=False)
    writer_args = writer_checkpoint["args"]
    sender_geometry = writer_checkpoint["sender_geometry"]
    receiver_geometry = writer_checkpoint["receiver_geometry"]
    writer = HeterogeneousNativeKVWriter(
        sender_layers=sender_geometry["layers"],
        sender_heads=sender_geometry["kv_heads"],
        sender_head_dim=sender_geometry["head_dim"],
        receiver_layers=receiver_geometry["layers"],
        receiver_heads=receiver_geometry["kv_heads"],
        receiver_head_dim=receiver_geometry["head_dim"],
        layer_width=int(writer_args["layer_width"]),
    ).to(device).eval()
    writer.load_state_dict(writer_checkpoint["writer"])

    records = []
    layer_rows = []
    route_rows = []
    for pair_index in tqdm(range(pair_count), desc="p2b_free_running"):
        sender_pair = sender_cache.load(pair_index)
        teacher_pair = teacher_cache.load(pair_index)
        other_sender_pair = sender_cache.load(fixed_negative(sender_cache, pair_index))
        base = sender_pair["base"]
        counterfactual = sender_pair["counterfactual"]
        with torch.inference_mode():
            sender_base = memory_to(base["memory"], device, dtype)
            sender_cf = memory_to(counterfactual["memory"], device, dtype)
            sender_other = memory_to(other_sender_pair["base"]["memory"], device, dtype)
            native_base = memory_to(teacher_pair["base"]["memory"], device, dtype)
            native_cf = memory_to(teacher_pair["counterfactual"]["memory"], device, dtype)
            writer_base = writer(sender_base, output_dtype=dtype)
            writer_cf = writer(sender_cf, output_dtype=dtype)
            writer_other = writer(sender_other, output_dtype=dtype)
            raw_base = shape_only_memory(
                sender_base, receiver_geometry["layers"], receiver_geometry["kv_heads"],
                receiver_geometry["head_dim"], dtype,
            )
            raw_cf = shape_only_memory(
                sender_cf, receiver_geometry["layers"], receiver_geometry["kv_heads"],
                receiver_geometry["head_dim"], dtype,
            )

        conditions = [
            ("full_text_base", full_text_prefixed_prompt(tokenizer, base), None, base["answer"], base["answer"], False, "accuracy"),
            ("full_text_counterfactual", full_text_prefixed_prompt(tokenizer, counterfactual), None, counterfactual["answer"], counterfactual["answer"], False, "accuracy"),
            ("native_8b_base", student_prefixed_prompt(tokenizer, base), native_base, base["answer"], base["answer"], True, "accuracy"),
            ("native_8b_counterfactual", student_prefixed_prompt(tokenizer, base), native_cf, counterfactual["answer"], counterfactual["answer"], True, "accuracy"),
            ("writer_1_7b_base", student_prefixed_prompt(tokenizer, base), writer_base, base["answer"], base["answer"], True, "accuracy"),
            ("writer_1_7b_counterfactual", student_prefixed_prompt(tokenizer, base), writer_cf, counterfactual["answer"], counterfactual["answer"], True, "accuracy"),
            ("raw_minimal_1_7b_base", student_prefixed_prompt(tokenizer, base), raw_base, base["answer"], base["answer"], True, "accuracy"),
            ("raw_minimal_1_7b_counterfactual", student_prefixed_prompt(tokenizer, base), raw_cf, counterfactual["answer"], counterfactual["answer"], True, "accuracy"),
            ("writer_shuffled", student_prefixed_prompt(tokenizer, base), writer_other, base["answer"], other_sender_pair["base"]["answer"], True, "original_answer_leakage"),
            ("writer_mismatched", student_prefixed_prompt(tokenizer, base), mismatched_memory(writer_base, writer_other), base["answer"], other_sender_pair["base"]["answer"], True, "original_answer_leakage"),
            ("zero_kv", student_prefixed_prompt(tokenizer, base), zero_memory(writer_base), base["answer"], None, True, "original_answer_leakage"),
            ("reader_off", student_prefixed_prompt(tokenizer, base), writer_base, base["answer"], None, False, "original_answer_leakage"),
        ]
        for condition, prompt, memory, target, memory_answer, enabled, metric_role in conditions:
            result = generate(receiver, tokenizer, reader, prompt, memory, args.max_new_tokens, device, enabled)
            prediction, extraction_method = extract_answer(result["text"], allowed_answers)
            records.append(
                {
                    "pair_id": base["pair_id"],
                    "condition": condition,
                    "metric_role": metric_role,
                    "target": target,
                    "memory_answer": memory_answer,
                    "prediction": prediction,
                    "generated_text": result["text"],
                    "generated_token_ids": result["token_ids"],
                    "generated_tokens": len(result["token_ids"]),
                    "eos_reached": result["eos_reached"],
                    "answer_found": bool(prediction),
                    "extraction_method": extraction_method,
                    "target_em": float(normalize_answer(prediction) == normalize_answer(target)),
                    "memory_answer_hit": float(
                        memory_answer is not None
                        and normalize_answer(prediction) == normalize_answer(memory_answer)
                    ),
                }
            )
            for layer in result["diagnostics"]:
                layer_rows.append({"pair_id": base["pair_id"], "condition": condition, **layer})

        for variant, row, writer_memory, native_memory in (
            ("base", base, writer_base, native_base),
            ("counterfactual", counterfactual, writer_cf, native_cf),
        ):
            writer_diag = capture_prompt(receiver, tokenizer, reader, row, writer_memory, device)
            native_diag = capture_prompt(receiver, tokenizer, reader, row, native_memory, device)
            route_rows.extend(
                route_readout_rows(
                    base["pair_id"],
                    variant,
                    writer_diag,
                    native_diag,
                    sender_pair[variant].get("evidence_token_ids")
                    == teacher_pair[variant].get("evidence_token_ids"),
                )
            )

    conditions = condition_summary(records)
    text_pc = paired_consistency(records, "full_text_base", "full_text_counterfactual")
    native_pc = paired_consistency(records, "native_8b_base", "native_8b_counterfactual")
    writer_pc = paired_consistency(records, "writer_1_7b_base", "writer_1_7b_counterfactual")
    raw_pc = paired_consistency(records, "raw_minimal_1_7b_base", "raw_minimal_1_7b_counterfactual")
    native_em = source_em(conditions, "native_8b_base", "native_8b_counterfactual")
    writer_em = source_em(conditions, "writer_1_7b_base", "writer_1_7b_counterfactual")
    raw_em = source_em(conditions, "raw_minimal_1_7b_base", "raw_minimal_1_7b_counterfactual")
    paired_gap = paired_gap_bootstrap(
        records,
        "writer_1_7b_base",
        "writer_1_7b_counterfactual",
        "raw_minimal_1_7b_base",
        "raw_minimal_1_7b_counterfactual",
    )
    paired_recovery = recovery(writer_pc, raw_pc, native_pc)
    if paired_recovery is not None and paired_recovery >= 0.85:
        success_level = "strong_success"
    elif writer_pc >= 0.70 and paired_gap["ci95_low"] > 0.0:
        success_level = "basic_success"
    else:
        success_level = "success_criteria_not_met"
    summary = {
        "status": "complete",
        "args": vars(args),
        "paired_consistency": {
            "full_text": text_pc,
            "native_8b": native_pc,
            "writer_1_7b": writer_pc,
            "raw_minimal_1_7b": raw_pc,
        },
        "conditions": conditions,
        "native_gap_recovery": {
            "mean_em_writer_vs_raw": recovery(writer_em, raw_em, native_em),
            "paired_consistency_writer_vs_raw": paired_recovery,
        },
        "native_performance_gap": {
            "mean_em_native_minus_writer": native_em - writer_em,
            "paired_consistency_native_minus_writer": native_pc - writer_pc,
        },
        "writer_vs_raw_paired_bootstrap": paired_gap,
        "success_level": success_level,
        "route_readout_summary": {
            "route_kl_mean": float(np.mean([row["route_kl"] for row in route_rows if row["route_kl"] is not None])) if any(row["route_kl"] is not None for row in route_rows) else None,
            "readout_cosine_mean": float(np.mean([row["readout_cosine"] for row in route_rows])),
            "writer_target_attention_mass_mean": float(np.mean([row["writer_target_attention_mass"] for row in route_rows])),
            "native_target_attention_mass_mean": float(np.mean([row["native_target_attention_mass"] for row in route_rows])),
        },
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_jsonl(output / "per_layer_reader_diagnostics.jsonl", layer_rows)
    write_jsonl(output / "per_layer_route_readout.jsonl", route_rows)
    write_csv(output / "condition_summary.csv", conditions)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
