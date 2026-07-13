import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2a_common import (
    NativeKVExternalReader,
    iter_cache,
    memory_to,
    pack_prefixed_answer,
    parse_dtype,
    resolve_device,
    student_prefixed_prompt,
    write_jsonl,
)


def load_pairs(index_path, max_pairs):
    grouped = defaultdict(dict)
    for example in iter_cache(index_path):
        grouped[example["pair_id"]][example["variant"]] = example
    pairs = [pair for pair in grouped.values() if {"base", "counterfactual"}.issubset(pair)]
    return pairs[:max_pairs] if max_pairs > 0 else pairs


@torch.inference_mode()
def sequence_nll(receiver, tokenizer, adapter, row, memory, answer, max_length, device):
    prompt = student_prefixed_prompt(tokenizer, row)
    ids, mask, labels = pack_prefixed_answer(tokenizer, prompt, answer, max_length, device)
    with adapter.inject(receiver, memory):
        output = receiver(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
    return float(output.loss.float().cpu())


@torch.inference_mode()
def capture_state(receiver, tokenizer, adapter, row, memory, device):
    encoded = tokenizer(
        student_prefixed_prompt(tokenizer, row),
        return_tensors="pt",
        add_special_tokens=False,
    )
    diagnostics = {"_capture_vectors": True}
    with adapter.inject(receiver, memory, diagnostics):
        receiver(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
    return diagnostics


def vector_distance(left, right):
    left = left.float()
    right = right.float()
    cosine = 1.0 - float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=-1).item())
    scale = 0.5 * (float(left.norm()) + float(right.norm())) + 1e-8
    normalized_l2 = float((left - right).norm()) / scale
    return cosine, normalized_l2


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_layer_rows(rows):
    output = []
    for layer in sorted({row["layer"] for row in rows}):
        selected = [row for row in rows if row["layer"] == layer]
        numeric = [key for key in selected[0] if key not in {"pair_id", "layer"}]
        summary = {"layer": layer, "n": len(selected)}
        for key in numeric:
            summary[f"{key}_mean"] = float(np.mean([row[key] for row in selected]))
        output.append(summary)
    return output


def main():
    parser = argparse.ArgumentParser(description="Held-out content diagnosis for P2-A1")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    if not train_args.get("prefill_final", False):
        raise ValueError("P2-A1 diagnosis requires a checkpoint trained with --prefill-final")
    pairs = load_pairs(args.test_index, args.max_pairs)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    adapter = NativeKVExternalReader(
        receiver,
        max_gate=float(train_args["max_gate"]),
        gate_init=float(train_args["gate_init"]),
        reader_rank=int(train_args["reader_rank"]),
    ).to(device).eval()
    adapter.load_state_dict(checkpoint["adapter"])

    nll_rows = []
    layer_rows = []
    for pair in tqdm(pairs, desc="p2a1_content_diagnosis"):
        base = pair["base"]
        counterfactual = pair["counterfactual"]
        base_memory = memory_to(base["memory"], device, dtype)
        cf_memory = memory_to(counterfactual["memory"], device, dtype)

        nll_b_y = sequence_nll(
            receiver, tokenizer, adapter, base, base_memory, base["answer"], args.max_length, device
        )
        nll_b_cf = sequence_nll(
            receiver, tokenizer, adapter, base, base_memory, counterfactual["answer"], args.max_length, device
        )
        nll_cf_cf = sequence_nll(
            receiver, tokenizer, adapter, base, cf_memory, counterfactual["answer"], args.max_length, device
        )
        nll_cf_y = sequence_nll(
            receiver, tokenizer, adapter, base, cf_memory, base["answer"], args.max_length, device
        )
        base_margin = nll_b_cf - nll_b_y
        cf_margin = nll_cf_y - nll_cf_cf
        nll_rows.append(
            {
                "pair_id": base["pair_id"],
                "base_answer": base["answer"],
                "counterfactual_answer": counterfactual["answer"],
                "nll_b_y": nll_b_y,
                "nll_b_counterfactual": nll_b_cf,
                "nll_counterfactual_y_counterfactual": nll_cf_cf,
                "nll_counterfactual_y": nll_cf_y,
                "base_target_margin": base_margin,
                "counterfactual_target_margin": cf_margin,
                "both_margins_positive": float(base_margin > 0.0 and cf_margin > 0.0),
            }
        )

        base_state = capture_state(receiver, tokenizer, adapter, base, base_memory, device)
        cf_state = capture_state(receiver, tokenizer, adapter, base, cf_memory, device)
        for layer in range(len(adapter.gate_logits)):
            left = base_state[str(layer)]
            right = cf_state[str(layer)]
            readout_cosine, readout_l2 = vector_distance(left["readout_vector"], right["readout_vector"])
            delta_cosine, delta_l2 = vector_distance(left["delta_vector"], right["delta_vector"])
            layer_rows.append(
                {
                    "pair_id": base["pair_id"],
                    "layer": layer,
                    "gate": left["gate"],
                    "base_readout_norm": float(left["readout_vector"].norm()),
                    "counterfactual_readout_norm": float(right["readout_vector"].norm()),
                    "readout_cosine_distance": readout_cosine,
                    "readout_normalized_l2": readout_l2,
                    "base_delta_norm": float(left["delta_vector"].norm()),
                    "counterfactual_delta_norm": float(right["delta_vector"].norm()),
                    "delta_cosine_distance": delta_cosine,
                    "delta_normalized_l2": delta_l2,
                    "base_target_attention_mass": float(left.get("target_attention_mass", 0.0)),
                    "counterfactual_target_attention_mass": float(right.get("target_attention_mass", 0.0)),
                    "base_attention_entropy": float(left.get("attention_entropy", 0.0)),
                    "counterfactual_attention_entropy": float(right.get("attention_entropy", 0.0)),
                }
            )

    layer_summary = aggregate_layer_rows(layer_rows)
    summary = {
        "status": "complete",
        "args": vars(args),
        "pairs": len(nll_rows),
        "base_target_margin_mean": float(np.mean([row["base_target_margin"] for row in nll_rows])),
        "counterfactual_target_margin_mean": float(
            np.mean([row["counterfactual_target_margin"] for row in nll_rows])
        ),
        "base_margin_positive_rate": float(np.mean([row["base_target_margin"] > 0 for row in nll_rows])),
        "counterfactual_margin_positive_rate": float(
            np.mean([row["counterfactual_target_margin"] > 0 for row in nll_rows])
        ),
        "paired_margin_success_rate": float(np.mean([row["both_margins_positive"] for row in nll_rows])),
        "readout_cosine_distance_mean": float(np.mean([row["readout_cosine_distance"] for row in layer_rows])),
        "readout_normalized_l2_mean": float(np.mean([row["readout_normalized_l2"] for row in layer_rows])),
        "delta_cosine_distance_mean": float(np.mean([row["delta_cosine_distance"] for row in layer_rows])),
        "delta_normalized_l2_mean": float(np.mean([row["delta_normalized_l2"] for row in layer_rows])),
        "target_attention_mass_mean": float(
            np.mean(
                [
                    0.5 * (row["base_target_attention_mass"] + row["counterfactual_target_attention_mass"])
                    for row in layer_rows
                ]
            )
        ),
        "gates": adapter.gates().detach().float().cpu().tolist(),
    }

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_pair_nll.jsonl", nll_rows)
    write_jsonl(output / "per_pair_layer_diagnosis.jsonl", layer_rows)
    write_csv(output / "layer_summary.csv", layer_summary)
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
