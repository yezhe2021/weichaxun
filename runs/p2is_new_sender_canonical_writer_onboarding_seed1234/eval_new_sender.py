import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from p2is_common import (
    PairCache, TokenCanonicalReader, canonical_to, extract_answer, fixed_negative, full_attention_layers,
    generate, load_receiver, normalize_answer, parse_dtype, resolve_device, write_json, write_jsonl,
)


def zero(memory):
    return {**memory, "keys": torch.zeros_like(memory["keys"]), "values": torch.zeros_like(memory["values"]), "answer_token_mask": torch.zeros_like(memory["answer_token_mask"])}


def resize(value, target):
    if value.shape[0] == target: return value
    index = torch.linspace(0, value.shape[0] - 1, target, device=value.device).round().long(); return value.index_select(0, index)


def mismatch(current, other):
    length = current["keys"].shape[0]
    return {"keys": current["keys"], "values": resize(other["values"], length), "mask": current["mask"], "answer_token_mask": resize(other["answer_token_mask"].float()[:, None], length)[:, 0].bool()}


def permute(memory, order):
    return {name: value.index_select(0, order) if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == len(order) else value for name, value in memory.items()}


def summary(records):
    conditions = []
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        conditions.append({
            "condition": condition, "n": len(selected), "target_em": float(np.mean([row["target_em"] for row in selected])),
            "source_memory_accuracy": float(np.mean([row["source_memory_correct"] for row in selected])),
            "eos_rate": float(np.mean([row["eos_reached"] for row in selected])),
        })
    grouped = defaultdict(dict)
    for row in records:
        grouped[(row["condition"], row["pair_id"])][row["variant"]] = row
    def paired(condition):
        pairs = [values for (name, _), values in grouped.items() if name == condition and {"base", "counterfactual"}.issubset(values)]
        return float(np.mean([pair["base"]["target_em"] * pair["counterfactual"]["target_em"] for pair in pairs]))
    def variant_em(condition, variant):
        values = [row for row in records if row["condition"] == condition and row["variant"] == variant]
        return float(np.mean([row["target_em"] for row in values]))
    return {
        "new_sender_base_em": variant_em("new_sender_correct", "base"),
        "new_sender_counterfactual_em": variant_em("new_sender_correct", "counterfactual"),
        "new_sender_paired": paired("new_sender_correct"), "old_sender_paired": paired("old_sender_correct"),
        "paired_gap_from_old": paired("new_sender_correct") - paired("old_sender_correct"), "conditions": conditions,
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--receiver-name", required=True); parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--reader-checkpoint", required=True); parser.add_argument("--old-index", required=True); parser.add_argument("--new-index", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--max-pairs", type=int, default=64); parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda"); parser.add_argument("--dtype", default="float16")
    args = parser.parse_args(); device = resolve_device(args.device); dtype = parse_dtype(args.dtype, device)
    old, new = PairCache(args.old_index, 3), PairCache(args.new_index, 3); count = min(len(old), len(new), args.max_pairs)
    if [entry["pair_id"] for entry in old.entries[:count]] != [entry["pair_id"] for entry in new.entries[:count]]: raise RuntimeError("Old/new sender Canonical caches are not aligned")
    labels = sorted({answer for entry in old.entries for answer in (entry["base_answer"], entry["counterfactual_answer"])})
    checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["writer_checkpoint_sha256"] != old.index["writer_checkpoint_sha256"]: raise RuntimeError("Frozen Reader does not belong to old public interface")
    model, tokenizer = load_receiver(args.receiver_model, device, dtype); metadata = checkpoint["reader_metadata"]
    reader = TokenCanonicalReader(model, canonical_dim=256, rank=metadata["rank"], max_gate=metadata["max_gate"], gate_init=0.0, active_layers=metadata["active_layers"]).to(device).eval()
    reader.load_state_dict(checkpoint["reader"])
    for parameter in list(model.parameters()) + list(reader.parameters()): parameter.requires_grad_(False)
    if metadata["active_layers"] != full_attention_layers(model): raise RuntimeError("Frozen Reader layer interface changed")
    generator = torch.Generator(device=device).manual_seed(args.seed + 33); records = []
    for index in tqdm(range(count), desc=f"eval_new_sender_{args.receiver_name}"):
        old_pair, new_pair = old.load(index), new.load(index); other_pair = new.load(fixed_negative(new, index))
        for variant in ("base", "counterfactual"):
            owner = new_pair[variant]; opposite = new_pair["counterfactual" if variant == "base" else "base"]
            old_memory = canonical_to(old_pair[variant]["memory"], device); current = canonical_to(owner["memory"], device)
            swapped = canonical_to(opposite["memory"], device); shuffled_row = other_pair[variant]; shuffled = canonical_to(shuffled_row["memory"], device)
            order = torch.randperm(current["keys"].shape[0], generator=generator, device=device)
            conditions = [
                ("old_sender_correct", old_memory, owner["answer"], owner["answer"], True),
                ("new_sender_correct", current, owner["answer"], owner["answer"], True),
                ("new_sender_base_cf_swap", swapped, owner["answer"], opposite["answer"], True),
                ("new_sender_shuffled", shuffled, owner["answer"], shuffled_row["answer"], True),
                ("new_sender_kv_mismatch", mismatch(current, shuffled), owner["answer"], shuffled_row["answer"], True),
                ("new_sender_zero", zero(current), owner["answer"], "", True),
                ("reader_off", current, owner["answer"], owner["answer"], False),
                ("new_sender_token_permutation", permute(current, order), owner["answer"], owner["answer"], True),
            ]
            for condition, memory, target, source_answer, enabled in conditions:
                result = generate(model, tokenizer, reader, owner, memory, args.max_new_tokens, device, enabled)
                prediction, method = extract_answer(result["text"], labels)
                records.append({
                    "pair_id": owner["pair_id"], "variant": variant, "condition": condition, "target": target,
                    "source_memory_answer": source_answer, "prediction": prediction, "generated_text": result["text"],
                    "token_ids": result["token_ids"], "eos_reached": result["eos_reached"], "extraction_method": method,
                    "target_em": float(normalize_answer(prediction) == normalize_answer(target)),
                    "source_memory_correct": float(bool(source_answer) and normalize_answer(prediction) == normalize_answer(source_answer)),
                })
    metrics = summary(records); cmap = {row["condition"]: row for row in metrics["conditions"]}
    metrics["within_five_points_of_old"] = bool(metrics["paired_gap_from_old"] >= -0.05)
    metrics["control_gate"] = bool(
        cmap["new_sender_base_cf_swap"]["source_memory_accuracy"] > cmap["new_sender_base_cf_swap"]["target_em"]
        and cmap["new_sender_zero"]["target_em"] <= 0.10 and cmap["reader_off"]["target_em"] <= 0.10
    )
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); write_jsonl(output / "per_sample_generation.jsonl", records)
    write_json(output / "SUCCESS.json", {
        "status": "complete", "receiver": args.receiver_name, "pairs": count, **metrics,
        "old_writer_checkpoint_sha256": old.index["writer_checkpoint_sha256"],
        "new_writer_checkpoint_sha256": new.index["writer_checkpoint_sha256"],
    })


if __name__ == "__main__": main()
