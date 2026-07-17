import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_common import LazyPairCache, load_manifest, verify_manifest_cache
from p2a_common import (
    NativeKVExternalReader,
    memory_to,
    mismatched_memory,
    parse_dtype,
    resolve_device,
    student_prefixed_prompt,
    zero_memory,
)
from train_memory_probes import cpu_memory, device_memory, load_writer


def build_reader(receiver, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    reader = NativeKVExternalReader(
        receiver,
        max_gate=float(args["max_gate"]),
        gate_init=float(args["gate_init"]),
        query_rank=int(args["query_rank"]),
        output_rank=int(args["output_rank"]),
    ).to(device).eval()
    reader.load_state_dict(checkpoint["adapter"])
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    return reader


def candidate_first_token_ids(tokenizer, labels):
    output = []
    for label in labels:
        ids = tokenizer(" " + label, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"No token IDs for city {label}")
        output.append(ids[0])
    return output


@torch.inference_mode()
def receiver_off(receiver, tokenizer, row, candidate_ids, device):
    encoded = tokenizer(
        student_prefixed_prompt(tokenizer, row),
        return_tensors="pt",
        add_special_tokens=False,
    )
    output = receiver(
        input_ids=encoded.input_ids.to(device),
        attention_mask=encoded.attention_mask.to(device),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return {
        "final_hidden": output.hidden_states[-1][0, -1].detach().half().cpu(),
        "candidate_first_logits": output.logits[0, -1, candidate_ids].detach().float().cpu(),
    }, encoded


@torch.inference_mode()
def receiver_on(receiver, reader, encoded, memory, off, candidate_ids):
    diagnostics = {
        "_capture_training_tensors": True,
        "_capture_query_index": int(encoded.input_ids.shape[1] - 1),
    }
    with reader.inject(receiver, memory, diagnostics):
        output = receiver(
            input_ids=encoded.input_ids.to(memory["keys"][0].device),
            attention_mask=encoded.attention_mask.to(memory["keys"][0].device),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    readouts = []
    deltas = []
    for layer in range(len(reader.gate_logits)):
        readout = diagnostics[str(layer)]["readout_tensor"][0].detach()
        readouts.append(readout.half().cpu())
        deltas.append((reader.gates()[layer] * readout).detach().half().cpu())
    final_hidden = output.hidden_states[-1][0, -1].detach()
    return {
        "reader_readouts": torch.stack(readouts),
        "injected_deltas": torch.stack(deltas),
        "final_cumulative_delta": (final_hidden.float().cpu() - off["final_hidden"].float()).half(),
        "receiver_final_hidden": final_hidden.half().cpu(),
        "candidate_first_logits": output.logits[0, -1, candidate_ids].detach().float().cpu(),
        "reader_off_final_hidden": off["final_hidden"],
        "reader_off_candidate_first_logits": off["candidate_first_logits"],
    }


def fixed_negative(cache, manifest_rows, position):
    target = manifest_rows[position]
    target_answers = {target["base_answer"], target["counterfactual_answer"]}
    for offset in range(1, len(manifest_rows)):
        candidate = manifest_rows[(position + offset) % len(manifest_rows)]
        if target_answers.isdisjoint(
            {candidate["base_answer"], candidate["counterfactual_answer"]}
        ):
            return cache.load(int(candidate["index"]))
    raise RuntimeError("No answer-disjoint negative in the quick manifest")


def cache_split(
    split,
    manifest,
    raw_cache,
    writer,
    receiver,
    reader,
    tokenizer,
    labels,
    candidate_ids,
    output,
    device,
    dtype,
):
    rows = manifest[split]
    pair_files = []
    controls = ("correct",) if split != "test" else (
        "correct", "counterfactual_state_swap", "shuffled", "mismatched", "zero"
    )
    for position, record in enumerate(tqdm(rows, desc=f"reader_chain_{split}")):
        pair = raw_cache.load(int(record["index"]))
        negative_pair = fixed_negative(raw_cache, rows, position) if split == "test" else None
        writer_memories = {}
        for variant in ("base", "counterfactual"):
            raw = memory_to(pair[variant]["memory"], device, dtype)
            with torch.inference_mode():
                writer_memories[variant] = writer(raw, output_dtype=dtype)
        if split == "test":
            negative_memories = {}
            for variant in ("base", "counterfactual"):
                raw = memory_to(negative_pair[variant]["memory"], device, dtype)
                with torch.inference_mode():
                    negative_memories[variant] = writer(raw, output_dtype=dtype)
        examples = []
        for variant in ("base", "counterfactual"):
            row = pair[variant]
            off, encoded = receiver_off(receiver, tokenizer, row, candidate_ids, device)
            for condition in controls:
                if condition == "correct":
                    memory = writer_memories[variant]
                    memory_answer = row["answer"]
                elif condition == "counterfactual_state_swap":
                    other = "counterfactual" if variant == "base" else "base"
                    memory = writer_memories[other]
                    memory_answer = pair[other]["answer"]
                elif condition == "shuffled":
                    memory = negative_memories[variant]
                    memory_answer = negative_pair[variant]["answer"]
                elif condition == "mismatched":
                    memory = mismatched_memory(writer_memories[variant], negative_memories[variant])
                    memory_answer = negative_pair[variant]["answer"]
                elif condition == "zero":
                    memory = zero_memory(writer_memories[variant])
                    memory_answer = None
                else:
                    raise ValueError(condition)
                states = receiver_on(
                    receiver, reader, encoded, memory, off, candidate_ids
                )
                examples.append(
                    {
                        "pair_id": row["pair_id"],
                        "variant": variant,
                        "condition": condition,
                        "answer": row["answer"],
                        "counterpart_answer": row["counterpart_answer"],
                        "memory_answer": memory_answer,
                        "states": states,
                    }
                )
        filename = f"pair_{position:04d}.pt"
        torch.save({"pair_id": record["pair_id"], "examples": examples}, output / filename)
        pair_files.append({"pair_id": record["pair_id"], "file": filename})
    with open(output / "index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "format_version": 1,
                "split": split,
                "conditions": list(controls),
                "labels": labels,
                "candidate_first_token_ids": candidate_ids,
                "pair_files": pair_files,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Cache the quick Writer->Reader->Qwen functional chain")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-train-index", required=True)
    parser.add_argument("--raw-test-index", required=True)
    parser.add_argument("--writer-checkpoint", required=True)
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    manifest = load_manifest(args.manifest)
    raw_train = LazyPairCache(args.raw_train_index)
    raw_test = LazyPairCache(args.raw_test_index)
    for split, cache in (("train", raw_train), ("validation", raw_train), ("test", raw_test)):
        verify_manifest_cache(manifest, split, cache)
    labels = sorted(
        {
            answer
            for entry in raw_train.entries
            for answer in (entry["base_answer"], entry["counterfactual_answer"])
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.receiver_model, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    candidate_ids = candidate_first_token_ids(tokenizer, labels)
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    reader = build_reader(receiver, args.reader_checkpoint, device)
    writer = load_writer(args.writer_checkpoint, device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for split, cache in (("train", raw_train), ("validation", raw_train), ("test", raw_test)):
        split_out = output / split
        split_out.mkdir(parents=True, exist_ok=True)
        cache_split(
            split, manifest, cache, writer, receiver, reader, tokenizer,
            labels, candidate_ids, split_out, device, dtype,
        )
    with open(output / "CACHE_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete"}, handle, indent=2)


if __name__ == "__main__":
    main()
