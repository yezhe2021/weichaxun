import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment import file_sha256, load_jsonl, normalize_answer, write_json


def overlap_mask(offsets, spans):
    return [
        bool(end > start and any(end > left and start < right for left, right in spans))
        for start, end in offsets
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-memory-tokens", type=int, default=1024)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.sender, trust_remote_code=True, local_files_only=True)
    sender = AutoModelForCausalLM.from_pretrained(
        args.sender, dtype=torch.float16, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in sender.parameters():
        parameter.requires_grad_(False)
    rows = load_jsonl(args.data)
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    entries, lengths = [], []
    with torch.inference_mode():
        for index, row in enumerate(tqdm(rows, desc="cache_heterogeneous_memory")):
            encoded = tokenizer(
                row["evidence"], return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            tokens = int(encoded["input_ids"].shape[1])
            if tokens > args.max_memory_tokens:
                raise RuntimeError(
                    f"Evidence for {row['id']} has {tokens} tokens, exceeding {args.max_memory_tokens}; refusing silent truncation"
                )
            output_state = sender.model(
                **{name: value.to(device) for name, value in encoded.items()},
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[0].detach().half().cpu()
            if output_state.shape[0] != tokens:
                raise RuntimeError("Sender memory/token length mismatch")
            valid = [end > start for start, end in offsets]
            support = overlap_mask(offsets, row["support_char_spans"])
            filename = f"sample_{index:05d}.pt"
            torch.save({
                "row": row,
                "memory": output_state,
                "metadata": {
                    "token_ids": encoded["input_ids"][0].tolist(),
                    "offsets": offsets,
                    "valid_mask": valid,
                    "support_token_mask": support,
                    "sender_output": "final_rmsnorm_hidden_state",
                },
            }, output / filename)
            entries.append({
                "id": row["id"],
                "file": filename,
                "tokens": tokens,
                "type": row["type"],
                "answer": row["answer"],
                "answer_type": row["answer_type"],
                "supporting_titles": row["supporting_titles"],
                "bridge_entity": row["bridge_entity"],
                "evidence_normalized": normalize_answer(row["evidence"]),
            })
            lengths.append(tokens)
    result = {
        "status": "complete",
        "samples": len(entries),
        "entries": entries,
        "memory_dim": int(sender.config.hidden_size),
        "memory_shape": "[T,dm]",
        "sender": args.sender,
        "sender_config_sha256": file_sha256(Path(args.sender) / "config.json"),
        "sender_frozen": True,
        "sender_input": "evidence_only",
        "max_tokens": max(lengths),
        "min_tokens": min(lengths),
        "silent_truncation": False,
    }
    write_json(output / "index.json", result)
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
