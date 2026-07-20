import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .core import capture_hidden_taps
from .data import load_jsonl, write_json
from .modeling import QWEN3_TAPS, QWEN35_TAPS, load_frozen_model, load_tokenizer, validate_architecture


def encode(model, tokenizer, text, taps, device):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    inputs = {name: value.to(device) for name, value in encoded.items()}
    hidden, _ = capture_hidden_taps(model, inputs, taps, use_cache=False)
    return {
        "hidden_taps": [value[0].to(device="cpu", dtype=torch.float16).contiguous() for value in hidden],
        "mask": encoded["attention_mask"][0].bool().cpu(),
        "token_ids": encoded["input_ids"][0].cpu(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=("qwen3", "qwen35"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-removed", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.data, args.limit)
    taps = QWEN35_TAPS if args.kind == "qwen35" else QWEN3_TAPS
    tokenizer = load_tokenizer(args.model)
    model = load_frozen_model(args.model, args.kind, args.device, torch.float16)
    validate_architecture(model, args.kind)
    entries = []
    for index, row in enumerate(tqdm(rows, desc=f"cache_{args.kind}")):
        payload = {"row": row, "correct": encode(model, tokenizer, row["evidence_text"], taps, args.device)}
        if args.include_removed and row.get("removed_evidence_text"):
            payload["answer_removed"] = encode(model, tokenizer, row["removed_evidence_text"], taps, args.device)
        filename = f"sample_{index:06d}.pt"
        torch.save(payload, output / filename)
        entries.append({"file": filename, "id": row["id"], "tokens": int(payload["correct"]["mask"].sum())})
    write_json(output / "index.json", {
        "format_version": 1, "kind": args.kind, "model": args.model,
        "tap_layers": list(taps), "samples": len(entries), "entries": entries,
        "all_tokens_preserved": True,
    })


if __name__ == "__main__":
    main()
