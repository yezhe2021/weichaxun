import argparse
import difflib
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2iw_common import PairCache, file_sha256, parse_dtype, resolve_device, write_json


def sender_text(row):
    text = (
        f"QUESTION\n{row['question']}\n\nEVIDENCE A\n{row['evidence_a']}\n\n"
        f"EVIDENCE B\n{row['evidence_b']}"
    )
    a_start = text.index(row["evidence_a"])
    a_end = a_start + len(row["evidence_a"])
    b_start = text.index(row["evidence_b"], a_end)
    return text, ((a_start, a_end), (b_start, b_start + len(row["evidence_b"])))


def evidence_indices(offsets, spans):
    return [
        index for index, (start, end) in enumerate(offsets)
        if start != end and any(start < right and end > left for left, right in spans)
    ]


def stable_alignment(base, counterfactual):
    def stripped(row):
        keep = [index for index, changed in enumerate(row["answer_mask"].tolist()) if not changed]
        return [row["token_ids"][index] for index in keep], keep
    base_ids, base_map = stripped(base)
    cf_ids, cf_map = stripped(counterfactual)
    matcher = difflib.SequenceMatcher(a=base_ids, b=cf_ids, autojunk=False)
    pairs = []
    for left, right, size in matcher.get_matching_blocks():
        pairs.extend((base_map[left + offset], cf_map[right + offset]) for offset in range(size))
    if not pairs:
        raise RuntimeError("No unchanged evidence-token alignment was found")
    return torch.tensor(pairs, dtype=torch.long)


@torch.inference_mode()
def encode_hidden(model, tokenizer, row, expected_ids, device, max_length):
    text, spans = sender_text(row)
    encoded = tokenizer(
        text, return_tensors="pt", return_offsets_mapping=True,
        add_special_tokens=True, truncation=False,
    )
    if encoded.input_ids.shape[1] > max_length:
        raise ValueError(f"{row['id']} exceeds max length")
    indices = evidence_indices(encoded.offset_mapping[0].tolist(), spans)
    actual_ids = encoded.input_ids[0, indices].tolist()
    if actual_ids != expected_ids:
        raise RuntimeError(f"Evidence tokenization drift for {row['id']}")
    captured = {}
    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
    if norm is None:
        raise RuntimeError("Could not find Qwen final norm")
    handle = norm.register_forward_hook(lambda module, args, output: captured.setdefault("hidden", output.detach()))
    try:
        model(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()
    return captured["hidden"][0, indices].half().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", default="float16", choices=("auto", "float16", "bfloat16", "float32"))
    args = parser.parse_args()
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    source = PairCache(args.native_index, capacity=1)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    count = min(len(source), args.max_pairs) if args.max_pairs else len(source)
    entries = []
    for pair_index in tqdm(range(count), desc="cache_p2iw_token_states"):
        pair = source.load(pair_index)
        variants = []
        for variant in ("base", "counterfactual"):
            row = pair[variant]
            key = row["memory"]["keys"][-1].transpose(0, 1).reshape(len(row["evidence_token_ids"]), -1)
            value = row["memory"]["values"][-1].transpose(0, 1).reshape(len(row["evidence_token_ids"]), -1)
            hidden = encode_hidden(model, tokenizer, row, row["evidence_token_ids"], device, args.max_length)
            variants.append({
                "pair_id": row["pair_id"], "id": row["id"], "variant": variant,
                "answer": row["answer"], "question": row["question"],
                "token_ids": row["evidence_token_ids"],
                "answer_mask": row["memory"]["answer_token_mask"].bool(),
                "key_flat": key.half().contiguous(), "value_flat": value.half().contiguous(),
                "hidden": hidden.contiguous(),
            })
        alignment = stable_alignment(variants[0], variants[1])
        filename = f"pair_{pair_index:05d}.pt"
        torch.save({"pair_id": variants[0]["pair_id"], "variants": variants, "stable_alignment": alignment}, output / filename)
        entries.append({
            "pair_id": variants[0]["pair_id"], "file": filename,
            "base_answer": variants[0]["answer"], "counterfactual_answer": variants[1]["answer"],
            "base_tokens": len(variants[0]["token_ids"]), "counterfactual_tokens": len(variants[1]["token_ids"]),
        })
    metadata = {
        "format_version": 1, "experiment": "P2-I-W", "pairs": count,
        "native_index": str(Path(args.native_index).resolve()),
        "native_index_sha256": file_sha256(args.native_index), "model": args.model,
        "key_dim": 1024, "value_dim": 1024, "hidden_dim": int(model.config.hidden_size),
        "canonical_dim": 256, "pair_files": entries,
    }
    write_json(output / "index.json", metadata)
    write_json(output / "CACHE_SUCCESS.json", {"status": "complete", **metadata})


if __name__ == "__main__":
    main()
