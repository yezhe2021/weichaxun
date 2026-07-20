import argparse
import hashlib
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from hotpot_common import load_jsonl, sender_text, write_json

P2IW = Path("/home/yezhe/伪查询/runs/p2iw_token_preserving_canonical_writer_qwen3_8b_seed1234")
sys.path.insert(0, str(P2IW))
from p2iw_common import TokenCanonicalWriter, file_sha256


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evidence_indices(offsets, spans):
    return [
        index for index, (start, end) in enumerate(offsets)
        if start != end and any(start < right and end > left for left, right in spans)
    ]


def answer_mask(text, offsets, indices, spans, answer):
    lowered, needle = text.casefold(), answer.casefold()
    ranges, start = [], 0
    while needle and (position := lowered.find(needle, start)) >= 0:
        end = position + len(answer)
        if any(position < right and end > left for left, right in spans):
            ranges.append((position, end))
        start = max(end, position + 1)
    return torch.tensor([
        any(offsets[index][0] < right and offsets[index][1] > left for left, right in ranges)
        for index in indices
    ], dtype=torch.bool)


@torch.inference_mode()
def encode(model, tokenizer, writer, row, device, max_length):
    text, spans = sender_text(row)
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True, truncation=False)
    tokens = int(encoded.input_ids.shape[1])
    if tokens > max_length:
        raise RuntimeError(f"{row['id']} has {tokens} sender tokens, limit {max_length}")
    indices = evidence_indices(encoded.offset_mapping[0].tolist(), spans)
    if not indices:
        raise RuntimeError(f"No evidence tokens for {row['id']}")
    layer = model.model.layers[-1]; attention = layer.self_attn; captured = {}
    def hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
        value = module.v_proj(hidden).view(shape).transpose(1, 2)
        captured["key"] = key[0, :, indices, :].transpose(0, 1).reshape(len(indices), -1).detach()
        captured["value"] = value[0, :, indices, :].transpose(0, 1).reshape(len(indices), -1).detach()
    handle = attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        model(input_ids=encoded.input_ids.to(device), attention_mask=encoded.attention_mask.to(device), use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if set(captured) != {"key", "value"}:
        raise RuntimeError(f"Last-layer K/V capture failed for {row['id']}")
    written = writer(captured["key"], captured["value"])
    return {
        "keys": written["keys"].half().cpu().contiguous(),
        "values": written["values"].half().cpu().contiguous(),
        "mask": torch.ones(len(indices), dtype=torch.bool),
        "answer_token_mask": answer_mask(text, encoded.offset_mapping[0].tolist(), indices, spans, row["answer"]),
    }, tokens, len(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--data", required=True)
    parser.add_argument("--writer-checkpoint", required=True); parser.add_argument("--projections", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device)
    rows = load_jsonl(args.data)
    checkpoint = torch.load(args.writer_checkpoint, map_location="cpu", weights_only=False)
    bundle = torch.load(args.projections, map_location="cpu", weights_only=False)
    writer = TokenCanonicalWriter(bundle["pca"], **checkpoint["writer_config"]).to(device).eval()
    writer.load_state_dict(checkpoint["writer"])
    for parameter in writer.parameters(): parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); entries = []; sender_lengths = []; evidence_lengths = []
    for index, row in enumerate(tqdm(rows, desc="hotpot_w8_canonical")):
        memory, sender_tokens, evidence_tokens = encode(model, tokenizer, writer, row, device, args.max_length)
        filename = f"sample_{index:05d}.pt"; torch.save({"row": row, "memory": memory}, output / filename)
        entries.append({"id": row["id"], "file": filename, "answer": row["answer"], "type": row["type"]})
        sender_lengths.append(sender_tokens); evidence_lengths.append(evidence_tokens)
    metadata = {
        "status": "complete", "format_version": 1, "interface": "token_preserving_canonical_evidence_kv",
        "samples": len(entries), "canonical_dim": 256, "variable_token_axis": True,
        "writer_checkpoint": str(Path(args.writer_checkpoint).resolve()),
        "writer_checkpoint_sha256": file_sha256(args.writer_checkpoint), "writer_state_sha256": state_sha256(writer.state_dict()),
        "entries": entries, "max_sender_tokens": max(sender_lengths), "max_evidence_tokens": max(evidence_lengths),
    }
    write_json(output / "index.json", metadata); write_json(output / "SUCCESS.json", metadata)


if __name__ == "__main__":
    main()
