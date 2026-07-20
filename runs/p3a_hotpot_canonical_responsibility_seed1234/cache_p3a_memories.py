import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3a_common import file_sha256, load_jsonl, sender_text, state_sha256, write_json

P2IW = Path("/home/yezhe/伪查询/runs/p2iw_token_preserving_canonical_writer_qwen3_8b_seed1234")
sys.path.insert(0, str(P2IW))
from p2iw_common import TokenCanonicalWriter, projection


def token_indices(offsets, spans):
    return [i for i, (start, end) in enumerate(offsets) if start != end and any(start < right and end > left for left, right in spans)]


def answer_mask(text, offsets, indices, spans, answer):
    lowered, needle, ranges, start = text.casefold(), answer.casefold(), [], 0
    while needle:
        position = lowered.find(needle, start)
        if position < 0: break
        end = position + len(answer)
        if any(position < right and end > left for left, right in spans): ranges.append((position, end))
        start = max(end, position + 1)
    return torch.tensor([any(offsets[i][0] < right and offsets[i][1] > left for left, right in ranges) for i in indices], dtype=torch.bool)


@torch.inference_mode()
def encode(model, tokenizer, writer, projections, row, device, max_length):
    text, spans = sender_text(row)
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True, truncation=False)
    if encoded.input_ids.shape[1] > max_length: raise RuntimeError(f"{row['id']} exceeds {max_length} tokens")
    indices = token_indices(encoded.offset_mapping[0].tolist(), spans)
    if not indices: raise RuntimeError(f"No evidence tokens for {row['id']}")
    captured = {}; attention = model.model.layers[-1].self_attn; norm = model.model.norm
    def attention_hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None); shape = (*hidden.shape[:-1], -1, module.head_dim)
        key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
        value = module.v_proj(hidden).view(shape).transpose(1, 2)
        captured["key"] = key[0, :, indices, :].transpose(0, 1).reshape(len(indices), -1).detach()
        captured["value"] = value[0, :, indices, :].transpose(0, 1).reshape(len(indices), -1).detach()
    def norm_hook(module, args, output): captured["hidden"] = output[0, indices].detach()
    handles = [attention.register_forward_pre_hook(attention_hook, with_kwargs=True), norm.register_forward_hook(norm_hook)]
    try: model(input_ids=encoded.input_ids.to(device), attention_mask=encoded.attention_mask.to(device), use_cache=False, return_dict=True)
    finally:
        for handle in handles: handle.remove()
    if set(captured) != {"key", "value", "hidden"}: raise RuntimeError(f"State capture failed for {row['id']}")
    pca_key = F.layer_norm(projection(captured["key"], projections["key"], whiten=False), (256,))
    pca_value = F.layer_norm(projection(captured["value"], projections["value"], whiten=False), (256,))
    hidden = F.layer_norm(projection(captured["hidden"], projections["hidden"], whiten=True), (256,))
    canonical = writer(captured["key"], captured["value"])
    mask = torch.ones(len(indices), dtype=torch.bool); target = answer_mask(text, encoded.offset_mapping[0].tolist(), indices, spans, row["answer"])
    def pack(keys, values): return {"keys": keys.half().cpu().contiguous(), "values": values.half().cpu().contiguous(), "mask": mask, "answer_token_mask": target}
    memories = {
        "hidden": pack(hidden, hidden), "raw_kv": pack(captured["key"], captured["value"]),
        "pca_kv": pack(pca_key, pca_value), "canonical": pack(canonical["keys"], canonical["values"]),
    }
    return memories, int(encoded.input_ids.shape[1]), len(indices)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--data", required=True)
    parser.add_argument("--writer", required=True); parser.add_argument("--projections", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=0); parser.add_argument("--max-length", type=int, default=2048); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); rows = load_jsonl(args.data)
    if args.max_samples: rows = rows[:args.max_samples]
    bundle = torch.load(args.projections, map_location="cpu", weights_only=False); checkpoint = torch.load(args.writer, map_location="cpu", weights_only=False)
    writer = TokenCanonicalWriter(bundle["pca"], **checkpoint["writer_config"]).to(device).eval(); writer.load_state_dict(checkpoint["writer"])
    for parameter in writer.parameters(): parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); entries, sender_lengths, evidence_lengths = [], [], []
    for index, row in enumerate(tqdm(rows, desc=f"p3a_cache_{output.name}")):
        memories, sender_tokens, evidence_tokens = encode(model, tokenizer, writer, bundle["pca"], row, device, args.max_length)
        filename = f"sample_{index:05d}.pt"; torch.save({"row": row, "memories": memories}, output / filename)
        entries.append({"id": row["id"], "file": filename, "answer": row["answer"], "type": row["type"]})
        sender_lengths.append(sender_tokens); evidence_lengths.append(evidence_tokens)
    metadata = {"status": "complete", "entries": entries, "samples": len(entries), "sources": ["hidden", "raw_kv", "pca_kv", "canonical"], "canonical_dim": 256, "raw_dim": 1024, "writer_checkpoint_sha256": file_sha256(args.writer), "writer_state_sha256": state_sha256(writer.state_dict()), "max_sender_tokens": max(sender_lengths), "max_evidence_tokens": max(evidence_lengths)}
    write_json(output / "index.json", metadata); write_json(output / "SUCCESS.json", metadata)


if __name__ == "__main__": main()
