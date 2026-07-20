import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from hotpot_common import (
    answer_scores, extract_prediction, greedy_generate, question_prompt, summarize, write_json, write_jsonl,
)

P2IR = Path("/home/yezhe/伪查询/runs/p2ir_token_preserving_canonical_reader_onboarding_seed1234")
sys.path.insert(0, str(P2IR))
from p2ir_common import load_receiver
from p2ir_reader import TokenCanonicalReader, full_attention_layers


def to_device(memory, device):
    return {name: value.to(device) for name, value in memory.items()}


def resize(value, target):
    if len(value) == target: return value
    index = torch.linspace(0, len(value) - 1, target, device=value.device).round().long()
    return value.index_select(0, index)


def zero(memory):
    return {**memory, "keys": torch.zeros_like(memory["keys"]), "values": torch.zeros_like(memory["values"]), "answer_token_mask": torch.zeros_like(memory["answer_token_mask"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-model", required=True); parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--canonical-index", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device)
    with open(args.canonical_index, encoding="utf-8") as handle: index = json.load(handle)
    checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["writer_checkpoint_sha256"] != index["writer_checkpoint_sha256"] or checkpoint["writer_state_sha256"] != index["writer_state_sha256"]:
        raise RuntimeError("The frozen Reader and Hotpot Canonical cache use different W8 Writer weights")
    model, tokenizer = load_receiver(args.receiver_model, device, torch.float16)
    metadata = checkpoint["reader_metadata"]
    reader = TokenCanonicalReader(model, canonical_dim=metadata["canonical_dim"], rank=metadata["rank"], max_gate=metadata["max_gate"], gate_init=0.0, active_layers=metadata["active_layers"]).to(device).eval()
    reader.load_state_dict(checkpoint["reader"])
    for parameter in reader.parameters(): parameter.requires_grad_(False)
    if metadata["active_layers"] != full_attention_layers(model): raise RuntimeError("Reader layer interface drift")
    root = Path(args.canonical_index).parent; payloads = [torch.load(root / entry["file"], map_location="cpu", weights_only=False) for entry in index["entries"]]
    generator = torch.Generator(device=device).manual_seed(args.seed + 99); records = []
    for position, payload in enumerate(tqdm(payloads, desc="hotpot_w8_to_r4_canonical")):
        row = payload["row"]; memory = to_device(payload["memory"], device)
        other_position = next((offset for offset in range(1, len(payloads)) if payloads[(position + offset) % len(payloads)]["row"]["answer"].casefold() != row["answer"].casefold()), 1)
        other = payloads[(position + other_position) % len(payloads)]; other_memory = to_device(other["memory"], device)
        order = torch.randperm(len(memory["keys"]), generator=generator, device=device)
        permuted = {name: value.index_select(0, order) if value.ndim and len(value) == len(order) else value for name, value in memory.items()}
        mismatch = {**memory, "values": resize(other_memory["values"], len(memory["keys"])), "answer_token_mask": torch.zeros_like(memory["answer_token_mask"])}
        conditions = [
            ("correct", memory, row["answer"], True),
            ("shuffled", other_memory, other["row"]["answer"], True),
            ("zero", zero(memory), "", True),
            ("reader_off", memory, "", False),
            ("kv_mismatch", mismatch, other["row"]["answer"], True),
            ("token_permutation", permuted, row["answer"], True),
        ]
        for condition, current, source_answer, enabled in conditions:
            generated = greedy_generate(model, tokenizer, question_prompt(tokenizer, row), args.max_new_tokens, reader, current, enabled)
            prediction, method = extract_prediction(generated["text"]); em, f1 = answer_scores(prediction, row["answer"])
            source_em = answer_scores(prediction, source_answer)[0] if source_answer else 0.0
            records.append({
                "id": row["id"], "condition": condition, "type": row["type"], "answer_type": row["answer_type"],
                "target": row["answer"], "source_memory_answer": source_answer, "prediction": prediction,
                "generated_text": generated["text"], "token_ids": generated["token_ids"], "eos_reached": generated["eos_reached"],
                "extraction_method": method, "em": em, "f1": f1, "source_em": source_em,
            })
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_json(output / "SUCCESS.json", {"status": "complete", "samples": len(payloads), "conditions": summarize(records), "active_layers": metadata["active_layers"]})


if __name__ == "__main__":
    main()
