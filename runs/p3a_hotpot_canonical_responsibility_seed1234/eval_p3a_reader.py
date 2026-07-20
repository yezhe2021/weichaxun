import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from p3a_common import (
    MemoryCache, answer_scores, extract_prediction, generate, memory_for, mismatch_memory,
    normalize_answer, summarize_records, write_json, write_jsonl, zero_memory,
)

P2IR = Path("/home/yezhe/伪查询/runs/p2ir_token_preserving_canonical_reader_onboarding_seed1234")
sys.path.insert(0, str(P2IR))
from p2ir_common import load_receiver
from p2ir_reader import TokenCanonicalReader, full_attention_layers


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--cache", required=True)
    parser.add_argument("--source", choices=("hidden", "raw_kv", "pca_kv", "canonical"), required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--out", required=True); parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); cache = MemoryCache(args.cache, 3)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, tokenizer = load_receiver(args.model, device, torch.float16); metadata = checkpoint["reader_metadata"]
    reader = TokenCanonicalReader(model, canonical_dim=256, rank=metadata["rank"], max_gate=metadata["max_gate"], gate_init=0.0, active_layers=metadata["active_layers"]).to(device).eval(); reader.load_state_dict(checkpoint["reader"])
    for parameter in reader.parameters(): parameter.requires_grad_(False)
    if not set(metadata["active_layers"]).issubset(full_attention_layers(model)): raise RuntimeError("Reader layer interface drift")
    count = min(len(cache), args.max_samples) if args.max_samples else len(cache); records = []
    for index in tqdm(range(count), desc=f"p3a_eval_{args.source}"):
        payload = cache.load(index); row = payload["row"]; correct = memory_for(payload, args.source, device)
        other_index = next((offset for offset in range(1, count) if normalize_answer(cache.load((index + offset) % count)["row"]["answer"]) != normalize_answer(row["answer"])), 1)
        other = cache.load((index + other_index) % count); shuffled = memory_for(other, args.source, device)
        conditions = [("correct", correct, row["answer"], True), ("shuffled", shuffled, other["row"]["answer"], True), ("zero", zero_memory(correct), "", True), ("reader_off", correct, "", False), ("kv_mismatch", mismatch_memory(correct, shuffled), other["row"]["answer"], True)]
        for condition, memory, source_answer, enabled in conditions:
            generated = generate(model, tokenizer, reader, row, memory, args.max_new_tokens, enabled)
            prediction, method = extract_prediction(generated["text"]); em, f1 = answer_scores(prediction, row["answer"]); source_em = answer_scores(prediction, source_answer)[0] if source_answer else 0.0
            records.append({"id": row["id"], "condition": condition, "source": args.source, "type": row["type"], "answer_type": row["answer_type"], "target": row["answer"], "source_memory_answer": source_answer, "prediction": prediction, "generated_text": generated["text"], "token_ids": generated["token_ids"], "eos_reached": generated["eos_reached"], "extraction_method": method, "em": em, "f1": f1, "source_em": source_em})
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); write_jsonl(output / "per_sample_generation.jsonl", records)
    write_json(output / "SUCCESS.json", {"status": "complete", "source": args.source, "samples": count, "profile": checkpoint.get("profile", "legacy_synthetic"), "conditions": summarize_records(records)})


if __name__ == "__main__": main()
