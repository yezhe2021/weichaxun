import argparse
import gc
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import (answer_scores, extract_prediction, load_receiver, normalize_answer, question_prompt,
                         seed_everything)
from p3e_b_common import NativeHeadwiseReader, SenderNativeHeadwiseCache, native_memory_to
from p3e_c1_common import LearnableCanonicalHeadReader
from p3e_d_common import (CONDITIONS, CudaStageTimer, evidence_prompt, model_context_limit, read_json,
                          supporting_text, tensor_bytes, timed_model_forward, timed_reader, write_json, write_jsonl)


def load_native_reader(model, path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["reader_metadata"]
    reader = NativeHeadwiseReader(model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"]).to(device)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    reader.eval()
    return reader


def load_canonical_reader(model, path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["reader_metadata"]
    reader = LearnableCanonicalHeadReader(
        model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"], metadata["top_k"], 0.25
    ).to(device)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    reader.eval()
    return reader


def canonical_memory(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    memory = {
        "keys": payload["keys"].to(device),
        "values": payload["values"].to(device),
        "mask": payload["mask"].to(device),
        "support_mask": payload["support_mask"].to(device),
    }
    return memory


@torch.inference_mode()
def generate_timed(model, tokenizer, prompt, max_new_tokens, reader=None, memory=None, enabled=False):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
    input_tokens = int(encoded["input_ids"].shape[1])
    context_limit = model_context_limit(model)
    if input_tokens + max_new_tokens > context_limit:
        raise RuntimeError(
            f"Receiver prompt requires {input_tokens}+{max_new_tokens} tokens, context is {context_limit}; refusing to truncate"
        )
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    timer = CudaStageTimer()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(timed_model_forward(model, timer))
        if reader is not None and enabled:
            stack.enter_context(timed_reader(reader, timer))
            stack.enter_context(reader.inject(model, memory))
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    wall_ms = 1000.0 * (time.perf_counter() - wall_start)
    stage_times = timer.totals()
    token_ids = output[0, input_tokens:].tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    prediction, method = extract_prediction(text)
    return {
        "text": text,
        "prediction": prediction,
        "parse_method": method,
        "token_ids": token_ids,
        "input_tokens": input_tokens,
        "output_tokens": len(token_ids),
        "eos_reached": tokenizer.eos_token_id in token_ids,
        "context_limit": context_limit,
        "truncated": False,
        "timing_ms": {
            "receiver_prefill": stage_times.get("receiver_prefill", 0.0),
            "receiver_decode_forward": stage_times.get("receiver_decode_forward", 0.0),
            "reader": stage_times.get("reader", 0.0),
            "generation_wall": wall_ms,
        },
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "incremental_peak_allocated_bytes": int(max(0, torch.cuda.max_memory_allocated() - baseline)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--canonical-cache", required=True)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--native-reader", required=True)
    parser.add_argument("--canonical-reader", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P3-E-D timing requires CUDA")
    seed_everything(args.seed)
    device = torch.device(args.device)
    root = Path(args.out)
    samples_root = root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    cache = SenderNativeHeadwiseCache(args.memory, capacity=3)
    summaries = {}
    with Path(args.summaries).open(encoding="utf-8") as handle:
        import json
        for line in handle:
            if line.strip():
                row = json.loads(line)
                summaries[row["id"]] = row

    model, tokenizer = load_receiver(args.model, device)
    native_reader = load_native_reader(model, args.native_reader, device)
    canonical_reader = load_canonical_reader(model, args.canonical_reader, device)

    selected_samples = manifest["samples"][:args.max_samples]
    if not selected_samples:
        raise RuntimeError("No Receiver samples selected")
    warm_row = cache.load(selected_samples[0]["index"])["row"]
    warm_prompt = question_prompt(tokenizer, warm_row)
    warm_encoded = tokenizer(warm_prompt, return_tensors="pt", add_special_tokens=False)
    warm_encoded = {name: value.to(device) for name, value in warm_encoded.items()}
    with torch.inference_mode():
        model.generate(**warm_encoded, max_new_tokens=4, do_sample=False, use_cache=True,
                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    prompt_rows = []
    for sample in tqdm(selected_samples, desc="p3e_d_receiver_eval"):
        index, sample_id = sample["index"], sample["id"]
        sample_path = samples_root / f"sample_{index:05d}.json"
        if sample_path.exists():
            continue
        payload = cache.load(index)
        row = payload["row"]
        source_index = sample["hard_negative_index"]
        source_payload = cache.load(source_index)
        summary = summaries[sample_id]
        full_evidence = payload["evidence"]
        support = supporting_text(row)
        prompts = {
            "question_only": question_prompt(tokenizer, row),
            "full_evidence_text": evidence_prompt(tokenizer, row, full_evidence, "FULL EVIDENCE"),
            "supporting_text": evidence_prompt(tokenizer, row, support, "OFFICIAL SUPPORTING SENTENCES"),
            "sender_summary_text": evidence_prompt(tokenizer, row, summary["raw_text"], "SENDER EVIDENCE SUMMARY"),
            "native_headwise_kv": question_prompt(tokenizer, row),
            "learned_canonical_kv": question_prompt(tokenizer, row),
            "hard_shuffled_canonical_kv": question_prompt(tokenizer, row),
            "reader_off": question_prompt(tokenizer, row),
        }
        outputs = {}
        for condition in CONDITIONS:
            reader, memory, enabled = None, None, False
            memory_bytes, memory_dtype = 0, None
            if condition == "native_headwise_kv":
                reader = native_reader
                memory = native_memory_to(payload, device)
                enabled = True
                memory_bytes = tensor_bytes(memory["keys"], memory["values"])
                memory_dtype = str(memory["keys"].dtype)
            elif condition == "learned_canonical_kv":
                reader = canonical_reader
                memory = canonical_memory(Path(args.canonical_cache) / f"sample_{index:05d}.pt", device)
                enabled = True
                memory_bytes = tensor_bytes(memory["keys"], memory["values"])
                memory_dtype = str(memory["keys"].dtype)
            elif condition == "hard_shuffled_canonical_kv":
                reader = canonical_reader
                memory = canonical_memory(Path(args.canonical_cache) / f"sample_{source_index:05d}.pt", device)
                enabled = True
                memory_bytes = tensor_bytes(memory["keys"], memory["values"])
                memory_dtype = str(memory["keys"].dtype)
            elif condition == "reader_off":
                reader = canonical_reader
                enabled = False
            output = generate_timed(
                model, tokenizer, prompts[condition], args.max_new_tokens, reader=reader, memory=memory, enabled=enabled
            )
            em, f1 = answer_scores(output["prediction"], row["answer"])
            output.update({"em": em, "f1": f1, "memory_runtime_bytes": memory_bytes, "memory_dtype": memory_dtype})
            if condition == "hard_shuffled_canonical_kv":
                source_em, source_f1 = answer_scores(output["prediction"], source_payload["row"]["answer"])
                output.update({
                    "source_id": source_payload["row"]["id"],
                    "source_answer": source_payload["row"]["answer"],
                    "source_answer_em": source_em,
                    "source_answer_f1": source_f1,
                })
            outputs[condition] = output
            del memory
            gc.collect()

        question_prediction = normalize_answer(outputs["question_only"]["prediction"])
        for condition, output in outputs.items():
            output["switch_from_question_only"] = float(normalize_answer(output["prediction"]) != question_prediction)
        result = {
            "id": sample_id,
            "index": index,
            "type": row["type"],
            "answer": row["answer"],
            "conditions": outputs,
            "canonical_shuffled_prediction_switch": float(
                normalize_answer(outputs["learned_canonical_kv"]["prediction"])
                != normalize_answer(outputs["hard_shuffled_canonical_kv"]["prediction"])
            ),
            "question_only_reader_off_exact": float(outputs["question_only"]["text"] == outputs["reader_off"]["text"]),
            "text_payloads": {
                "full_evidence_utf8_bytes": len(full_evidence.encode("utf-8")),
                "supporting_text_utf8_bytes": len(support.encode("utf-8")),
                "summary_utf8_bytes": len(summary["raw_text"].encode("utf-8")),
                "summary_output_tokens_sender": summary["output_tokens"],
                "summary_contains_gold_answer": summary["contains_gold_answer"],
            },
        }
        write_json(sample_path, result)
        prompt_rows.append({"id": sample_id, "prompts": prompts})

    rows = [read_json(samples_root / f"sample_{sample['index']:05d}.json") for sample in selected_samples]
    write_jsonl(root / "per_example.jsonl", rows)
    all_prompts = []
    for row in rows:
        payload = cache.load(row["index"])
        source_summary = summaries[row["id"]]
        source_row = payload["row"]
        all_prompts.append({
            "id": row["id"],
            "question_only": question_prompt(tokenizer, source_row),
            "full_evidence_text": evidence_prompt(tokenizer, source_row, payload["evidence"], "FULL EVIDENCE"),
            "supporting_text": evidence_prompt(tokenizer, source_row, supporting_text(source_row), "OFFICIAL SUPPORTING SENTENCES"),
            "sender_summary_text": evidence_prompt(tokenizer, source_row, source_summary["raw_text"], "SENDER EVIDENCE SUMMARY"),
        })
    write_jsonl(root / "prompts.jsonl", all_prompts)
    write_json(root / "SUCCESS.json", {"status": "complete", "samples": len(rows), "conditions": CONDITIONS})


if __name__ == "__main__":
    main()
