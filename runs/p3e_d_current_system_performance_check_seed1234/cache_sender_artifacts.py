import argparse
import gc
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3d3_common import SELECTED_LAYERS, evidence_block, normalize_answer
from p3e_b_common import SenderNativeHeadwiseCache
from p3e_c2_common import load_writer, native_payload_to
from p3e_d_common import (CudaStageTimer, answer_in_text, model_context_limit, read_json, strip_summary,
                          summary_prompt, tensor_bytes, write_json, write_jsonl)


class NativeCapture:
    def __init__(self, model, layers):
        self.model, self.layers, self.states, self.handles = model, list(layers), {}, []

    def __enter__(self):
        for layer_index in self.layers:
            attention = self.model.model.layers[layer_index].self_attn

            def hook(module, args, kwargs, layer_index=layer_index):
                hidden = args[0] if args else kwargs["hidden_states"]
                shape = (*hidden.shape[:-1], -1, module.head_dim)
                keys = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
                values = module.v_proj(hidden).view(shape).transpose(1, 2)
                self.states[layer_index] = (keys.detach(), values.detach())

            self.handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


def encode_no_truncation(tokenizer, text, model, reserve=0):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=False)
    length = int(encoded["input_ids"].shape[1])
    limit = model_context_limit(model)
    if length + reserve > limit:
        raise RuntimeError(f"Input requires {length}+{reserve} tokens, model context is {limit}; refusing to truncate")
    return {name: value.to(model.device) for name, value in encoded.items()}, length, limit


def generate_summary(model, tokenizer, prompt, max_new_tokens):
    encoded, input_tokens, context_limit = encode_no_truncation(tokenizer, prompt, model, reserve=max_new_tokens)
    timer = CudaStageTimer()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with timer.stage("summary_generation"):
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    timings = timer.totals()
    tokens = output[0, input_tokens:].tolist()
    text = strip_summary(tokenizer.decode(tokens, skip_special_tokens=True))
    return {
        "raw_text": text,
        "token_ids": tokens,
        "input_tokens": input_tokens,
        "output_tokens": len(tokens),
        "eos_reached": tokenizer.eos_token_id in tokens,
        "context_limit": context_limit,
        "truncated": False,
        "summary_generation_ms": timings["summary_generation"],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "incremental_peak_allocated_bytes": int(max(0, torch.cuda.max_memory_allocated() - baseline)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--writer", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P3-E-D timing requires CUDA")
    device = torch.device(args.device)
    root = Path(args.out)
    canonical_root = root / "canonical"
    summaries_root = root / "summaries"
    canonical_root.mkdir(parents=True, exist_ok=True)
    summaries_root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    cache = SenderNativeHeadwiseCache(args.memory, capacity=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    writer, writer_checkpoint = load_writer(args.writer, device)
    writer.requires_grad_(False)
    writer.eval()

    selected_samples = manifest["samples"][:args.max_samples]
    if not selected_samples:
        raise RuntimeError("No Sender samples selected")
    first = cache.load(selected_samples[0]["index"])
    warm_evidence = evidence_block(first["row"])
    warm_inputs, _, _ = encode_no_truncation(tokenizer, warm_evidence, model)
    warm_summary = summary_prompt(tokenizer, first["row"], warm_evidence)
    warm_summary_inputs, _, _ = encode_no_truncation(tokenizer, warm_summary, model, reserve=8)
    with torch.inference_mode():
        model(**warm_inputs, use_cache=False)
        model.generate(**warm_summary_inputs, max_new_tokens=8, do_sample=False, use_cache=True,
                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        keys, values, _, _ = native_payload_to(first, device)
        writer(keys, values)
    torch.cuda.synchronize()

    with NativeCapture(model, SELECTED_LAYERS) as capture, torch.inference_mode():
        for sample in tqdm(selected_samples, desc="p3e_d_sender_cache"):
            index, sample_id = sample["index"], sample["id"]
            canonical_path = canonical_root / f"sample_{index:05d}.pt"
            summary_path = summaries_root / f"sample_{index:05d}.json"
            if canonical_path.exists() and summary_path.exists():
                continue
            payload = cache.load(index)
            row = payload["row"]
            if row["id"] != sample_id:
                raise RuntimeError("Manifest/cache sample order changed")
            evidence = evidence_block(row)

            sender_inputs, sender_tokens, context_limit = encode_no_truncation(tokenizer, evidence, model)
            cached_ids = payload["metadata"]["token_ids"]
            current_ids = sender_inputs["input_ids"][0].tolist()
            if current_ids != cached_ids:
                raise RuntimeError(f"Sender tokenization differs from fixed Native cache for {sample_id}")
            capture.states.clear()
            timer = CudaStageTimer()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            with timer.stage("sender_evidence_prefill"):
                model(**sender_inputs, use_cache=False)
            prefill_timing = timer.totals()
            if sorted(capture.states) != SELECTED_LAYERS:
                raise RuntimeError("Native K/V capture missed selected layers")

            keys, values, valid, support = native_payload_to(payload, device)
            writer_timer = CudaStageTimer()
            with writer_timer.stage("writer"):
                canonical_keys, canonical_values, route = writer(keys, values)
            writer_timing = writer_timer.totals()
            canonical = {
                "id": sample_id,
                "keys": canonical_keys.cpu(),
                "values": canonical_values.cpu(),
                "mask": valid.cpu(),
                "support_mask": support.cpu(),
                "writer_route": route.cpu(),
            }
            torch.save(canonical, canonical_path)

            prompt = summary_prompt(tokenizer, row, evidence)
            summary = generate_summary(model, tokenizer, prompt, args.summary_max_new_tokens)
            summary.update({
                "id": sample_id,
                "index": index,
                "prompt": prompt,
                "utf8_bytes": len(summary["raw_text"].encode("utf-8")),
                "contains_gold_answer": answer_in_text(row["answer"], summary["raw_text"], normalize_answer),
                "full_evidence_tokens_sender": sender_tokens,
                "full_evidence_utf8_bytes": len(evidence.encode("utf-8")),
                "sender_context_limit": context_limit,
                "sender_input_truncated": False,
                "sender_evidence_prefill_ms": prefill_timing["sender_evidence_prefill"],
                "writer_ms": writer_timing["writer"],
                "sender_prefill_incremental_peak_bytes": int(max(0, torch.cuda.max_memory_allocated() - baseline)),
                "native_cache_storage_bytes": tensor_bytes(payload["keys"], payload["values"]),
                "native_runtime_bytes": tensor_bytes(keys, values),
                "canonical_runtime_bytes": tensor_bytes(canonical_keys, canonical_values),
                "canonical_dtype": str(canonical_keys.dtype),
                "native_cache_dtype": str(payload["keys"].dtype),
                "native_runtime_dtype": str(keys.dtype),
            })
            write_json(summary_path, summary)
            del keys, values, canonical_keys, canonical_values, canonical
            gc.collect()

    summaries = [read_json(summaries_root / f"sample_{sample['index']:05d}.json") for sample in selected_samples]
    write_jsonl(root / "sender_summaries.jsonl", summaries)
    write_jsonl(root / "sender_timing.jsonl", [{key: value for key, value in row.items() if key not in ("prompt", "raw_text", "token_ids")} for row in summaries])
    entries = [{"id": sample["id"], "index": sample["index"], "file": f"sample_{sample['index']:05d}.pt"} for sample in selected_samples]
    write_json(canonical_root / "index.json", {
        "status": "complete",
        "entries": entries,
        "samples": len(entries),
        "shape": "[16,T,16,128]",
        "writer_checkpoint": args.writer,
        "writer_metadata": writer_checkpoint["writer_metadata"],
        "dtype": summaries[0]["canonical_dtype"],
    })
    write_json(root / "SUCCESS.json", {"status": "complete", "samples": len(entries), "summary_max_new_tokens": args.summary_max_new_tokens})


if __name__ == "__main__":
    main()
