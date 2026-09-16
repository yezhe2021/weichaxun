import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment import (ROOT, GemmaToQwen, QwenToLlama, ResidualAdapter, capture_llama,
                        encode_llama, load_state, make_cache, read_json, save_json,
                        target_token, token_spans)


@torch.no_grad()
def greedy_generate(model, tokenizer, suffix_ids, key, value, max_new_tokens):
    cache = make_cache(model, key, value, torch.arange(key.shape[1], device="cuda"))
    prefix = key.shape[1]
    suffix = torch.tensor([suffix_ids], device="cuda", dtype=torch.long)
    output = model.model(
        input_ids=suffix,
        attention_mask=torch.ones((1, prefix + len(suffix_ids)), device="cuda", dtype=torch.long),
        position_ids=torch.arange(prefix, prefix + len(suffix_ids), device="cuda")[None],
        past_key_values=cache,
        use_cache=True,
    )
    logits = model.lm_head(output.last_hidden_state[:, -1])[0].float()
    generated = []
    past = output.past_key_values
    for step in range(max_new_tokens):
        token = int(logits.argmax())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        position = prefix + len(suffix_ids) + step
        next_ids = torch.tensor([[token]], device="cuda", dtype=torch.long)
        output = model.model(
            input_ids=next_ids,
            attention_mask=torch.ones((1, position + 1), device="cuda", dtype=torch.long),
            position_ids=torch.tensor([[position]], device="cuda", dtype=torch.long),
            past_key_values=past,
            use_cache=True,
        )
        logits = model.lm_head(output.last_hidden_state[:, -1])[0].float()
        past = output.past_key_values
    return generated, tokenizer.decode(generated, skip_special_tokens=True)


def parsed_label(text, labels):
    match = re.search(r"(?:^|\s|[\(\[])([A-D])(?:\b|[\)\].,:])", text.strip(), re.IGNORECASE)
    value = match.group(1).upper() if match else None
    return value if value in labels else None


def summarize(records, condition):
    rows = [record["conditions"][condition] for record in records]
    count = len(rows)
    return {
        "samples": count,
        "choice_argmax_correct": sum(row["choice_argmax_correct"] for row in rows),
        "choice_argmax_accuracy": sum(row["choice_argmax_correct"] for row in rows) / count,
        "first_generated_token_is_choice": sum(row["first_generated_token_is_choice"] for row in rows),
        "first_generated_token_choice_rate": sum(row["first_generated_token_is_choice"] for row in rows) / count,
        "first_generated_token_correct": sum(row["first_generated_token_correct"] for row in rows),
        "first_generated_token_accuracy": sum(row["first_generated_token_correct"] for row in rows) / count,
        "parsed_valid": sum(row["parsed_label"] is not None for row in rows),
        "parsed_valid_rate": sum(row["parsed_label"] is not None for row in rows) / count,
        "parsed_correct": sum(row["parsed_correct"] for row in rows),
        "parsed_accuracy": sum(row["parsed_correct"] for row in rows) / count,
    }


@torch.no_grad()
def run(cfg, max_new_tokens):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    forward_root = Path(cfg["forward_root"])
    rows = read_json(forward_root / "runs/study/manifests/test.json")["rows"][:cfg["test_samples"]]

    forward_base, _ = load_state(GemmaToQwen(cfg["forward_depth_hidden_dim"],
                                              cfg["forward_depth_output_dim"]),
                                 cfg["forward_stage_a_checkpoint"])
    forward_adapter, _ = load_state(ResidualAdapter(36, cfg["adapter_rank"]),
                                    cfg["forward_residual_checkpoint"], "residual")
    reverse_base, _ = load_state(QwenToLlama(cfg["reverse_hidden_dim"]),
                                 cfg["reverse_stage_a_checkpoint"])
    reverse_adapter, _ = load_state(ResidualAdapter(28, cfg["adapter_rank"]),
                                    cfg["reverse_residual_checkpoint"], "residual")

    llama_tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["llama"], local_files_only=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["qwen"], local_files_only=True)
    llama = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["llama"], local_files_only=True, dtype=torch.float16,
        attn_implementation=cfg["attention_implementation"]).cuda().eval().requires_grad_(False)

    records = []
    for number, row in enumerate(rows, 1):
        sample_id = row["id"]
        source = torch.load(forward_root / "runs/study/source_selected_cache/test" / f"{sample_id}.pt",
                            map_location="cpu", weights_only=True)
        pair = torch.load(forward_root / "runs/study/pair_cache/test" / f"{sample_id}.pt",
                          map_location="cpu", weights_only=True)
        source_k = source["source_k"][:, 1:].cuda()[None]
        source_v = source["source_v"][:, 1:].cuda()[None]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            qwen_a_k, qwen_a_v = forward_base(source_k, source_v)
            qwen_b_k, qwen_b_v = forward_adapter(qwen_a_k, qwen_a_v)
            qwen_inputs = {
                "qwen_native_bridge": (pair["target_k"][:, 1:].cuda()[None],
                                       pair["target_v"][:, 1:].cuda()[None]),
                "gemma_stage_a": (qwen_a_k, qwen_a_v),
                "gemma_stage_b_residual": (qwen_b_k, qwen_b_v),
            }
            reverse_a = {name: reverse_base(key, value) for name, (key, value) in qwen_inputs.items()}
            reverse_b_stage_b = reverse_adapter(*reverse_a["gemma_stage_b_residual"])

        encoded = encode_llama(llama_tokenizer, row)
        llama_k, llama_v, _ = capture_llama(llama, encoded["full"], len(encoded["body"]))
        llama_spans = token_spans(llama_tokenizer, row["body"], encoded["body"])
        qwen_spans = {span.index: span for span in token_spans(
            qwen_tokenizer, row["body"], row["encoded"]["qwen"]["body"])}
        option_start, option_end = row["options_char_span"]
        llama_options = [span for span in llama_spans
                         if min(span.end, option_end) > max(span.start, option_start)]
        indices = [target_token(anchor["anchor_char"], qwen_spans[anchor["target_index"]],
                                llama_options).index for anchor in pair["metadata"]["anchors"]]
        question_k = llama_k[:, :encoded["question_prefix_length"]].cuda()
        question_v = llama_v[:, :encoded["question_prefix_length"]].cuda()
        external = {
            "llama_native_oracle32": (llama_k[:, indices].cuda(), llama_v[:, indices].cuda()),
            "qwen_native_bridge_reverse_stage_a": (reverse_a["qwen_native_bridge"][0][0],
                                                     reverse_a["qwen_native_bridge"][1][0]),
            "gemma_stage_a_reverse_stage_a": (reverse_a["gemma_stage_a"][0][0],
                                               reverse_a["gemma_stage_a"][1][0]),
            "gemma_stage_b_reverse_stage_a": (reverse_a["gemma_stage_b_residual"][0][0],
                                               reverse_a["gemma_stage_b_residual"][1][0]),
            "gemma_stage_b_reverse_residual": (reverse_b_stage_b[0][0], reverse_b_stage_b[1][0]),
        }
        condition_rows = {}
        labels = row["labels"]
        gold_label = labels[row["gold_index"]]
        choice_ids = encoded["choice_ids"]
        for name, (external_k, external_v) in external.items():
            key = torch.cat((question_k, external_k), dim=1)
            value = torch.cat((question_v, external_v), dim=1)
            generated_ids, text = greedy_generate(
                llama, llama_tokenizer, encoded["receiver_answer"], key, value, max_new_tokens)
            first_id = generated_ids[0] if generated_ids else None
            first_choice = choice_ids.index(first_id) if first_id in choice_ids else None
            label = parsed_label(text, labels)
            # Choice argmax at the same first generation position, computed without a second forward.
            # If greedy selected a non-choice token, rerun only the suffix once for exact A-D logits.
            cache = make_cache(llama, key, value, torch.arange(key.shape[1], device="cuda"))
            suffix = torch.tensor([encoded["receiver_answer"]], device="cuda", dtype=torch.long)
            output = llama.model(
                input_ids=suffix,
                attention_mask=torch.ones((1, key.shape[1] + suffix.shape[1]), device="cuda", dtype=torch.long),
                position_ids=torch.arange(key.shape[1], key.shape[1] + suffix.shape[1], device="cuda")[None],
                past_key_values=cache, use_cache=False)
            logits = llama.lm_head(output.last_hidden_state[:, -1])[0].float()
            choice_prediction = int(logits[torch.tensor(choice_ids, device="cuda")].argmax())
            condition_rows[name] = {
                "generated_text": text,
                "generated_token_ids": generated_ids,
                "choice_argmax_prediction": labels[choice_prediction],
                "choice_argmax_correct": choice_prediction == row["gold_index"],
                "first_generated_token_id": first_id,
                "first_generated_token_is_choice": first_choice is not None,
                "first_generated_token_label": labels[first_choice] if first_choice is not None else None,
                "first_generated_token_correct": first_choice == row["gold_index"],
                "parsed_label": label,
                "parsed_correct": label == gold_label,
            }
        records.append({"id": sample_id, "gold_label": gold_label, "question": row["question"],
                        "options": row["options"], "conditions": condition_rows})
        print(f"Generation audit: {number}/{len(rows)} id={sample_id}", flush=True)

    conditions = list(records[0]["conditions"])
    summary = {
        "status": "completed",
        "protocol": cfg["protocol"] + "_greedy_generation_v1",
        "sample_count": len(records),
        "decoding": {"method": "greedy", "max_new_tokens": max_new_tokens,
                     "stop_on_eos": True, "prompt_suffix": "Receiver-native Answer:"},
        "metrics": {condition: summarize(records, condition) for condition in conditions},
    }
    root = ROOT / "runs/study/generation"
    save_json(root / "generation_summary.json", summary)
    with (root / "per_sample_generations.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    run(read_json(args.config), args.max_new_tokens)


if __name__ == "__main__":
    main()
