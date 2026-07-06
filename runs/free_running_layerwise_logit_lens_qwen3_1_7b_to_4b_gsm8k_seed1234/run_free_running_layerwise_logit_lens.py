import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


def add_import_roots(base_experiment_root):
    for root in (
        PROJECT_ROOT,
        Path(base_experiment_root).resolve(),
        PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234",
    ):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def parse_dtype(name):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def load_model(path, dtype, device):
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, trust_remote_code=True).to(device).eval()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def finite_median(values):
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def normalize_answer(text):
    text = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(prediction, gold):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum(min(pred_tokens.count(token), gold_tokens.count(token)) for token in set(pred_tokens) & set(gold_tokens))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_final_answer(text):
    text = str(text)
    if "####" in text:
        text = text.split("####")[-1]
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return normalize_answer(text)


def final_answer_exact_match(prediction, gold):
    pred = extract_final_answer(prediction)
    target = extract_final_answer(gold)
    return float(pred == target), pred, target


def checkpoint_for(base_root, method):
    checkpoint = Path(base_root) / "train" / method / "checkpoint_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {method}: {checkpoint}")
    return checkpoint


def layer_lens_metrics(receiver, hidden_states, gold_token_id):
    rows = []
    if gold_token_id is None:
        for layer_idx in range(len(hidden_states) - 1):
            rows.append(
                {
                    "layer": layer_idx,
                    "gold_logit": float("nan"),
                    "gold_prob": float("nan"),
                    "gold_rank": float("nan"),
                    "gold_margin": float("nan"),
                    "top1_token_id": None,
                    "top1_logit": float("nan"),
                }
            )
        return rows

    target = torch.tensor([int(gold_token_id)], device=hidden_states[0].device, dtype=torch.long)
    for layer_idx, hidden in enumerate(hidden_states[1:]):
        selected = hidden[:, -1:, :]
        logits = receiver.lm_head(receiver.model.norm(selected)).float()[0, 0]
        gold_logit = logits[target].squeeze(0)
        log_probs = F.log_softmax(logits, dim=-1)
        gold_prob = log_probs[target].exp().squeeze(0)
        rank = (logits > gold_logit).sum().float() + 1.0
        masked = logits.clone()
        masked[target] = float("-inf")
        margin = gold_logit - masked.max()
        top1_logit, top1_token = logits.max(dim=-1)
        rows.append(
            {
                "layer": layer_idx,
                "gold_logit": float(gold_logit.detach().cpu()),
                "gold_prob": float(gold_prob.detach().cpu()),
                "gold_rank": float(rank.detach().cpu()),
                "gold_margin": float(margin.detach().cpu()),
                "top1_token_id": int(top1_token.detach().cpu()),
                "top1_logit": float(top1_logit.detach().cpu()),
            }
        )
    return rows


def decode_ids(tokenizer, ids):
    return tokenizer.decode([int(x) for x in ids], skip_special_tokens=True).strip()


def generate_one(receiver, tokenizer, pairs, example, args):
    from real_kv_common import make_cache

    prompt_ids = example["unaware_prefix_ids"].to(args.device_obj)
    answer_ids = example["answer_ids"].to(args.device_obj)
    gold_ids = answer_ids[0].detach().cpu().tolist()
    cache = make_cache(pairs, receiver.config)
    input_ids = prompt_ids
    generated_ids = []
    trajectory = []
    prefix_aligned = True
    first_error_step = None
    stopped_by_eos = False

    for step in range(args.max_new_tokens):
        outputs = receiver(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :].float()
        next_id = int(logits.argmax(dim=-1)[0].detach().cpu())
        gold_id = int(gold_ids[step]) if step < len(gold_ids) else None
        token_matches_gold = bool(gold_id is not None and next_id == gold_id)
        prefix_before = prefix_aligned
        if prefix_aligned and not token_matches_gold:
            first_error_step = step
            prefix_aligned = False
        prefix_after = prefix_aligned
        layer_rows = layer_lens_metrics(receiver, outputs.hidden_states, gold_id)
        top1_token_text = tokenizer.decode([next_id], skip_special_tokens=False)
        gold_token_text = tokenizer.decode([gold_id], skip_special_tokens=False) if gold_id is not None else ""
        trajectory.append(
            {
                "step": step,
                "generated_token_id": next_id,
                "generated_token_text": top1_token_text,
                "gold_token_id": gold_id,
                "gold_token_text": gold_token_text,
                "token_matches_gold": token_matches_gold,
                "prefix_aligned_before_step": prefix_before,
                "prefix_aligned_after_step": prefix_after,
                "layers": layer_rows,
            }
        )
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            stopped_by_eos = True
            break
        cache = outputs.past_key_values
        input_ids = torch.tensor([[next_id]], device=args.device_obj, dtype=torch.long)

    if first_error_step is None:
        first_error_step = len(generated_ids) if len(generated_ids) <= len(gold_ids) else len(gold_ids)
    generated_text = decode_ids(tokenizer, generated_ids)
    final_em, pred_final, gold_final = final_answer_exact_match(generated_text, example["answer"])
    return {
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "pred_final_answer": pred_final,
        "gold_final_answer": gold_final,
        "final_answer_exact_match": final_em,
        "answer_f1": answer_f1(generated_text, example["answer"]),
        "first_error_step": int(first_error_step),
        "prefix_survival_length": int(first_error_step),
        "generated_len": len(generated_ids),
        "gold_len": len(gold_ids),
        "stopped_by_eos": stopped_by_eos,
        "trajectory": trajectory,
    }


def flatten_step_layer_rows(sample_row, generation):
    rows = []
    for step_item in generation["trajectory"]:
        for layer_item in step_item["layers"]:
            rows.append(
                {
                    "sample": sample_row["sample"],
                    "id": sample_row["id"],
                    "method": sample_row["method"],
                    "step": step_item["step"],
                    "layer": layer_item["layer"],
                    "gold_token_id": step_item["gold_token_id"],
                    "generated_token_id": step_item["generated_token_id"],
                    "gold_token_text": step_item["gold_token_text"],
                    "generated_token_text": step_item["generated_token_text"],
                    "token_matches_gold": float(step_item["token_matches_gold"]),
                    "prefix_aligned_before_step": float(step_item["prefix_aligned_before_step"]),
                    "prefix_aligned_after_step": float(step_item["prefix_aligned_after_step"]),
                    "gold_logit": layer_item["gold_logit"],
                    "gold_prob": layer_item["gold_prob"],
                    "gold_rank": layer_item["gold_rank"],
                    "gold_margin": layer_item["gold_margin"],
                    "top1_token_id": layer_item["top1_token_id"],
                    "top1_logit": layer_item["top1_logit"],
                }
            )
    return rows


def summarize_layerwise(rows):
    groups = {}
    for row in rows:
        key = (row["method"], int(row["step"]), int(row["layer"]))
        groups.setdefault(key, []).append(row)
    output = []
    for (method, step, layer), selected in sorted(groups.items()):
        output.append(
            {
                "method": method,
                "step": step,
                "layer": layer,
                "n": len(selected),
                "mean_gold_logit": finite_mean(row["gold_logit"] for row in selected),
                "mean_gold_prob": finite_mean(row["gold_prob"] for row in selected),
                "mean_gold_rank": finite_mean(row["gold_rank"] for row in selected),
                "median_gold_rank": finite_median(row["gold_rank"] for row in selected),
                "mean_gold_margin": finite_mean(row["gold_margin"] for row in selected),
                "median_gold_margin": finite_median(row["gold_margin"] for row in selected),
                "token_match_rate": finite_mean(row["token_matches_gold"] for row in selected),
                "prefix_survival_rate_before_step": finite_mean(row["prefix_aligned_before_step"] for row in selected),
                "prefix_survival_rate_after_step": finite_mean(row["prefix_aligned_after_step"] for row in selected),
            }
        )
    return output


def summarize_samples(rows):
    output = []
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "n": len(selected),
                "final_answer_exact_match": finite_mean(row["final_answer_exact_match"] for row in selected),
                "answer_f1": finite_mean(row["answer_f1"] for row in selected),
                "mean_first_error_step": finite_mean(row["first_error_step"] for row in selected),
                "median_first_error_step": finite_median(row["first_error_step"] for row in selected),
                "mean_prefix_survival_length": finite_mean(row["prefix_survival_length"] for row in selected),
                "mean_generated_len": finite_mean(row["generated_len"] for row in selected),
                "stopped_by_eos_rate": finite_mean(float(row["stopped_by_eos"]) for row in selected),
            }
        )
    return output


def first_error_rows(sample_rows):
    return [
        {
            "sample": row["sample"],
            "id": row["id"],
            "method": row["method"],
            "first_error_step": row["first_error_step"],
            "prefix_survival_length": row["prefix_survival_length"],
            "final_answer_exact_match": row["final_answer_exact_match"],
            "answer_f1": row["answer_f1"],
            "generated_len": row["generated_len"],
            "gold_len": row["gold_len"],
            "pred_final_answer": row["pred_final_answer"],
            "gold_final_answer": row["gold_final_answer"],
        }
        for row in sample_rows
    ]


def main():
    parser = argparse.ArgumentParser(description="Free-running layerwise logit-lens diagnosis for Qwen3-1.7B to Qwen3-4B on GSM8K")
    parser.add_argument("--base-experiment-root", default="/home/yezhe/伪查询/runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234")
    parser.add_argument("--sender-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/models/Qwen/Qwen3-4B")
    parser.add_argument("--data", default="/home/yezhe/数据集/gsm8k/test.jsonl")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--methods", default="native,mse_only,paper_rec_then_mixed_generation,q_aware_functional")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--tokenizer-check-samples", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cpu" and args.dtype in {"float16", "bfloat16"}:
        raise ValueError("float16/bfloat16 on CPU is unsupported")
    add_import_roots(args.base_experiment_root)
    from paper_dense_common import assert_tokenizer_compatible, build_paper_example, load_rows
    from real_kv_common import extract_cache
    from real_kv_translator import load_real_translator

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device_obj = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    rows = load_rows(args.data, args.max_samples)
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(sender_tokenizer, receiver_tokenizer, rows, args.max_source_tokens, args.tokenizer_check_samples)

    sender = load_model(args.sender_model, dtype, args.device_obj)
    receiver = load_model(args.receiver_model, dtype, args.device_obj)
    adapters = {}
    adapter_metadata = {}
    for method in methods:
        if method == "native":
            continue
        adapter, metadata = load_real_translator(checkpoint_for(args.base_experiment_root, method), map_location=args.device_obj)
        adapter = adapter.to(args.device_obj).eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        adapters[method] = adapter
        adapter_metadata[method] = metadata
    for module in (sender, receiver):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_sample = []
    per_step_layer = []

    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc="free_running_layerwise")):
            example = build_paper_example(receiver_tokenizer, row, args.max_source_tokens)
            source_ids = example["source_ids"].to(args.device_obj)
            sender_pairs = extract_cache(sender(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            native_pairs = extract_cache(receiver(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            method_pairs = {"native": native_pairs}
            for method, adapter in adapters.items():
                method_pairs[method] = adapter(sender_pairs)

            for method in methods:
                generation = generate_one(receiver, receiver_tokenizer, method_pairs[method], example, args)
                sample_row = {
                    "sample": sample,
                    "id": example["id"],
                    "method": method,
                    "dataset": "gsm8k",
                    "receiver_prompt_mode": "context_unaware",
                    "answer": example["answer"],
                    "generated_text": generation["generated_text"],
                    "pred_final_answer": generation["pred_final_answer"],
                    "gold_final_answer": generation["gold_final_answer"],
                    "final_answer_exact_match": generation["final_answer_exact_match"],
                    "answer_f1": generation["answer_f1"],
                    "first_error_step": generation["first_error_step"],
                    "prefix_survival_length": generation["prefix_survival_length"],
                    "generated_len": generation["generated_len"],
                    "gold_len": generation["gold_len"],
                    "stopped_by_eos": generation["stopped_by_eos"],
                }
                per_sample.append(sample_row)
                per_step_layer.extend(flatten_step_layer_rows(sample_row, generation))
                trajectory_payload = {
                    **sample_row,
                    "generated_ids": generation["generated_ids"],
                    "trajectory": generation["trajectory"],
                }
                with open(out / "per_sample_trajectory.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(trajectory_payload, ensure_ascii=False) + "\n")

    write_jsonl(out / "per_sample_generation.jsonl", per_sample)
    write_jsonl(out / "per_step_layer_logit_lens.jsonl", per_step_layer)
    write_csv(out / "free_running_layerwise_summary.csv", summarize_layerwise(per_step_layer))
    write_csv(out / "free_running_summary.csv", summarize_samples(per_sample))
    write_csv(out / "first_error_analysis.csv", first_error_rows(per_sample))
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "samples": len(rows),
                "methods": methods,
                "base_experiment_root": args.base_experiment_root,
                "sender_model": args.sender_model,
                "receiver_model": args.receiver_model,
                "dataset": "gsm8k",
                "receiver_prompt_mode": "context_unaware",
                "max_new_tokens": args.max_new_tokens,
                "adapter_metadata": adapter_metadata,
                "outputs": {
                    "per_sample_trajectory": "per_sample_trajectory.jsonl",
                    "per_sample_generation": "per_sample_generation.jsonl",
                    "per_step_layer_logit_lens": "per_step_layer_logit_lens.jsonl",
                    "free_running_layerwise_summary": "free_running_layerwise_summary.csv",
                    "free_running_summary": "free_running_summary.csv",
                    "first_error_analysis": "first_error_analysis.csv",
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
