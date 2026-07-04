import argparse
import csv
import json
import math
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
    roots = [
        PROJECT_ROOT,
        Path(base_experiment_root).resolve(),
        PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234",
    ]
    for root in roots:
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


def mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(values) / len(values)) if values else float("nan")


def median(values):
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return float("nan")
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def normalize_answer(text):
    import re

    text = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_texts(tokenizer, token_ids):
    return [tokenizer.decode([int(token_id)], skip_special_tokens=True) for token_id in token_ids[0].tolist()]


def build_token_groups(tokenizer, answer_ids, answer_text, critical_mode):
    tokens = token_texts(tokenizer, answer_ids)
    normalized_gold_words = set(normalize_answer(answer_text).split())
    numeric_mask = []
    answer_mask = []
    for token in tokens:
        normalized_token = normalize_answer(token)
        numeric_mask.append(any(ch.isdigit() for ch in token))
        answer_mask.append(bool(normalized_token and normalized_token in normalized_gold_words))

    answer_len = len(tokens)
    groups = {
        "all_answer_tokens": list(range(answer_len)),
        "first_answer_token": [0] if answer_len else [],
    }
    if any(numeric_mask):
        groups["numeric_tokens"] = [idx for idx, keep in enumerate(numeric_mask) if keep]
    if critical_mode == "numeric":
        critical = [idx for idx, keep in enumerate(numeric_mask) if keep]
    elif critical_mode == "answer":
        critical = [idx for idx, keep in enumerate(answer_mask) if keep]
        if not critical:
            critical = [idx for idx, keep in enumerate(numeric_mask) if keep]
    else:
        raise ValueError(f"Unknown critical mode: {critical_mode}")
    if not critical:
        critical = list(range(answer_len))
    groups["critical_tokens"] = critical
    return groups, tokens


def checkpoint_for(base_root, method):
    checkpoint = Path(base_root) / "train" / method / "checkpoint_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {method}: {checkpoint}")
    return checkpoint


def logit_lens_rows(receiver, hidden_states, answer_ids, answer_text, tokenizer, prefix_len, method, mode, sample, row_id, dataset_label, critical_mode):
    answer_ids = answer_ids.detach().cpu()
    answer_len = answer_ids.shape[1]
    token_groups, decoded_tokens = build_token_groups(tokenizer, answer_ids, answer_text, critical_mode)
    max_pos = hidden_states[0].shape[1]
    positions = [prefix_len - 1 + idx for idx in range(answer_len)]
    valid = [(idx, pos) for idx, pos in enumerate(positions) if 0 <= pos < max_pos]
    rows = []
    if not valid:
        return rows

    answer_index = torch.tensor([idx for idx, _ in valid], dtype=torch.long, device=hidden_states[0].device)
    hidden_positions = torch.tensor([pos for _, pos in valid], dtype=torch.long, device=hidden_states[0].device)
    target_ids = answer_ids[:, answer_index.cpu()].to(hidden_states[0].device).reshape(-1)

    for layer_idx, hidden in enumerate(hidden_states[1:]):
        selected = hidden.index_select(1, hidden_positions)
        lens_logits = receiver.lm_head(receiver.model.norm(selected)).float()[0]
        target_logits = lens_logits.gather(1, target_ids[:, None]).squeeze(1)
        log_probs = F.log_softmax(lens_logits, dim=-1)
        target_log_probs = log_probs.gather(1, target_ids[:, None]).squeeze(1)
        ranks = (lens_logits > target_logits[:, None]).sum(dim=-1).float() + 1.0
        masked = lens_logits.clone()
        masked.scatter_(1, target_ids[:, None], float("-inf"))
        margins = target_logits - masked.max(dim=-1).values

        per_token = {
            int(answer_idx): {
                "gold_logit": float(target_logits[offset].detach().cpu()),
                "gold_prob": float(target_log_probs[offset].exp().detach().cpu()),
                "gold_rank": float(ranks[offset].detach().cpu()),
                "gold_margin": float(margins[offset].detach().cpu()),
            }
            for offset, answer_idx in enumerate(answer_index.detach().cpu().tolist())
        }
        for group, indices in token_groups.items():
            indices = [idx for idx in indices if idx in per_token]
            if not indices:
                continue
            rows.append(
                {
                    "dataset": dataset_label,
                    "sample": sample,
                    "id": row_id,
                    "method": method,
                    "receiver_prompt_mode": mode,
                    "layer": layer_idx,
                    "token_group": group,
                    "token_count": len(indices),
                    "gold_logit": mean(per_token[idx]["gold_logit"] for idx in indices),
                    "gold_prob": mean(per_token[idx]["gold_prob"] for idx in indices),
                    "gold_rank": mean(per_token[idx]["gold_rank"] for idx in indices),
                    "gold_margin": mean(per_token[idx]["gold_margin"] for idx in indices),
                    "median_gold_rank": median(per_token[idx]["gold_rank"] for idx in indices),
                    "answer_text": str(answer_text),
                    "group_tokens": " ".join(decoded_tokens[idx] for idx in indices),
                }
            )
    return rows


def summarize(rows):
    keys = ["gold_logit", "gold_prob", "gold_rank", "gold_margin", "median_gold_rank", "token_count"]
    groups = {}
    for row in rows:
        key = (
            row["dataset"],
            row["method"],
            row["receiver_prompt_mode"],
            row["token_group"],
            int(row["layer"]),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for (dataset, method, mode, token_group, layer), selected in sorted(groups.items()):
        item = {
            "dataset": dataset,
            "method": method,
            "receiver_prompt_mode": mode,
            "token_group": token_group,
            "layer": layer,
            "n_samples": len({row["sample"] for row in selected}),
            "n_rows": len(selected),
        }
        for key in keys:
            values = [row[key] for row in selected if key in row]
            item[f"mean_{key}"] = mean(values)
            if key in {"gold_rank", "gold_margin"}:
                item[f"median_{key}"] = median(values)
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description="Layerwise logit-lens diagnosis for cross-model KV translation")
    parser.add_argument("--base-experiment-root", required=True)
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", default="native,mse_only,mse_then_ce,paper_rec_then_mixed_generation,q_aware_functional")
    parser.add_argument("--receiver-prompt-modes", default="context_unaware,context_aware")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--critical-mode", choices=["answer", "numeric"], default="answer")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--tokenizer-check-samples", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cpu" and args.dtype in {"float16", "bfloat16"}:
        raise ValueError("float16/bfloat16 on CPU is unsupported for this script")

    add_import_roots(args.base_experiment_root)
    from paper_dense_common import assert_tokenizer_compatible, build_paper_example, load_rows
    from real_kv_common import extract_cache, make_cache
    from real_kv_translator import load_real_translator

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    modes = [item.strip() for item in args.receiver_prompt_modes.split(",") if item.strip()]
    invalid_modes = sorted(set(modes) - {"context_aware", "context_unaware"})
    if invalid_modes:
        raise ValueError(f"Unknown receiver prompt modes: {invalid_modes}")

    rows = load_rows(args.data, args.max_samples)
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(sender_tokenizer, receiver_tokenizer, rows, args.max_source_tokens, args.tokenizer_check_samples)

    sender = load_model(args.sender_model, dtype, device)
    receiver = load_model(args.receiver_model, dtype, device)
    adapters = {}
    adapter_metadata = {}
    for method in methods:
        if method == "native":
            continue
        adapter, metadata = load_real_translator(checkpoint_for(args.base_experiment_root, method), map_location=device)
        adapter = adapter.to(device).eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        adapters[method] = adapter
        adapter_metadata[method] = metadata
    for module in (sender, receiver):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_layer = []
    with torch.no_grad():
        for sample, row in enumerate(tqdm(rows, desc=f"logit_lens:{args.dataset_label}")):
            example = build_paper_example(receiver_tokenizer, row, args.max_source_tokens)
            source_ids = example["source_ids"].to(device)
            sender_pairs = extract_cache(sender(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            native_pairs = extract_cache(receiver(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
            method_pairs = {"native": native_pairs}
            for method, adapter in adapters.items():
                method_pairs[method] = adapter(sender_pairs)

            for method in methods:
                pairs = method_pairs[method]
                for mode in modes:
                    if mode == "context_aware":
                        tail_ids = example["aware_tail_ids"].to(device)
                        prefix_len = example["aware_prefix_len"]
                    else:
                        tail_ids = example["unaware_tail_ids"].to(device)
                        prefix_len = example["unaware_prefix_len"]
                    cache = make_cache(pairs, receiver.config)
                    outputs = receiver(
                        input_ids=tail_ids,
                        past_key_values=cache,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    per_layer.extend(
                        logit_lens_rows(
                            receiver=receiver,
                            hidden_states=outputs.hidden_states,
                            answer_ids=example["answer_ids"].to(device),
                            answer_text=example["answer"],
                            tokenizer=receiver_tokenizer,
                            prefix_len=prefix_len,
                            method=method,
                            mode=mode,
                            sample=sample,
                            row_id=example["id"],
                            dataset_label=args.dataset_label,
                            critical_mode=args.critical_mode,
                        )
                    )

    summary = summarize(per_layer)
    write_jsonl(out / "per_layer.jsonl", per_layer)
    write_csv(out / "layerwise_logit_lens_summary.csv", summary)
    payload = {
        "status": "complete",
        "dataset": args.dataset_label,
        "samples": len(rows),
        "methods": methods,
        "receiver_prompt_modes": modes,
        "args": vars(args),
        "adapter_metadata": adapter_metadata,
        "outputs": {
            "per_layer": "per_layer.jsonl",
            "summary": "layerwise_logit_lens_summary.csv",
        },
        "definitions": {
            "gold_logit": "logit-lens logit assigned to the gold answer token at a teacher-forced answer position",
            "gold_prob": "softmax probability of the gold answer token under logit lens",
            "gold_rank": "1 + number of vocabulary tokens whose logit-lens logit is greater than the gold token logit",
            "gold_margin": "gold token logit minus the best non-gold token logit",
            "context_unaware": "receiver reads translated/native source cache, then consumes only answer prompt plus answer prefix",
            "context_aware": "receiver reads translated/native source cache, then consumes source input plus answer prompt plus answer prefix, matching the mixed-generation paper-style setting",
        },
    }
    with open(out / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
