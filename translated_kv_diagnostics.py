import argparse
import csv
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kv_translators import load_translator


def load_rows(path, limit):
    if limit <= 0:
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            source = (json.loads(line) for line in f if line.strip())
        else:
            source = iter(json.load(f))
        for row in source:
            if row.get("context") and row.get("question") and row.get("answer") is not None:
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def flatten_context(context):
    if isinstance(context, str):
        return context
    return "\n".join(f"[{title}] {' '.join(sentences)}" for title, sentences in context)


def build_example(tokenizer, row, max_context_tokens):
    context_text = f"Context:\n{flatten_context(row['context'])}"
    query_text = f"\n\nQuestion:\n{row['question']}\n\nAnswer:"
    answer_text = " " + str(row["answer"]).strip()
    context_ids = tokenizer(
        context_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_context_tokens,
    ).input_ids
    query_ids = tokenizer(query_text, return_tensors="pt", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False).input_ids
    if answer_ids.shape[1] == 0:
        raise ValueError("Answer tokenization produced no tokens")
    tail_ids = torch.cat([query_ids, answer_ids[:, :-1]], dim=1)
    return {
        "id": row.get("id"),
        "answer": str(row["answer"]),
        "context_ids": context_ids,
        "query_ids": query_ids,
        "answer_ids": answer_ids,
        "tail_ids": tail_ids,
    }


def get_layers(model):
    return model.model.layers


def extract_cache(cache, detach=True, cpu=False):
    pairs = []
    for layer in cache.layers:
        k, v = layer.keys, layer.values
        if detach:
            k, v = k.detach(), v.detach()
        if cpu:
            k, v = k.float().cpu(), v.float().cpu()
        else:
            k, v = k.clone(), v.clone()
        pairs.append((k, v))
    return pairs


def make_cache(pairs, config):
    return DynamicCache(ddp_cache_data=[(k, v) for k, v in pairs], config=config)


class AttentionCapture:
    def __init__(self, model):
        self.weights = {}
        self.handles = []
        for idx, layer in enumerate(get_layers(model)):
            handle = layer.self_attn.register_forward_hook(self._hook(idx), with_kwargs=True)
            self.handles.append(handle)

    def _hook(self, idx):
        def hook(module, args, kwargs, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.weights[idx] = output[1].detach().float().cpu()

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def capture_attention(model):
    capture = AttentionCapture(model)
    try:
        yield capture
    finally:
        capture.close()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def phase_shift_key(k, delta, rope_theta):
    dim = k.shape[-1]
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=k.device, dtype=torch.float32) / dim))
    angle = delta * inv_freq
    cos = torch.cat([angle.cos(), angle.cos()]).to(k.dtype).view(1, 1, 1, dim)
    sin = torch.cat([angle.sin(), angle.sin()]).to(k.dtype).view(1, 1, 1, dim)
    return k * cos + rotate_half(k) * sin


def low_rank_matrix(x, rank):
    shape = x.shape
    rows = x.float().reshape(-1, shape[-1])
    q = min(rank, min(rows.shape))
    if q <= 0:
        return torch.zeros_like(x)
    u, s, vh = torch.linalg.svd(rows, full_matrices=False)
    out = (u[:, :q] * s[:q]) @ vh[:q]
    return out.reshape(shape).to(x.dtype)


def transform_cache(native_pairs, method, args, seed):
    if method == "translator":
        if args.translator is None:
            raise ValueError("method=translator requires --translator-checkpoint")
        return args.translator(native_pairs)
    generator = torch.Generator(device=native_pairs[0][0].device)
    generator.manual_seed(seed)
    out = []
    for layer_idx, (k, v) in enumerate(native_pairs):
        k2, v2 = k.clone(), v.clone()
        if method == "native":
            pass
        elif method == "noise":
            k_scale = k.float().pow(2).mean().sqrt().to(k.dtype)
            v_scale = v.float().pow(2).mean().sqrt().to(v.dtype)
            k2 = k2 + torch.randn(k.shape, generator=generator, device=k.device, dtype=k.dtype) * k_scale * args.noise_sigma
            v2 = v2 + torch.randn(v.shape, generator=generator, device=v.device, dtype=v.dtype) * v_scale * args.noise_sigma
        elif method == "token_shuffle":
            perm = torch.randperm(k.shape[-2], generator=generator, device=k.device)
            k2, v2 = k2[..., perm, :], v2[..., perm, :]
        elif method == "head_shuffle":
            perm = torch.randperm(k.shape[1], generator=generator, device=k.device)
            k2, v2 = k2[:, perm], v2[:, perm]
        elif method == "low_rank":
            k2, v2 = low_rank_matrix(k2, args.rank), low_rank_matrix(v2, args.rank)
        elif method == "rope_shift":
            k2 = phase_shift_key(k2, args.rope_shift, args.rope_theta)
        else:
            raise ValueError(f"Unknown method: {method}")
        out.append((k2, v2))
    return out


def fuse_cache(native_pairs, translated_pairs, alpha):
    return [
        ((1.0 - alpha) * nk + alpha * tk, (1.0 - alpha) * nv + alpha * tv)
        for (nk, nv), (tk, tv) in zip(native_pairs, translated_pairs)
    ]


def cosine_mean(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()


def relative_mse(a, b):
    return ((a.float() - b.float()).pow(2).mean() / (a.float().pow(2).mean() + 1e-12)).item()


def joint_consistency(native_k, native_v, test_k, test_v):
    scores = []
    for head in range(native_k.shape[1]):
        nk = native_k[0, head].float()
        nv = native_v[0, head].float()
        tk = test_k[0, head].float()
        tv = test_v[0, head].float()
        nk, nv = nk - nk.mean(0), nv - nv.mean(0)
        tk, tv = tk - tk.mean(0), tv - tv.mean(0)
        native_cross = (nk.T @ nv) / max(1, nk.shape[0])
        test_cross = (tk.T @ tv) / max(1, tk.shape[0])
        scores.append(F.cosine_similarity(native_cross.flatten(), test_cross.flatten(), dim=0))
    return torch.stack(scores).mean().item()


def topk_overlap(a, b, k):
    k = min(k, a.shape[-1])
    ia = a.topk(k, dim=-1).indices
    ib = b.topk(k, dim=-1).indices
    overlap = (ia.unsqueeze(-1) == ib.unsqueeze(-2)).any(dim=-1).float().mean()
    return overlap.item()


def js_divergence(a, b):
    eps = 1e-12
    a = a.float().clamp_min(eps)
    b = b.float().clamp_min(eps)
    a = a / a.sum(dim=-1, keepdim=True)
    b = b / b.sum(dim=-1, keepdim=True)
    m = 0.5 * (a + b)
    return (0.5 * (a * (a.log() - m.log())).sum(-1) + 0.5 * (b * (b.log() - m.log())).sum(-1)).mean().item()


def attention_metrics(native_weights, test_weights, native_pairs, test_pairs, context_len, query_len, topk):
    rows = []
    n_query_heads = next(iter(native_weights.values())).shape[1]
    n_kv_heads = native_pairs[0][1].shape[1]
    repeats = n_query_heads // n_kv_heads
    for layer, nw in native_weights.items():
        if layer not in test_weights:
            continue
        tw = test_weights[layer]
        max_context = min(context_len, nw.shape[-1], tw.shape[-1])
        n_rows = min(nw.shape[-2], tw.shape[-2])
        nw = nw[..., :n_rows, :max_context]
        tw = tw[..., :n_rows, :max_context]
        native_v = native_pairs[layer][1].repeat_interleave(repeats, dim=1)[..., :max_context, :]
        test_v = test_pairs[layer][1].repeat_interleave(repeats, dim=1)[..., :max_context, :]
        scopes = {
            "query": slice(0, min(query_len, n_rows)),
            "answer": slice(max(0, query_len - 1), n_rows),
            "all": slice(0, n_rows),
        }
        for scope, sl in scopes.items():
            a, b = nw[..., sl, :], tw[..., sl, :]
            if a.shape[-2] == 0:
                continue
            native_out = torch.matmul(a, native_v)
            test_out = torch.matmul(b, test_v)
            rows.append({
                "layer": layer,
                "scope": scope,
                "attention_route_overlap": topk_overlap(a, b, topk),
                "attention_route_js": js_divergence(a, b),
                "attention_output_cos": cosine_mean(native_out, test_out),
            })
    return rows


def cache_metrics(native_pairs, test_pairs):
    rows = []
    for layer, ((nk, nv), (tk, tv)) in enumerate(zip(native_pairs, test_pairs)):
        rows.append({
            "layer": layer,
            "kv_mse": 0.5 * (F.mse_loss(tk.float(), nk.float()).item() + F.mse_loss(tv.float(), nv.float()).item()),
            "kv_relative_mse": 0.5 * (relative_mse(nk, tk) + relative_mse(nv, tv)),
            "k_cos": cosine_mean(nk, tk),
            "v_cos": cosine_mean(nv, tv),
            "kv_joint_consistency": joint_consistency(nk, nv, tk, tv),
        })
    return rows


def answer_logits(logits, query_len, answer_len):
    start = query_len - 1
    return logits[:, start : start + answer_len]


def distribution_metrics(native_logits, test_logits, targets):
    n = min(native_logits.shape[1], test_logits.shape[1], targets.shape[1])
    native_logits = native_logits[:, :n].float()
    test_logits = test_logits[:, :n].float()
    targets = targets[:, :n]
    native_ce = F.cross_entropy(native_logits.reshape(-1, native_logits.shape[-1]), targets.reshape(-1))
    test_ce = F.cross_entropy(test_logits.reshape(-1, test_logits.shape[-1]), targets.reshape(-1))
    log_p_test = F.log_softmax(test_logits, dim=-1)
    p_native = F.softmax(native_logits, dim=-1)
    return {
        "native_ce": native_ce.item(),
        "translated_ce": test_ce.item(),
        "ce_delta": (test_ce - native_ce).item(),
        "logit_kl": max(0.0, F.kl_div(log_p_test, p_native, reduction="batchmean").item() / n),
        "top1_match": (native_logits.argmax(-1) == test_logits.argmax(-1)).float().mean().item(),
    }


def normalize_answer(text):
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(prediction, gold):
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((min(p.count(token), g.count(token)) for token in set(p) & set(g)))
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def greedy_generate(model, tokenizer, context_pairs, query_ids, config, max_new_tokens):
    cache = make_cache(context_pairs, config)
    out = model(input_ids=query_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    generated = []
    token = out.logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(max_new_tokens):
        generated.append(token)
        if token.item() == tokenizer.eos_token_id:
            break
        out = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
        token = out.logits[:, -1].argmax(-1, keepdim=True)
    ids = torch.cat(generated, dim=1) if generated else query_ids[:, :0]
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def mean_metric(rows, key, scope=None):
    values = [r[key] for r in rows if key in r and (scope is None or r.get("scope") == scope)]
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def run_sample(model, tokenizer, example, methods, args, device, sample_idx):
    context_ids = example["context_ids"].to(device)
    query_ids = example["query_ids"].to(device)
    tail_ids = example["tail_ids"].to(device)
    answer_ids = example["answer_ids"].to(device)
    context_out = model(input_ids=context_ids, use_cache=True, logits_to_keep=1)
    native_device = extract_cache(context_out.past_key_values, detach=True, cpu=False)
    native_cpu = [(k.float().cpu(), v.float().cpu()) for k, v in native_device]

    with capture_attention(model) as native_capture:
        native_tail_cache = make_cache(native_device, model.config)
        native_tail = model(input_ids=tail_ids, past_key_values=native_tail_cache, use_cache=True)
    native_logits = answer_logits(native_tail.logits, query_ids.shape[1], answer_ids.shape[1]).float().cpu()

    full_ids = torch.cat([context_ids, tail_ids], dim=1)
    full = model(input_ids=full_ids, use_cache=False)
    start = context_ids.shape[1] + query_ids.shape[1] - 1
    full_logits = full.logits[:, start : start + answer_ids.shape[1]].float().cpu()
    n = min(full_logits.shape[1], native_logits.shape[1])
    equivalence = {
        "max_abs_logit": (full_logits[:, :n] - native_logits[:, :n]).abs().max().item(),
        "top1_match": (full_logits[:, :n].argmax(-1) == native_logits[:, :n].argmax(-1)).float().mean().item(),
    }

    results, layer_rows = [], []
    for method_idx, method in enumerate(methods):
        translated = transform_cache(native_device, method, args, args.seed + sample_idx * 1000 + method_idx)
        alphas = [0.0] if method == "native" else args.residual_alphas
        for alpha in alphas:
            test_device = fuse_cache(native_device, translated, alpha)
            test_cpu = [(k.detach().float().cpu(), v.detach().float().cpu()) for k, v in test_device]
            with capture_attention(model) as test_capture:
                test_cache = make_cache(test_device, model.config)
                test_tail = model(input_ids=tail_ids, past_key_values=test_cache, use_cache=True)
            test_logits = answer_logits(test_tail.logits, query_ids.shape[1], answer_ids.shape[1]).float().cpu()
            kv_rows = cache_metrics(native_cpu, test_cpu)
            attn_rows = attention_metrics(
                native_capture.weights,
                test_capture.weights,
                native_cpu,
                test_cpu,
                context_ids.shape[1],
                query_ids.shape[1],
                args.attention_topk,
            )
            label = "native" if method == "native" else f"{method}@alpha={alpha:g}"
            for row in kv_rows:
                layer_rows.append({"sample": sample_idx, "id": example["id"], "method": label, "kind": "cache", **row})
            for row in attn_rows:
                layer_rows.append({"sample": sample_idx, "id": example["id"], "method": label, "kind": "attention", **row})
            row = {
                "sample": sample_idx,
                "id": example["id"],
                "method": label,
                **distribution_metrics(native_logits, test_logits, answer_ids.cpu()),
                "kv_mse": mean_metric(kv_rows, "kv_mse"),
                "kv_relative_mse": mean_metric(kv_rows, "kv_relative_mse"),
                "k_cos": mean_metric(kv_rows, "k_cos"),
                "v_cos": mean_metric(kv_rows, "v_cos"),
                "kv_joint_consistency": mean_metric(kv_rows, "kv_joint_consistency"),
                "attention_route_overlap": mean_metric(attn_rows, "attention_route_overlap", "answer"),
                "attention_route_js": mean_metric(attn_rows, "attention_route_js", "answer"),
                "attention_output_cos": mean_metric(attn_rows, "attention_output_cos", "answer"),
            }
            if args.max_new_tokens > 0:
                prediction = greedy_generate(model, tokenizer, test_device, query_ids, model.config, args.max_new_tokens)
                row.update({
                    "prediction": prediction,
                    "answer": example["answer"],
                    "answer_em": float(normalize_answer(prediction) == normalize_answer(example["answer"])),
                    "answer_f1": answer_f1(prediction, example["answer"]),
                })
            results.append(row)
    return equivalence, results, layer_rows


def summarize(rows):
    keys = [
        "kv_mse", "kv_relative_mse", "k_cos", "v_cos", "logit_kl", "ce_delta", "top1_match",
        "attention_route_overlap", "attention_route_js", "attention_output_cos", "kv_joint_consistency",
        "answer_em", "answer_f1",
    ]
    output = []
    rng = np.random.default_rng(1234)
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary = {"method": method, "n": len(selected)}
        for key in keys:
            values = [row[key] for row in selected if key in row and np.isfinite(row[key])]
            if values:
                summary[key] = float(np.mean(values))
                if len(values) > 1:
                    samples = rng.choice(values, size=(1000, len(values)), replace=True).mean(axis=1)
                    summary[f"{key}_ci95_low"] = float(np.percentile(samples, 2.5))
                    summary[f"{key}_ci95_high"] = float(np.percentile(samples, 97.5))
        output.append(summary)
    return output


def parse_float_list(text):
    return [float(item) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Diagnose translated-like KV inside one receiver model")
    parser.add_argument("--model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B")
    parser.add_argument("--data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl")
    parser.add_argument("--out", default="runs/translated_kv_diagnostics")
    parser.add_argument("--methods", default="native,noise,token_shuffle,head_shuffle,low_rank,rope_shift")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--noise-sigma", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--rope-shift", type=float, default=4.0)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--residual-alphas", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--attention-topk", type=int, default=16)
    parser.add_argument("--translator-checkpoint")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--equivalence-atol", type=float, default=None)
    args = parser.parse_args()
    args.residual_alphas = parse_float_list(args.residual_alphas)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "cpu" and args.dtype == "float16":
        raise ValueError("float16 on CPU is unsupported for this experiment")
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    args.translator = None
    args.translator_metadata = None
    if args.translator_checkpoint:
        args.translator, args.translator_metadata = load_translator(args.translator_checkpoint, map_location=device)
        args.translator = args.translator.to(device).eval()
    rows = load_rows(args.data, args.max_samples)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results, all_layers, equivalence = [], [], []
    for idx, row in enumerate(tqdm(rows, desc="samples")):
        example = build_example(tokenizer, row, args.max_context_tokens)
        eq, result_rows, layer_rows = run_sample(model, tokenizer, example, methods, args, device, idx)
        equivalence.append({"sample": idx, "id": example["id"], **eq})
        all_results.extend(result_rows)
        all_layers.extend(layer_rows)
    equivalence_atol = args.equivalence_atol
    if equivalence_atol is None:
        equivalence_atol = 1e-3 if args.dtype == "float32" else 0.25
    for row in equivalence:
        row["atol"] = equivalence_atol
        row["passed"] = row["max_abs_logit"] <= equivalence_atol and row["top1_match"] == 1.0
    summary = summarize(all_results)
    with open(out_dir / "per_example.jsonl", "w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "per_layer.jsonl", "w", encoding="utf-8") as f:
        for row in all_layers:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        serializable_args = {key: value for key, value in vars(args).items() if key != "translator"}
        json.dump({"args": serializable_args, "equivalence": equivalence, "diagnostic_table": summary}, f, indent=2, ensure_ascii=False)
    if summary:
        fieldnames = sorted({key for row in summary for key in row})
        with open(out_dir / "diagnostic_table.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)
    print(json.dumps({"equivalence": equivalence, "diagnostic_table": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
