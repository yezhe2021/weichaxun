import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
try:
    from transformers.utils import import_utils as _transformers_import_utils

    _transformers_import_utils._torchvision_available = False
except Exception:
    pass


STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "and", "or", "is", "are", "was", "were",
    "by", "for", "with", "as", "from", "that", "this", "it", "be", "at", "which", "who",
}


def get_layers(model):
    base = getattr(model, "model", model)
    return base.layers


def self_attn(model, layer):
    return get_layers(model)[layer].self_attn


class LayerInputCapture:
    def __init__(self, modules):
        self.inputs = {}
        self.handles = []
        for idx, module in modules:
            self.handles.append(module.register_forward_pre_hook(self._make_hook(idx), with_kwargs=True))

    def _make_hook(self, idx):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            self.inputs[idx] = hidden.detach().float().cpu()

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def ranks(x):
    order = np.argsort(x)
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return corr(ranks(a), ranks(b))


def topk_overlap(a, b, ks):
    out = {}
    order_a = np.argsort(-np.asarray(a))
    order_b = np.argsort(-np.asarray(b))
    n = len(a)
    for k in ks:
        kk = min(k, n)
        if kk <= 0:
            continue
        out[str(k)] = len(set(order_a[:kk]).intersection(order_b[:kk])) / kk
    return out


def rbo(a, b, p=0.9):
    order_a = np.argsort(-np.asarray(a))
    order_b = np.argsort(-np.asarray(b))
    depth = min(len(order_a), len(order_b))
    score = 0.0
    seen_a, seen_b = set(), set()
    for d in range(1, depth + 1):
        seen_a.add(int(order_a[d - 1]))
        seen_b.add(int(order_b[d - 1]))
        score += (len(seen_a.intersection(seen_b)) / d) * (p ** (d - 1))
    return float((1 - p) * score)


def gini(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    x = np.abs(x)
    if np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1) * x)) / (n * np.sum(x))) - (n + 1) / n)


def entropy_from_positive(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0)
    s = x.sum()
    if s <= 0:
        return float("nan")
    p = x / s
    return float(-(p * np.log(p + 1e-12)).sum())


def kurtosis(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4 or np.std(x) == 0:
        return float("nan")
    z = (x - x.mean()) / x.std()
    return float(np.mean(z ** 4))


def hill_tail_index(x, frac=0.1):
    x = np.sort(np.asarray(x, dtype=np.float64))
    x = x[x > 0]
    if len(x) < 10:
        return float("nan")
    k = max(2, int(len(x) * frac))
    tail = x[-k:]
    xmin = tail[0]
    denom = np.mean(np.log(tail / max(xmin, 1e-12)))
    return float(1.0 / denom) if denom > 0 else float("inf")


def tail_stats(x):
    x = np.asarray(x, dtype=np.float64)
    p50 = np.percentile(x, 50)
    p95 = np.percentile(x, 95)
    p99 = np.percentile(x, 99)
    return {
        "p95_p50": float(p95 / (p50 + 1e-12)),
        "p99_p50": float(p99 / (p50 + 1e-12)),
        "max_p99": float(np.max(x) / (p99 + 1e-12)),
        "kurtosis": kurtosis(x),
        "outlier_ratio_p99": float(np.mean(x > p99)),
        "hill_tail_index": hill_tail_index(x),
        "gini": gini(x),
        "entropy": entropy_from_positive(x),
        "top10pct_mass": float(np.sort(np.maximum(x, 0))[-max(1, len(x) // 10):].sum() / (np.maximum(x, 0).sum() + 1e-12)),
    }


def effective_rank(matrix):
    x = torch.as_tensor(matrix, dtype=torch.float32)
    if x.ndim != 2 or min(x.shape) < 2:
        return {}
    s = torch.linalg.svdvals(x).cpu().numpy()
    ev = s ** 2
    p = ev / (ev.sum() + 1e-12)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "effective_rank": float(math.exp(entropy)),
        "spectral_entropy": entropy,
        "participation_ratio": float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12)),
        "top1_evr": float(p[0]),
        "top5_evr": float(p[:5].sum()),
    }


def word_spans(text):
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE)]


def token_type(word, question_words):
    w = word.strip()
    lw = w.lower()
    if not w:
        return "empty"
    if lw in question_words:
        return "question_related"
    if re.fullmatch(r"\d+([.,:/-]\d+)*", w):
        return "number"
    if re.fullmatch(r"[^\w\s]+", w):
        return "punct"
    if lw in STOPWORDS:
        return "stopword"
    if w[:1].isupper() and len(w) > 1:
        return "entity_like"
    if lw.endswith("ing") or lw.endswith("ed"):
        return "verb_like"
    return "content"


def aggregate_to_words(token_values, offsets, spans):
    vals = []
    for start, end, _ in spans:
        idx = [i for i, (a, b) in enumerate(offsets) if max(a, start) < min(b, end)]
        vals.append(float(np.mean(token_values[idx])) if idx else float("nan"))
    return np.asarray(vals, dtype=np.float64)


def aggregate_matrix_to_words(token_matrix, offsets, spans):
    vals = []
    for start, end, _ in spans:
        idx = [i for i, (a, b) in enumerate(offsets) if max(a, start) < min(b, end)]
        if idx:
            vals.append(token_matrix[idx].mean(axis=0))
    return np.asarray(vals, dtype=np.float32) if vals else np.empty((0, token_matrix.shape[-1]), dtype=np.float32)


@torch.no_grad()
def collect_model(model_path, text, layers, max_length, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, trust_remote_code=True).to(device).eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    valid_layers = [l for l in layers if l < len(get_layers(model))]
    cap = LayerInputCapture([(l, self_attn(model, l)) for l in valid_layers])
    model(**enc, use_cache=False)
    cap.close()
    out = {"offsets": offsets, "layers": {}, "num_layers": len(get_layers(model))}
    for l in valid_layers:
        hidden = cap.inputs[l].to(device)
        attn = self_attn(model, l)
        k = attn.k_proj(hidden)[0].float().cpu().numpy()
        v = attn.v_proj(hidden)[0].float().cpu().numpy()
        out["layers"][l] = {
            "k": k,
            "v": v,
            "k_norm": np.linalg.norm(k, axis=-1),
            "v_norm": np.linalg.norm(v, axis=-1),
        }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def distance_corr(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a = a[:n]
    b = b[:n]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    da = 1 - a @ a.T
    db = 1 - b @ b.T
    tri = np.triu_indices(n, k=1)
    return corr(da[tri], db[tri])


def linear_cka(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a = a[:n] - a[:n].mean(axis=0, keepdims=True)
    b = b[:n] - b[:n].mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(a.T @ b, ord="fro") ** 2
    denom = np.linalg.norm(a.T @ a, ord="fro") * np.linalg.norm(b.T @ b, ord="fro")
    return float(hsic / (denom + 1e-12))


def analyze_pair(a, b, text, layers):
    spans = word_spans(text)
    question_words = {w.lower() for _, _, w in spans[-64:]}
    layer_rows = []
    trajectories = defaultdict(lambda: {"a_k": [], "b_k": [], "a_v": [], "b_v": []})
    type_rows = defaultdict(list)
    for l in layers:
        if l not in a["layers"] or l not in b["layers"]:
            continue
        row = {"layer": l}
        for kv in ["k", "v"]:
            wa = aggregate_to_words(a["layers"][l][f"{kv}_norm"], a["offsets"], spans)
            wb = aggregate_to_words(b["layers"][l][f"{kv}_norm"], b["offsets"], spans)
            mask = np.isfinite(wa) & np.isfinite(wb)
            wa, wb = wa[mask], wb[mask]
            row[f"{kv}_spearman"] = spearman(wa, wb)
            row[f"{kv}_rbo"] = rbo(wa, wb)
            row[f"{kv}_top_overlap"] = topk_overlap(wa, wb, [5, 10, 20, 50])
            row[f"{kv}_tail_a"] = tail_stats(a["layers"][l][f"{kv}_norm"])
            row[f"{kv}_tail_b"] = tail_stats(b["layers"][l][f"{kv}_norm"])
            ma = aggregate_matrix_to_words(a["layers"][l][kv], a["offsets"], spans)
            mb = aggregate_matrix_to_words(b["layers"][l][kv], b["offsets"], spans)
            row[f"{kv}_spectral_a"] = effective_rank(a["layers"][l][kv])
            row[f"{kv}_spectral_b"] = effective_rank(b["layers"][l][kv])
            row[f"{kv}_rsa_corr"] = distance_corr(ma, mb)
            row[f"{kv}_cka"] = linear_cka(ma, mb)
            for idx, (_, _, word) in enumerate(np.asarray(spans)[mask]):
                typ = token_type(str(word), question_words)
                type_rows[(l, kv, typ)].append((wa[idx], wb[idx]))
            for idx, (_, _, word) in enumerate(np.asarray(spans)[mask]):
                trajectories[str(word)][f"a_{kv}"].append(float(wa[idx]))
                trajectories[str(word)][f"b_{kv}"].append(float(wb[idx]))
        ak = aggregate_to_words(a["layers"][l]["k_norm"], a["offsets"], spans)
        av = aggregate_to_words(a["layers"][l]["v_norm"], a["offsets"], spans)
        bk = aggregate_to_words(b["layers"][l]["k_norm"], b["offsets"], spans)
        bv = aggregate_to_words(b["layers"][l]["v_norm"], b["offsets"], spans)
        mask_a = np.isfinite(ak) & np.isfinite(av)
        mask_b = np.isfinite(bk) & np.isfinite(bv)
        row["kv_coupling_a"] = {
            "norm_corr": corr(ak[mask_a], av[mask_a]),
            "top10_overlap": topk_overlap(ak[mask_a], av[mask_a], [10]).get("10", float("nan")),
            "ratio_mean": float(np.mean(ak[mask_a] / (av[mask_a] + 1e-12))),
        }
        row["kv_coupling_b"] = {
            "norm_corr": corr(bk[mask_b], bv[mask_b]),
            "top10_overlap": topk_overlap(bk[mask_b], bv[mask_b], [10]).get("10", float("nan")),
            "ratio_mean": float(np.mean(bk[mask_b] / (bv[mask_b] + 1e-12))),
        }
        layer_rows.append(row)

    traj_rows = []
    for word, vals in trajectories.items():
        if len(vals["a_k"]) >= 3:
            traj_rows.append({
                "word": word,
                "k_layer_corr": corr(vals["a_k"], vals["b_k"]),
                "v_layer_corr": corr(vals["a_v"], vals["b_v"]),
                "k_peak_a": int(np.argmax(vals["a_k"])),
                "k_peak_b": int(np.argmax(vals["b_k"])),
                "v_peak_a": int(np.argmax(vals["a_v"])),
                "v_peak_b": int(np.argmax(vals["b_v"])),
            })
    token_type_summary = []
    for (l, kv, typ), pairs in type_rows.items():
        arr = np.asarray(pairs)
        token_type_summary.append({
            "layer": l,
            "kv": kv,
            "type": typ,
            "count": int(len(arr)),
            "mean_a": float(arr[:, 0].mean()),
            "mean_b": float(arr[:, 1].mean()),
            "spearman": spearman(arr[:, 0], arr[:, 1]) if len(arr) > 2 else float("nan"),
        })
    return {"layers": layer_rows, "trajectories": traj_rows[:200], "token_types": token_type_summary}


def summarize(results):
    rows = [r for sample in results for r in sample["analysis"]["layers"]]
    out = {}
    for key in ["k_spearman", "v_spearman", "k_rbo", "v_rbo", "k_rsa_corr", "v_rsa_corr", "k_cka", "v_cka"]:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    for kv in ["k", "v"]:
        for model_key in ["a", "b"]:
            vals = [r[f"{kv}_tail_{model_key}"]["gini"] for r in rows if f"{kv}_tail_{model_key}" in r]
            out[f"{kv}_gini_{model_key}"] = float(np.mean(vals)) if vals else float("nan")
            er = [r[f"{kv}_spectral_{model_key}"].get("effective_rank", float("nan")) for r in rows if f"{kv}_spectral_{model_key}" in r]
            out[f"{kv}_effective_rank_{model_key}"] = float(np.nanmean(er)) if er else float("nan")
    return out


def load_texts(path, limit):
    if path.endswith(".jsonl"):
        texts = []
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            text = f"Context:\n{row.get('context','')}\n\nQuestion:\n{row.get('question','')}\n\nAnswer:"
            texts.append(text)
            if len(texts) >= limit:
                break
        return texts
    rows = json.load(open(path, encoding="utf-8"))
    texts = []
    for row in rows:
        if "context" in row and "question" in row:
            ctx = row["context"]
            if isinstance(ctx, list):
                ctx = "\n".join(f"[{t}] {' '.join(s)}" for t, s in ctx)
            texts.append(f"Context:\n{ctx}\n\nQuestion:\n{row['question']}\n\nAnswer:")
        if len(texts) >= limit:
            break
    return texts


def parse_layers(text):
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-a", default="Qwen3-0.6B")
    p.add_argument("--model-b", default="/home/yezhe/all_models/models/LLM-Research/Llama-3___2-1B-Instruct")
    p.add_argument("--data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl")
    p.add_argument("--out", default="runs/kv_invariants_qwen_llama")
    p.add_argument("--layers", default="0,4,8,12,15")
    p.add_argument("--max-samples", type=int, default=4)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    layers = parse_layers(args.layers)
    texts = load_texts(args.data, args.max_samples)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    results = []
    for i, text in enumerate(tqdm(texts, desc="samples")):
        a = collect_model(args.model_a, text, layers, args.max_length, device)
        b = collect_model(args.model_b, text, layers, args.max_length, device)
        common_layers = [l for l in layers if l in a["layers"] and l in b["layers"]]
        results.append({"sample": i, "text_chars": len(text), "analysis": analyze_pair(a, b, text, common_layers)})
    summary = summarize(results)
    with open(Path(args.out) / "kv_invariants.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
