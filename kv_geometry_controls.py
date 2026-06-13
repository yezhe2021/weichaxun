import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kv_invariant_probe import (
    aggregate_matrix_to_words,
    collect_model,
    corr,
    distance_corr,
    linear_cka,
    load_texts,
    parse_layers,
    word_spans,
)


def cosine_distance_matrix(x):
    x = np.asarray(x, dtype=np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return 1.0 - x @ x.T


def upper_tri(x):
    idx = np.triu_indices(x.shape[0], k=1)
    return x[idx]


def position_residual_rsa(a, b):
    n = min(len(a), len(b))
    if n < 4:
        return float("nan")
    da = cosine_distance_matrix(a[:n])
    db = cosine_distance_matrix(b[:n])
    pos = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(np.float64)
    y_a = upper_tri(da)
    y_b = upper_tri(db)
    x = upper_tri(pos)
    x = np.stack([np.ones_like(x), x], axis=1)
    beta_a = np.linalg.lstsq(x, y_a, rcond=None)[0]
    beta_b = np.linalg.lstsq(x, y_b, rcond=None)[0]
    return corr(y_a - x @ beta_a, y_b - x @ beta_b)


def knn_overlap(a, b, k=10):
    n = min(len(a), len(b))
    if n <= k + 1:
        return float("nan")
    da = cosine_distance_matrix(a[:n])
    db = cosine_distance_matrix(b[:n])
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    overlaps = []
    for i in range(n):
        na = set(np.argsort(da[i])[:k].tolist())
        nb = set(np.argsort(db[i])[:k].tolist())
        overlaps.append(len(na & nb) / k)
    return float(np.mean(overlaps))


def layer_word_mats(collected, text, layers):
    spans = word_spans(text)
    out = {}
    for layer in layers:
        if layer not in collected["layers"]:
            continue
        out[layer] = {}
        for kv in ["k", "v"]:
            out[layer][kv] = aggregate_matrix_to_words(
                collected["layers"][layer][kv],
                collected["offsets"],
                spans,
            )
    return out


def compare_mats(a, b, knn_k):
    n = min(len(a), len(b))
    if n < 4:
        return {}
    a = a[:n]
    b = b[:n]
    rng = np.random.default_rng(1234)
    shuffled = b[rng.permutation(n)]
    return {
        "raw_rsa": distance_corr(a, b),
        "shuffled_rsa": distance_corr(a, shuffled),
        "position_residual_rsa": position_residual_rsa(a, b),
        "cka": linear_cka(a, b),
        "knn_overlap": knn_overlap(a, b, knn_k),
        "shuffled_knn_overlap": knn_overlap(a, shuffled, knn_k),
        "n_words": int(n),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-a", default="Qwen3-0.6B")
    p.add_argument("--model-b", default="/home/yezhe/all_models/models/LLM-Research/Llama-3___2-1B-Instruct")
    p.add_argument("--data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl")
    p.add_argument("--out", default="runs/kv_geometry_controls_qwen_llama")
    p.add_argument("--layers-a", default="0,4,8,12,15")
    p.add_argument("--layers-b", default="0,4,8,12,15")
    p.add_argument("--max-samples", type=int, default=2)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    layers_a = parse_layers(args.layers_a)
    layers_b = parse_layers(args.layers_b)
    all_layers = sorted(set(layers_a + layers_b))
    texts = load_texts(args.data, args.max_samples)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    collected_a = []
    collected_b = []
    mats_a = []
    mats_b = []
    for text in tqdm(texts, desc="collect"):
        ca = collect_model(args.model_a, text, all_layers, args.max_length, device)
        cb = collect_model(args.model_b, text, all_layers, args.max_length, device)
        collected_a.append(ca)
        collected_b.append(cb)
        mats_a.append(layer_word_mats(ca, text, all_layers))
        mats_b.append(layer_word_mats(cb, text, all_layers))

    same_text = []
    heatmap = []
    for s, text in enumerate(texts):
        for la in layers_a:
            for lb in layers_b:
                for kv in ["k", "v"]:
                    if la not in mats_a[s] or lb not in mats_b[s]:
                        continue
                    row = compare_mats(mats_a[s][la][kv], mats_b[s][lb][kv], args.knn_k)
                    row.update({"sample": s, "kv": kv, "layer_a": la, "layer_b": lb})
                    heatmap.append(row)
                    if la == lb:
                        same_text.append(row)

    different_text = []
    if len(texts) > 1:
        for s in range(len(texts)):
            t = (s + 1) % len(texts)
            for layer in sorted(set(layers_a) & set(layers_b)):
                for kv in ["k", "v"]:
                    if layer in mats_a[s] and layer in mats_b[t]:
                        row = compare_mats(mats_a[s][layer][kv], mats_b[t][layer][kv], args.knn_k)
                        row.update({"sample_a": s, "sample_b": t, "kv": kv, "layer": layer})
                        different_text.append(row)

    within_model = []
    for s in range(len(texts)):
        for model_name, mats in [("a", mats_a[s]), ("b", mats_b[s])]:
            layers = layers_a if model_name == "a" else layers_b
            for i, li in enumerate(layers):
                for lj in layers[i + 1 :]:
                    for kv in ["k", "v"]:
                        if li in mats and lj in mats:
                            row = compare_mats(mats[li][kv], mats[lj][kv], args.knn_k)
                            row.update({"sample": s, "model": model_name, "kv": kv, "layer_i": li, "layer_j": lj})
                            within_model.append(row)

    def mean(rows, key, kv=None):
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key]) and (kv is None or r.get("kv") == kv)]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "same_text_raw_rsa_k": mean(same_text, "raw_rsa", "k"),
        "same_text_raw_rsa_v": mean(same_text, "raw_rsa", "v"),
        "shuffle_rsa_k": mean(same_text, "shuffled_rsa", "k"),
        "shuffle_rsa_v": mean(same_text, "shuffled_rsa", "v"),
        "position_residual_rsa_k": mean(same_text, "position_residual_rsa", "k"),
        "position_residual_rsa_v": mean(same_text, "position_residual_rsa", "v"),
        "same_text_knn_k": mean(same_text, "knn_overlap", "k"),
        "same_text_knn_v": mean(same_text, "knn_overlap", "v"),
        "shuffle_knn_k": mean(same_text, "shuffled_knn_overlap", "k"),
        "shuffle_knn_v": mean(same_text, "shuffled_knn_overlap", "v"),
        "different_text_rsa_k": mean(different_text, "raw_rsa", "k"),
        "different_text_rsa_v": mean(different_text, "raw_rsa", "v"),
        "within_model_rsa_k": mean(within_model, "raw_rsa", "k"),
        "within_model_rsa_v": mean(within_model, "raw_rsa", "v"),
    }

    with open(Path(args.out) / "geometry_controls.json", "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "summary": summary,
            "same_text": same_text,
            "different_text": different_text,
            "within_model": within_model,
            "heatmap": heatmap,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
