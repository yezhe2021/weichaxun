import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_invariant_probe import (
    LayerInputCapture,
    aggregate_matrix_to_words,
    corr,
    distance_corr,
    get_layers,
    linear_cka,
    load_texts,
    parse_layers,
    self_attn,
    word_spans,
)


def cosine_distance_matrix(x):
    x = np.asarray(x, dtype=np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return 1.0 - x @ x.T


def upper_tri(x):
    idx = np.triu_indices(x.shape[0], k=1)
    return x[idx]


def pair_features(n):
    idx = np.triu_indices(n, k=1)
    i, j = idx
    dist = np.abs(i - j).astype(np.float64)
    sent = (i // 24 == j // 24).astype(np.float64)
    para = (i // 96 == j // 96).astype(np.float64)
    local5 = (dist <= 5).astype(np.float64)
    local10 = (dist <= 10).astype(np.float64)
    return idx, np.stack([
        np.ones_like(dist),
        dist,
        np.log1p(dist),
        sent,
        para,
        local5,
        local10,
    ], axis=1), dist


def residual_rsa(a, b):
    n = min(len(a), len(b))
    if n < 4:
        return float("nan")
    da = cosine_distance_matrix(a[:n])
    db = cosine_distance_matrix(b[:n])
    idx, x, _ = pair_features(n)
    y_a = da[idx]
    y_b = db[idx]
    beta_a = np.linalg.lstsq(x, y_a, rcond=None)[0]
    beta_b = np.linalg.lstsq(x, y_b, rcond=None)[0]
    return corr(y_a - x @ beta_a, y_b - x @ beta_b)


def far_rsa(a, b, min_gap):
    n = min(len(a), len(b))
    if n < min_gap + 4:
        return float("nan")
    da = cosine_distance_matrix(a[:n])
    db = cosine_distance_matrix(b[:n])
    idx, _, dist = pair_features(n)
    mask = dist > min_gap
    return corr(da[idx][mask], db[idx][mask])


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


def collect_loaded_model(model, tokenizer, text, layers, max_length, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    valid_layers = [l for l in layers if l < len(get_layers(model))]
    cap = LayerInputCapture([(l, self_attn(model, l)) for l in valid_layers])
    with torch.no_grad():
        model(**enc, use_cache=False)
    cap.close()
    out = {"offsets": offsets, "layers": {}, "num_layers": len(get_layers(model))}
    for l in valid_layers:
        hidden = cap.inputs[l].to(device)
        attn = self_attn(model, l)
        k = attn.k_proj(hidden)[0].detach().float().cpu().numpy()
        v = attn.v_proj(hidden)[0].detach().float().cpu().numpy()
        out["layers"][l] = {"k": k, "v": v}
    return out


def shuffled_knn_from_mats(da, db, perm, k):
    n = da.shape[0]
    pdb = db[np.ix_(perm, perm)]
    overlaps = []
    for i in range(n):
        na = set(np.argsort(da[i])[:k].tolist())
        nb = set(np.argsort(pdb[i])[:k].tolist())
        overlaps.append(len(na & nb) / k)
    return float(np.mean(overlaps))


def shuffle_stats_from_distance(da, db, knn_k, n_shuffles, raw_rsa, raw_knn):
    n = da.shape[0]
    idx = np.triu_indices(n, k=1)
    a_vec = da[idx]
    rng = np.random.default_rng(1234)
    rsa_vals = []
    knn_vals = []
    for _ in range(n_shuffles):
        perm = rng.permutation(n)
        pdb = db[np.ix_(perm, perm)]
        rsa_vals.append(corr(a_vec, pdb[idx]))
        knn_vals.append(shuffled_knn_from_mats(da, db, perm, knn_k))
    return {
        "shuffled_rsa_mean": float(np.nanmean(rsa_vals)),
        "shuffled_rsa_std": float(np.nanstd(rsa_vals)),
        "shuffled_rsa_p_ge_real": float(np.mean(np.asarray(rsa_vals) >= raw_rsa)),
        "shuffled_knn_mean": float(np.nanmean(knn_vals)),
        "shuffled_knn_std": float(np.nanstd(knn_vals)),
        "shuffled_knn_p_ge_real": float(np.mean(np.asarray(knn_vals) >= raw_knn)),
    }


def shuffle_stats(a, b, knn_k, n_shuffles):
    n = min(len(a), len(b))
    rng = np.random.default_rng(1234)
    rsa_vals = []
    knn_vals = []
    for _ in range(n_shuffles):
        shuffled = b[rng.permutation(n)]
        rsa_vals.append(distance_corr(a, shuffled))
        knn_vals.append(knn_overlap(a, shuffled, knn_k))
    return {
        "shuffled_rsa_mean": float(np.nanmean(rsa_vals)),
        "shuffled_rsa_std": float(np.nanstd(rsa_vals)),
        "shuffled_rsa_p_ge_real": None,
        "shuffled_knn_mean": float(np.nanmean(knn_vals)),
        "shuffled_knn_std": float(np.nanstd(knn_vals)),
        "shuffled_knn_p_ge_real": None,
    }, rsa_vals, knn_vals


def compare_mats(a, b, knn_k, n_shuffles):
    n = min(len(a), len(b))
    if n < 4:
        return {}
    a = a[:n]
    b = b[:n]
    da = cosine_distance_matrix(a)
    db = cosine_distance_matrix(b)
    raw = corr(upper_tri(da), upper_tri(db))
    knn = knn_overlap(a, b, knn_k)
    shuf = shuffle_stats_from_distance(da, db, knn_k, n_shuffles, raw, knn)
    row = {
        "raw_rsa": raw,
        "position_residual_rsa": residual_rsa(a, b),
        "far_rsa_gap5": far_rsa(a, b, 5),
        "far_rsa_gap10": far_rsa(a, b, 10),
        "cka": linear_cka(a, b),
        "knn_overlap": knn,
        "n_words": int(n),
    }
    row.update(shuf)
    return row


def compare_basic(a, b, knn_k):
    n = min(len(a), len(b))
    if n < 4:
        return {}
    a = a[:n]
    b = b[:n]
    return {
        "raw_rsa": distance_corr(a, b),
        "cka": linear_cka(a, b),
        "knn_overlap": knn_overlap(a, b, knn_k),
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
    p.add_argument("--n-shuffles", type=int, default=20)
    p.add_argument("--full-heatmap", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    layers_a = parse_layers(args.layers_a)
    layers_b = parse_layers(args.layers_b)
    all_layers = sorted(set(layers_a + layers_b))
    texts = load_texts(args.data, args.max_samples)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    tok_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    tok_b = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=True)
    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, dtype=torch.float32, trust_remote_code=True).to(device).eval()
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, dtype=torch.float32, trust_remote_code=True).to(device).eval()

    mats_a = []
    mats_b = []
    for text in tqdm(texts, desc="collect"):
        ca = collect_loaded_model(model_a, tok_a, text, all_layers, args.max_length, device)
        cb = collect_loaded_model(model_b, tok_b, text, all_layers, args.max_length, device)
        mats_a.append(layer_word_mats(ca, text, all_layers))
        mats_b.append(layer_word_mats(cb, text, all_layers))

    same_text = []
    heatmap = []
    for s, text in enumerate(texts):
        for la in layers_a:
            for lb in layers_b:
                if not args.full_heatmap and la != lb:
                    continue
                for kv in ["k", "v"]:
                    if la not in mats_a[s] or lb not in mats_b[s]:
                        continue
                    if la == lb:
                        row = compare_mats(mats_a[s][la][kv], mats_b[s][lb][kv], args.knn_k, args.n_shuffles)
                    else:
                        row = compare_basic(mats_a[s][la][kv], mats_b[s][lb][kv], args.knn_k)
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
                        row = compare_basic(mats_a[s][layer][kv], mats_b[t][layer][kv], args.knn_k)
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
                            row = compare_basic(mats[li][kv], mats[lj][kv], args.knn_k)
                            row.update({"sample": s, "model": model_name, "kv": kv, "layer_i": li, "layer_j": lj})
                            within_model.append(row)

    def mean(rows, key, kv=None):
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key]) and (kv is None or r.get("kv") == kv)]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "same_text_raw_rsa_k": mean(same_text, "raw_rsa", "k"),
        "same_text_raw_rsa_v": mean(same_text, "raw_rsa", "v"),
        "shuffle_rsa_k": mean(same_text, "shuffled_rsa_mean", "k"),
        "shuffle_rsa_v": mean(same_text, "shuffled_rsa_mean", "v"),
        "shuffle_rsa_std_k": mean(same_text, "shuffled_rsa_std", "k"),
        "shuffle_rsa_std_v": mean(same_text, "shuffled_rsa_std", "v"),
        "shuffle_rsa_p_ge_real_k": mean(same_text, "shuffled_rsa_p_ge_real", "k"),
        "shuffle_rsa_p_ge_real_v": mean(same_text, "shuffled_rsa_p_ge_real", "v"),
        "position_residual_rsa_k": mean(same_text, "position_residual_rsa", "k"),
        "position_residual_rsa_v": mean(same_text, "position_residual_rsa", "v"),
        "far_rsa_gap5_k": mean(same_text, "far_rsa_gap5", "k"),
        "far_rsa_gap5_v": mean(same_text, "far_rsa_gap5", "v"),
        "far_rsa_gap10_k": mean(same_text, "far_rsa_gap10", "k"),
        "far_rsa_gap10_v": mean(same_text, "far_rsa_gap10", "v"),
        "same_text_knn_k": mean(same_text, "knn_overlap", "k"),
        "same_text_knn_v": mean(same_text, "knn_overlap", "v"),
        "shuffle_knn_k": mean(same_text, "shuffled_knn_mean", "k"),
        "shuffle_knn_v": mean(same_text, "shuffled_knn_mean", "v"),
        "shuffle_knn_std_k": mean(same_text, "shuffled_knn_std", "k"),
        "shuffle_knn_std_v": mean(same_text, "shuffled_knn_std", "v"),
        "shuffle_knn_p_ge_real_k": mean(same_text, "shuffled_knn_p_ge_real", "k"),
        "shuffle_knn_p_ge_real_v": mean(same_text, "shuffled_knn_p_ge_real", "v"),
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
