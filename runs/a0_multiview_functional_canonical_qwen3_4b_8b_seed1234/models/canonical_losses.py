import math

import torch
import torch.nn.functional as F


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x, positions, theta):
    inverse = 1.0 / (
        float(theta)
        ** (
            torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32)
            / x.shape[-1]
        )
    )
    frequency = torch.outer(
        torch.tensor(positions, device=x.device, dtype=torch.float32), inverse
    )
    embedding = torch.cat((frequency, frequency), dim=-1)
    cosine = embedding.cos().to(x.dtype)[None, :, None, :]
    sine = embedding.sin().to(x.dtype)[None, :, None, :]
    return x * cosine + rotate_half(x) * sine


def layerwise_terms(prediction, target):
    pred, gold = prediction.float(), target.float()
    nmse = (pred - gold).square().mean((1, 2, 3)) / (
        gold.square().mean((1, 2, 3)) + 1e-6
    )
    cosine = 1 - F.cosine_similarity(
        pred.reshape(pred.shape[0], -1, pred.shape[-1]),
        gold.reshape(gold.shape[0], -1, gold.shape[-1]),
        dim=-1,
    ).mean(1)
    return nmse.mean(), cosine.mean()


def canonical_alignment(c4k, c4v, c8k, c8v):
    k_nmse, k_cos = layerwise_terms(c4k, c8k)
    v_nmse, v_cos = layerwise_terms(c4v, c8v)
    return k_nmse + v_nmse + 0.5 * (k_cos + v_cos)


def route_and_output(cfg, query, pred_k, pred_v, target_k, target_v, epos, qpos):
    groups = cfg["num_query_heads"] // cfg["num_kv_heads"]
    query = apply_rope(query.to(pred_k.device), qpos, cfg["rope_theta"])
    pred_k = apply_rope(pred_k, epos, cfg["rope_theta"]).repeat_interleave(groups, 2)
    target_k = apply_rope(target_k, epos, cfg["rope_theta"]).repeat_interleave(groups, 2)
    pred_v = pred_v.repeat_interleave(groups, 2)
    target_v = target_v.repeat_interleave(groups, 2)
    scale = math.sqrt(cfg["head_dim"])
    native_logits = torch.einsum("lqhd,lthd->lhqt", query, target_k) / scale
    decoded_logits = torch.einsum("lqhd,lthd->lhqt", query, pred_k) / scale
    native_attention = native_logits.float().softmax(-1)
    decoded_log = decoded_logits.float().log_softmax(-1)
    route = (
        native_attention
        * (native_attention.clamp_min(1e-9).log() - decoded_log)
    ).sum(-1).mean()
    native_out = torch.einsum(
        "lhqt,lthd->lqhd", native_attention.to(target_v.dtype), target_v
    )
    decoded_attention = decoded_logits.softmax(-1)
    decoded_out = torch.einsum("lhqt,lthd->lqhd", decoded_attention, pred_v)
    out_nmse, out_cos = layerwise_terms(decoded_out, native_out)
    return route, out_nmse + 0.5 * out_cos, 1 - out_cos


def path_loss(cfg, query, pred_k, pred_v, target_k, target_v, epos, qpos):
    k_nmse, k_cos = layerwise_terms(pred_k, target_k)
    v_nmse, v_cos = layerwise_terms(pred_v, target_v)
    route, output, output_cosine = route_and_output(
        cfg, query, pred_k, pred_v, target_k, target_v, epos, qpos
    )
    decode = k_nmse + v_nmse + 0.5 * (k_cos + v_cos)
    return {
        "decode": decode,
        "route": route,
        "output": output,
        "output_cosine": output_cosine,
        "total": decode + 0.5 * route + output,
    }


def pooled(key, value):
    return torch.cat((key.float().mean((0, 1, 2)), value.float().mean((0, 1, 2))))


def variance_loss(vectors, target):
    features = torch.stack(vectors)
    return F.relu(float(target) - features.std(0, unbiased=False)).mean()


def contrastive(z4, z8, temperature):
    if len(z4) < 2:
        return z4[0].new_zeros(())
    left = F.normalize(torch.stack(z4), dim=-1)
    right = F.normalize(torch.stack(z8), dim=-1)
    labels = torch.arange(left.shape[0], device=left.device)
    logits = left @ right.T / float(temperature)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
