import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
for path in (PROJECT_ROOT, REAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_kv_common import fixed_index_map, head_dim_from_config  # noqa: E402


def dtype_from_name(name):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def require_int4_backend():
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "INT4 requested, but bitsandbytes is not installed in attnkv. "
            "Install one pinned bitsandbytes version before running. "
            "The experiment never falls back to FP16."
        )


def load_quantized_model(path, precision, dtype, device, eager=False):
    kwargs = {"dtype": dtype, "trust_remote_code": True}
    if eager:
        kwargs["attn_implementation"] = "eager"
    if precision == "int4":
        if device.type != "cuda":
            raise ValueError("bitsandbytes INT4 requires --device cuda")
        require_int4_backend()
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()
    audit = quantization_audit(model, precision, path, dtype)
    return model, audit


def quantization_audit(model, requested_precision, path, compute_dtype):
    params4 = sum(1 for parameter in model.parameters() if "4bit" in parameter.__class__.__name__.lower())
    loaded4 = bool(getattr(model, "is_loaded_in_4bit", False))
    if requested_precision == "int4" and not (loaded4 and params4 > 0):
        raise RuntimeError("INT4 load silently failed: no 4-bit parameters were found")
    if requested_precision == "fp16" and (loaded4 or params4 > 0):
        raise RuntimeError("FP16 control unexpectedly contains 4-bit parameters")
    quant_config = getattr(model.config, "quantization_config", None)
    if hasattr(quant_config, "to_dict"):
        quant_config = quant_config.to_dict()
    return {
        "model_path": str(path),
        "requested_precision": requested_precision,
        "compute_dtype": str(compute_dtype),
        "is_loaded_in_4bit": loaded4,
        "num_4bit_parameter_tensors": params4,
        "quantization_config": quant_config,
    }


def structure_signature(config):
    return {
        "layers": int(config.num_hidden_layers),
        "kv_heads": int(config.num_key_value_heads),
        "attention_heads": int(config.num_attention_heads),
        "head_dim": head_dim_from_config(config),
        "rope_theta": float(getattr(config, "rope_theta", 10_000.0)),
        "vocab_size": int(config.vocab_size),
    }


def alignment_plan(sender_config, receiver_config):
    sender = structure_signature(sender_config)
    receiver = structure_signature(receiver_config)
    direct = all(sender[key] == receiver[key] for key in ("layers", "kv_heads", "head_dim"))
    return {
        "mode": "direct_layer_head" if direct else "fixed_index_relation_only",
        "sender": sender,
        "receiver": receiver,
        "layer_map": list(range(receiver["layers"]))
        if direct
        else fixed_index_map(sender["layers"], receiver["layers"]),
        "head_map": list(range(receiver["kv_heads"]))
        if direct
        else fixed_index_map(sender["kv_heads"], receiver["kv_heads"]),
        "direct_tensor_metrics": direct,
    }


def _flatten_relation(x):
    if x.shape[-1] < 2:
        return x.flatten()
    indices = torch.triu_indices(x.shape[-2], x.shape[-1], offset=1, device=x.device)
    return x[..., indices[0], indices[1]].flatten()


def _pearson(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    n = min(a.numel(), b.numel())
    if n < 2:
        return float("nan")
    a, b = a[:n], b[:n]
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return (a @ b / denom.clamp_min(1e-12)).item()


def _rank(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), device=x.device, dtype=torch.float32)
    return ranks


def token_gram(x):
    x = F.normalize(x[0].float(), dim=-1)
    return torch.matmul(x, x.transpose(-1, -2)).mean(dim=0)


def relation_metrics(sender_tensor, receiver_tensor, knn_k):
    sender_gram = token_gram(sender_tensor)
    receiver_gram = token_gram(receiver_tensor)
    token_count = min(sender_gram.shape[-1], receiver_gram.shape[-1])
    sender_gram = sender_gram[:token_count, :token_count]
    receiver_gram = receiver_gram[:token_count, :token_count]
    sender_flat = _flatten_relation(sender_gram)
    receiver_flat = _flatten_relation(receiver_gram)
    k = min(knn_k, max(0, token_count - 1))
    if k:
        eye = torch.eye(token_count, dtype=torch.bool, device=sender_gram.device)
        sender_neighbors = sender_gram.masked_fill(eye, -math.inf).topk(k, dim=-1).indices
        receiver_neighbors = receiver_gram.masked_fill(eye, -math.inf).topk(k, dim=-1).indices
        overlap = (
            (sender_neighbors.unsqueeze(-1) == receiver_neighbors.unsqueeze(-2))
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )
    else:
        overlap = float("nan")
    return {
        "gram_correlation": _pearson(sender_flat, receiver_flat),
        "rsa": _pearson(_rank(sender_flat), _rank(receiver_flat)),
        "knn_overlap": overlap,
    }


def direct_cosine(sender_tensor, receiver_tensor):
    if sender_tensor.shape != receiver_tensor.shape:
        return float("nan")
    return F.cosine_similarity(sender_tensor.float(), receiver_tensor.float(), dim=-1).mean().item()


def geometry_rows(sender_pairs, receiver_pairs, plan, knn_k):
    rows = []
    for receiver_layer, sender_layer in enumerate(plan["layer_map"]):
        sender_k, sender_v = sender_pairs[sender_layer]
        receiver_k, receiver_v = receiver_pairs[receiver_layer]
        if not plan["direct_tensor_metrics"]:
            sender_k = sender_k[:, plan["head_map"]]
            sender_v = sender_v[:, plan["head_map"]]
        k_relation = relation_metrics(sender_k, receiver_k, knn_k)
        v_relation = relation_metrics(sender_v, receiver_v, knn_k)
        sender_joint = _pearson(_flatten_relation(token_gram(sender_k)), _flatten_relation(token_gram(sender_v)))
        receiver_joint = _pearson(
            _flatten_relation(token_gram(receiver_k)), _flatten_relation(token_gram(receiver_v))
        )
        rows.append(
            {
                "receiver_layer": receiver_layer,
                "sender_layer": sender_layer,
                "k_cos": direct_cosine(sender_k, receiver_k),
                "v_cos": direct_cosine(sender_v, receiver_v),
                "k_gram_correlation": k_relation["gram_correlation"],
                "v_gram_correlation": v_relation["gram_correlation"],
                "k_rsa": k_relation["rsa"],
                "v_rsa": v_relation["rsa"],
                "k_knn_overlap": k_relation["knn_overlap"],
                "v_knn_overlap": v_relation["knn_overlap"],
                "sender_kv_joint_corr": sender_joint,
                "receiver_kv_joint_corr": receiver_joint,
                "kv_joint_consistency": 1.0 - min(2.0, abs(sender_joint - receiver_joint)) / 2.0,
            }
        )
    return rows


def attention_route_rows(sender_attentions, receiver_attentions, layer_map, topk):
    rows = []
    for receiver_layer, sender_layer in enumerate(layer_map):
        sender = sender_attentions[sender_layer][0].float().mean(dim=0)
        receiver = receiver_attentions[receiver_layer][0].float().mean(dim=0)
        query_count = min(sender.shape[-2], receiver.shape[-2])
        key_count = min(sender.shape[-1], receiver.shape[-1])
        sender = sender[-query_count:, :key_count]
        receiver = receiver[-query_count:, :key_count]
        k = min(topk, key_count)
        sender_top = sender.topk(k, dim=-1).indices
        receiver_top = receiver.topk(k, dim=-1).indices
        overlap = (
            (sender_top.unsqueeze(-1) == receiver_top.unsqueeze(-2)).any(dim=-1).float().mean().item()
        )
        rows.append(
            {
                "receiver_layer": receiver_layer,
                "sender_layer": sender_layer,
                "self_attention_route_overlap": overlap,
                "self_attention_route_correlation": _pearson(sender, receiver),
            }
        )
    return rows


def mean_finite(rows, key):
    values = [row[key] for row in rows if key in row and np.isfinite(row[key])]
    return float(np.mean(values)) if values else float("nan")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

