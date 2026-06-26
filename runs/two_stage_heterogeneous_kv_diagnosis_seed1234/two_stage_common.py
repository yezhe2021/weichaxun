import hashlib
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
for path in (PROJECT_ROOT, REAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_kv_common import make_cache  # noqa: E402
from translated_kv_diagnostics import (  # noqa: E402
    answer_f1,
    answer_logits,
    cache_metrics,
    distribution_metrics,
    mean_metric,
)

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except Exception:  # pragma: no cover - only used on non-Qwen3 transformer builds.
    apply_rotary_pos_emb = None


STAGE_CONDITIONS = (
    "native",
    "translated",
    "native_k_translated_v",
    "translated_k_native_v",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def swap_cache(native_pairs, translated_pairs, condition):
    if condition == "native":
        return [(k, v) for k, v in native_pairs]
    if condition == "translated":
        return [(k, v) for k, v in translated_pairs]
    if condition == "native_k_translated_v":
        return [(nk, tv) for (nk, _), (_, tv) in zip(native_pairs, translated_pairs)]
    if condition == "translated_k_native_v":
        return [(tk, nv) for (_, nv), (tk, _) in zip(native_pairs, translated_pairs)]
    raise ValueError(f"Unknown condition: {condition}")


def assert_cache_shapes(native_pairs, translated_pairs):
    if len(native_pairs) != len(translated_pairs):
        raise ValueError(f"Layer mismatch: native={len(native_pairs)}, translated={len(translated_pairs)}")
    for layer, ((nk, nv), (tk, tv)) in enumerate(zip(native_pairs, translated_pairs)):
        if nk.shape != tk.shape or nv.shape != tv.shape:
            raise ValueError(
                f"Layer {layer} cache mismatch: "
                f"native K/V={tuple(nk.shape)}/{tuple(nv.shape)}, "
                f"translated K/V={tuple(tk.shape)}/{tuple(tv.shape)}"
            )


def cosine_mean(a, b):
    a = a.float()
    b = b.float()
    return F.cosine_similarity(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1).mean().item()


def teacher_forced_prediction(tokenizer, logits):
    ids = logits.argmax(dim=-1)[0]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def row_js_divergence(a, b):
    eps = 1e-12
    a = a.float().clamp_min(eps)
    b = b.float().clamp_min(eps)
    a = a / a.sum(dim=-1, keepdim=True)
    b = b / b.sum(dim=-1, keepdim=True)
    m = 0.5 * (a + b)
    return (
        0.5 * (a * (a.log() - m.log())).sum(-1)
        + 0.5 * (b * (b.log() - m.log())).sum(-1)
    ).mean().item()


def topk_overlap(a, b, k):
    k = min(k, a.shape[-1], b.shape[-1])
    if k <= 0:
        return float("nan")
    ai = a.topk(k, dim=-1).indices
    bi = b.topk(k, dim=-1).indices
    return (ai.unsqueeze(-1) == bi.unsqueeze(-2)).any(dim=-1).float().mean().item()


class TailCapture:
    def __init__(self, model, capture_q=False):
        self.attention_outputs = {}
        self.query_states = {}
        self.handles = []
        self.capture_q = capture_q
        for layer_idx, layer in enumerate(model.model.layers):
            if capture_q:
                self.handles.append(
                    layer.self_attn.register_forward_pre_hook(
                        self._q_hook(layer_idx), with_kwargs=True
                    )
                )
            self.handles.append(layer.self_attn.register_forward_hook(self._attention_hook(layer_idx)))

    def _attention_hook(self, layer_idx):
        def hook(module, args, output):
            attn_output = output[0] if isinstance(output, tuple) else output
            self.attention_outputs[layer_idx] = attn_output.detach().float().cpu()

        return hook

    def _q_hook(self, layer_idx):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states", None)
            if hidden_states is None:
                hidden_states = args[0]
            input_shape = hidden_states.shape[:-1]
            q = module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim)
            if hasattr(module, "q_norm"):
                q = module.q_norm(q)
            q = q.transpose(1, 2)
            position_embeddings = kwargs.get("position_embeddings", None)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                if apply_rotary_pos_emb is not None:
                    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
                else:
                    q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
            self.query_states[layer_idx] = q.detach().float().cpu()
            return None

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def capture_tail(model, capture_q=False):
    capture = TailCapture(model, capture_q=capture_q)
    try:
        yield capture
    finally:
        capture.close()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def run_tail(model, context_pairs, tail_ids, query_len, answer_len, capture_q=False):
    cache = make_cache(context_pairs, model.config)
    with capture_tail(model, capture_q=capture_q) as capture:
        output = model(input_ids=tail_ids, past_key_values=cache, use_cache=True)
    logits = answer_logits(output.logits, query_len, answer_len).detach().float().cpu()
    return {
        "logits": logits,
        "attention_outputs": capture.attention_outputs,
        "query_states": capture.query_states,
    }


def answer_position_slice(query_len, answer_len, seq_len):
    start = max(0, query_len - 1)
    end = min(seq_len, start + answer_len)
    return slice(start, end)


def attention_output_cos(native_run, condition_run, query_len, answer_len):
    values = []
    rows = []
    for layer in sorted(native_run["attention_outputs"]):
        native = native_run["attention_outputs"][layer]
        test = condition_run["attention_outputs"][layer]
        sl = answer_position_slice(query_len, answer_len, min(native.shape[1], test.shape[1]))
        if sl.stop <= sl.start:
            continue
        score = cosine_mean(native[:, sl], test[:, sl])
        rows.append({"layer": layer, "attention_output_cos": score})
        values.append(score)
    return (float(np.mean(values)) if values else float("nan")), rows


def evaluate_stage1_condition(condition, tokenizer, example, native_pairs, condition_pairs, native_run, condition_run):
    query_len = example["query_ids"].shape[1]
    answer_len = example["answer_ids"].shape[1]
    distribution = distribution_metrics(native_run["logits"], condition_run["logits"], example["answer_ids"].cpu())
    kv_rows = cache_metrics(
        [(k.detach().float().cpu(), v.detach().float().cpu()) for k, v in native_pairs],
        [(k.detach().float().cpu(), v.detach().float().cpu()) for k, v in condition_pairs],
    )
    attn_cos, attention_rows = attention_output_cos(native_run, condition_run, query_len, answer_len)
    prediction = teacher_forced_prediction(tokenizer, condition_run["logits"])
    result = {
        "condition": condition,
        "receiver_native_ce": distribution.pop("native_ce"),
        "condition_ce": distribution.pop("translated_ce"),
        **distribution,
        "answer_prediction_mode": "teacher_forced_token_argmax",
        "answer_prediction": prediction,
        "answer_f1": answer_f1(prediction, example["answer"]),
        "attention_output_cos": attn_cos,
        "kv_joint_consistency": mean_metric(kv_rows, "kv_joint_consistency"),
    }
    return result, kv_rows, attention_rows


def repeat_kv(x, repeats):
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


def offline_readout(query_states, pairs, num_attention_heads):
    outputs = {}
    routes = {}
    for layer, q in query_states.items():
        k, v = pairs[layer]
        k = k.detach().float().cpu()
        v = v.detach().float().cpu()
        repeats = num_attention_heads // k.shape[1]
        k = repeat_kv(k, repeats)
        v = repeat_kv(v, repeats)
        q = q.float()
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        routes[layer] = weights
        outputs[layer] = out
    return routes, outputs


def readout_probe_rows(query_states, native_pairs, translated_pairs, num_attention_heads, attention_topk):
    condition_pairs = {
        condition: swap_cache(native_pairs, translated_pairs, condition)
        for condition in STAGE_CONDITIONS
    }
    native_routes, native_outputs = offline_readout(query_states, condition_pairs["native"], num_attention_heads)
    rows = []
    for condition, pairs in condition_pairs.items():
        routes, outputs = offline_readout(query_states, pairs, num_attention_heads)
        for layer in sorted(native_routes):
            native_route = native_routes[layer]
            test_route = routes[layer]
            native_output = native_outputs[layer]
            test_output = outputs[layer]
            rows.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "route_overlap": topk_overlap(native_route, test_route, attention_topk),
                    "attention_js": row_js_divergence(native_route, test_route),
                    "attention_output_cos": cosine_mean(native_output, test_output),
                    "output_mse": F.mse_loss(test_output.float(), native_output.float()).item(),
                }
            )
    return rows


def summarize(rows, group_key, metric_keys):
    output = []
    for group in sorted({row[group_key] for row in rows}):
        selected = [row for row in rows if row[group_key] == group]
        item = {group_key: group, "n": len(selected)}
        for key in metric_keys:
            values = [row[key] for row in selected if key in row and np.isfinite(row[key])]
            if values:
                item[key] = float(np.mean(values))
        output.append(item)
    return output
