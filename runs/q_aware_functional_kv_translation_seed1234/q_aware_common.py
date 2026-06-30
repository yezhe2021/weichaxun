import math
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
for path in (PROJECT_ROOT, REAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_kv_common import make_cache  # noqa: E402
from translated_kv_diagnostics import answer_logits  # noqa: E402

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except Exception:  # pragma: no cover
    apply_rotary_pos_emb = None


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class ReceiverQCapture:
    def __init__(self, receiver, keep_device=True):
        self.query_states = {}
        self.attention_outputs = {}
        self.keep_device = keep_device
        self.handles = []
        for layer_idx, layer in enumerate(receiver.model.layers):
            self.handles.append(
                layer.self_attn.register_forward_pre_hook(
                    self._pre_hook(layer_idx), with_kwargs=True
                )
            )
            self.handles.append(layer.self_attn.register_forward_hook(self._out_hook(layer_idx)))

    def _pre_hook(self, layer_idx):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                hidden_states = args[0]
            input_shape = hidden_states.shape[:-1]
            q = module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim)
            if hasattr(module, "q_norm"):
                q = module.q_norm(q)
            q = q.transpose(1, 2)
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is not None:
                cos, sin = position_embeddings
                if apply_rotary_pos_emb is not None:
                    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
                else:
                    q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
            q = q.detach()
            self.query_states[layer_idx] = q if self.keep_device else q.float().cpu()
            return None

        return hook

    def _out_hook(self, layer_idx):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out.detach()
            self.attention_outputs[layer_idx] = out if self.keep_device else out.float().cpu()

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def capture_receiver_q(receiver, keep_device=True):
    capture = ReceiverQCapture(receiver, keep_device=keep_device)
    try:
        yield capture
    finally:
        capture.close()


def answer_position_slice(query_len, answer_len, seq_len):
    start = max(0, query_len - 1)
    end = min(seq_len, start + answer_len)
    return slice(start, end)


def repeat_kv(x, repeats):
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


def offline_readout(query_states, pairs, num_attention_heads, query_len, answer_len):
    routes = {}
    outputs = {}
    for layer, q in query_states.items():
        q = q.float()
        sl = answer_position_slice(query_len, answer_len, q.shape[-2])
        q = q[..., sl, :]
        k, v = pairs[layer]
        k = k.float()
        v = v.float()
        repeats = num_attention_heads // k.shape[1]
        k = repeat_kv(k, repeats)
        v = repeat_kv(v, repeats)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        routes[layer] = weights
        outputs[layer] = out
    return routes, outputs


def kl_route_loss(native_routes, translated_routes):
    losses = []
    for layer in native_routes:
        teacher = native_routes[layer].detach().float().clamp_min(1e-12)
        student = translated_routes[layer].float().clamp_min(1e-12)
        losses.append(F.kl_div(student.log(), teacher, reduction="batchmean") / teacher.shape[-2])
    return torch.stack(losses).mean()


def js_route_loss(native_routes, translated_routes):
    losses = []
    for layer in native_routes:
        teacher = native_routes[layer].detach().float().clamp_min(1e-12)
        student = translated_routes[layer].float().clamp_min(1e-12)
        midpoint = 0.5 * (teacher + student)
        js = 0.5 * (teacher * (teacher.log() - midpoint.log())).sum(dim=-1)
        js = js + 0.5 * (student * (student.log() - midpoint.log())).sum(dim=-1)
        losses.append(js.mean())
    return torch.stack(losses).mean()


def readout_alignment_loss(native_outputs, translated_outputs, cosine_weight=1.0, mse_weight=1.0):
    mse_losses = []
    cos_losses = []
    for layer in native_outputs:
        native = native_outputs[layer].detach().float()
        translated = translated_outputs[layer].float()
        mse_losses.append(F.mse_loss(translated, native))
        cos = F.cosine_similarity(
            translated.reshape(-1, translated.shape[-1]),
            native.reshape(-1, native.shape[-1]),
            dim=-1,
        ).mean()
        cos_losses.append(1.0 - cos)
    mse = torch.stack(mse_losses).mean()
    cos = torch.stack(cos_losses).mean()
    return mse_weight * mse + cosine_weight * cos, mse, cos


def q_aware_losses(receiver, query_states, native_pairs, translated_pairs, query_len, answer_len, route_kind):
    native_routes, native_outputs = offline_readout(
        query_states,
        native_pairs,
        receiver.config.num_attention_heads,
        query_len,
        answer_len,
    )
    translated_routes, translated_outputs = offline_readout(
        query_states,
        translated_pairs,
        receiver.config.num_attention_heads,
        query_len,
        answer_len,
    )
    if route_kind == "js":
        route = js_route_loss(native_routes, translated_routes)
    elif route_kind == "kl":
        route = kl_route_loss(native_routes, translated_routes)
    else:
        raise ValueError(f"Unknown route loss: {route_kind}")
    readout, readout_mse, readout_cos = readout_alignment_loss(native_outputs, translated_outputs)
    return {
        "route_loss": route,
        "readout_loss": readout,
        "readout_mse": readout_mse,
        "readout_cos_loss": readout_cos,
    }


def tail_logits(receiver, context_pairs, tail_ids, query_len, answer_len, capture_q=False):
    cache = make_cache(context_pairs, receiver.config)
    if capture_q:
        with capture_receiver_q(receiver, keep_device=True) as capture:
            out = receiver(input_ids=tail_ids, past_key_values=cache, use_cache=True)
        return answer_logits(out.logits, query_len, answer_len), capture.query_states, capture.attention_outputs
    out = receiver(input_ids=tail_ids, past_key_values=cache, use_cache=True)
    return answer_logits(out.logits, query_len, answer_len), None, None


def logit_kl_loss(native_logits, translated_logits):
    n = min(native_logits.shape[1], translated_logits.shape[1])
    teacher = F.softmax(native_logits[:, :n].detach().float(), dim=-1)
    student_log = F.log_softmax(translated_logits[:, :n].float(), dim=-1)
    return F.kl_div(student_log, teacher, reduction="batchmean") / n


def topk_overlap(a, b, k):
    k = min(k, a.shape[-1], b.shape[-1])
    if k <= 0:
        return float("nan")
    ai = a.topk(k, dim=-1).indices
    bi = b.topk(k, dim=-1).indices
    return (ai.unsqueeze(-1) == bi.unsqueeze(-2)).any(dim=-1).float().mean().item()


def route_js_value(a, b):
    eps = 1e-12
    a = a.float().clamp_min(eps)
    b = b.float().clamp_min(eps)
    a = a / a.sum(dim=-1, keepdim=True)
    b = b / b.sum(dim=-1, keepdim=True)
    midpoint = 0.5 * (a + b)
    return (
        0.5 * (a * (a.log() - midpoint.log())).sum(-1)
        + 0.5 * (b * (b.log() - midpoint.log())).sum(-1)
    ).mean().item()


def cosine_mean(a, b):
    return F.cosine_similarity(
        a.float().reshape(-1, a.shape[-1]),
        b.float().reshape(-1, b.shape[-1]),
        dim=-1,
    ).mean().item()
