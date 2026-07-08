import json
import math
import re
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

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except Exception:  # pragma: no cover
    apply_rotary_pos_emb = None


def load_rows(path, limit):
    rows = []
    with open(path, encoding="utf-8") as handle:
        source = (json.loads(line) for line in handle if line.strip()) if str(path).endswith(".jsonl") else iter(json.load(handle))
        for row in source:
            if row.get("question") and row.get("answer") is not None:
                rows.append(row)
            if 0 < limit <= len(rows):
                break
    return rows


def flatten_context(context):
    if isinstance(context, str):
        return context
    return "\n".join(f"[{title}] {' '.join(sentences)}" for title, sentences in context)


def extract_gsm8k_final_answer(text):
    text = str(text)
    if "####" in text:
        text = text.split("####")[-1]
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else text.strip()


def build_paper_example(tokenizer, row, max_source_tokens, answer_mode="full", target_field="answer"):
    if row.get("source_text"):
        source_text = str(row["source_text"])
        task_type = row.get("task_type", "question_answer")
    elif row.get("context"):
        source_text = f"Context:\n{flatten_context(row['context'])}\n\nQuestion:\n{row['question']}"
        task_type = "context_question_answer"
    else:
        source_text = f"Question: {row['question']}\n"
        task_type = "question_answer"
    generation_prompt = str(row.get("generation_prompt", row.get("continuation_prompt", "Answer:")))
    if target_field != "answer" and row.get(target_field) is not None:
        answer_text = str(row[target_field]).strip()
        if not answer_text.startswith((" ", "\n", "\t")):
            answer_text = " " + answer_text
    elif answer_mode == "full":
        answer_text = " " + str(row["answer"]).strip()
    elif answer_mode == "final_only":
        answer_text = " #### " + extract_gsm8k_final_answer(row["answer"])
    else:
        raise ValueError(f"Unknown answer_mode: {answer_mode}")
    source_ids = tokenizer(
        source_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_source_tokens,
    ).input_ids
    aware_prompt_ids = tokenizer(generation_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    unaware_prompt_ids = aware_prompt_ids
    answer_ids = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False).input_ids
    if answer_ids.shape[1] == 0:
        raise ValueError("Answer tokenization produced no tokens")
    aware_prefix_ids = torch.cat([source_ids, aware_prompt_ids], dim=1)
    unaware_prefix_ids = unaware_prompt_ids
    return {
        "id": row.get("id"),
        "answer": str(row["answer"]),
        "target_field": target_field if row.get(target_field) is not None else "answer",
        "task_type": task_type,
        "source_text": source_text,
        "source_ids": source_ids,
        "answer_ids": answer_ids,
        "aware_prefix_ids": aware_prefix_ids,
        "unaware_prefix_ids": unaware_prefix_ids,
        "aware_tail_ids": torch.cat([aware_prefix_ids, answer_ids[:, :-1]], dim=1),
        "unaware_tail_ids": torch.cat([unaware_prefix_ids, answer_ids[:, :-1]], dim=1),
        "aware_prefix_len": aware_prefix_ids.shape[1],
        "unaware_prefix_len": unaware_prefix_ids.shape[1],
    }


def tokenizer_signature(tokenizer):
    return {
        "class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "len": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "special_tokens_map": tokenizer.special_tokens_map,
        "chat_template": getattr(tokenizer, "chat_template", None),
    }


def assert_tokenizer_compatible(sender_tokenizer, receiver_tokenizer, rows, max_source_tokens, max_checks=8):
    sender_sig = tokenizer_signature(sender_tokenizer)
    receiver_sig = tokenizer_signature(receiver_tokenizer)
    strict_keys = [
        "vocab_size",
        "len",
        "eos_token",
        "eos_token_id",
        "pad_token",
        "pad_token_id",
        "bos_token",
        "bos_token_id",
        "unk_token",
        "unk_token_id",
        "special_tokens_map",
        "chat_template",
    ]
    mismatches = [
        key for key in strict_keys
        if json.dumps(sender_sig.get(key), sort_keys=True, ensure_ascii=False)
        != json.dumps(receiver_sig.get(key), sort_keys=True, ensure_ascii=False)
    ]
    if mismatches:
        raise ValueError("Sender/receiver tokenizers are not strictly compatible: " + ", ".join(mismatches))
    for idx, row in enumerate(rows[:max_checks]):
        if row.get("context"):
            example_text = f"Context:\n{flatten_context(row['context'])}\n\nQuestion:\n{row['question']}"
        else:
            example_text = f"Question:\n{row['question']}"
        probes = {
            "source": (example_text, True, max_source_tokens),
            "aware": (example_text + "\n\nAnswer:", True, max_source_tokens + 32),
            "answer": (" " + str(row["answer"]).strip(), False, None),
        }
        for name, (text, add_special_tokens, max_len) in probes.items():
            kwargs = {"add_special_tokens": add_special_tokens}
            if max_len is not None:
                kwargs.update({"truncation": True, "max_length": max_len})
            sid = sender_tokenizer(text, **kwargs).input_ids
            rid = receiver_tokenizer(text, **kwargs).input_ids
            if sid != rid:
                raise ValueError(f"Tokenizer id mismatch on sample={idx} probe={name}")


def receiver_cache_reconstruction_loss(native_pairs, translated_pairs):
    losses = []
    for (nk, nv), (tk, tv) in zip(native_pairs, translated_pairs):
        losses.append(F.mse_loss(tk.float(), nk.float()))
        losses.append(F.mse_loss(tv.float(), nv.float()))
    return torch.stack(losses).mean()


def answer_logits_from_tail(logits, prefix_len, answer_len):
    start = prefix_len - 1
    return logits[:, start : start + answer_len]


def generation_ce(receiver, context_pairs, tail_ids, prefix_len, answer_ids):
    cache = make_cache(context_pairs, receiver.config)
    out = receiver(input_ids=tail_ids, past_key_values=cache, use_cache=True)
    logits = answer_logits_from_tail(out.logits, prefix_len, answer_ids.shape[1])
    n = min(logits.shape[1], answer_ids.shape[1])
    return F.cross_entropy(logits[:, :n].float().reshape(-1, logits.shape[-1]), answer_ids[:, :n].reshape(-1))


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class ReceiverTraceCapture:
    def __init__(self, receiver, keep_device=True):
        self.query_states = {}
        self.attention_outputs = {}
        self.handles = []
        self.keep_device = keep_device
        for layer_idx, layer in enumerate(receiver.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._q_hook(layer_idx), with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(self._out_hook(layer_idx)))

    def _q_hook(self, layer_idx):
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
def capture_receiver_trace(receiver, keep_device=True):
    capture = ReceiverTraceCapture(receiver, keep_device=keep_device)
    try:
        yield capture
    finally:
        capture.close()


def run_generation(receiver, context_pairs, tail_ids, prefix_len, answer_len, capture_trace=False):
    cache = make_cache(context_pairs, receiver.config)
    if capture_trace:
        with capture_receiver_trace(receiver, keep_device=True) as capture:
            out = receiver(input_ids=tail_ids, past_key_values=cache, use_cache=True)
        return answer_logits_from_tail(out.logits, prefix_len, answer_len), capture.query_states, capture.attention_outputs
    out = receiver(input_ids=tail_ids, past_key_values=cache, use_cache=True)
    return answer_logits_from_tail(out.logits, prefix_len, answer_len), {}, {}


def repeat_kv(x, repeats):
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


def answer_query_slice(prefix_len, answer_len, seq_len):
    start = max(0, prefix_len - 1)
    end = min(seq_len, start + answer_len)
    return slice(start, end)


def offline_readout(query_states, pairs, num_attention_heads, prefix_len, answer_len):
    routes = {}
    outputs = {}
    for layer, q in query_states.items():
        sl = answer_query_slice(prefix_len, answer_len, q.shape[-2])
        q = q[..., sl, :].float()
        k, v = pairs[layer]
        k = k.float()
        v = v.float()
        repeats = num_attention_heads // k.shape[1]
        k = repeat_kv(k, repeats)
        v = repeat_kv(v, repeats)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        routes[layer] = weights
        outputs[layer] = torch.matmul(weights, v)
    return routes, outputs


def logit_kl_loss(native_logits, translated_logits):
    n = min(native_logits.shape[1], translated_logits.shape[1])
    teacher = F.softmax(native_logits[:, :n].detach().float(), dim=-1)
    student_log = F.log_softmax(translated_logits[:, :n].float(), dim=-1)
    return F.kl_div(student_log, teacher, reduction="batchmean") / n


def route_js_loss(native_routes, translated_routes):
    losses = []
    for layer in native_routes:
        teacher = native_routes[layer].detach().float().clamp_min(1e-12)
        student = translated_routes[layer].float().clamp_min(1e-12)
        midpoint = 0.5 * (teacher + student)
        js = 0.5 * (teacher * (teacher.log() - midpoint.log())).sum(dim=-1)
        js = js + 0.5 * (student * (student.log() - midpoint.log())).sum(dim=-1)
        losses.append(js.mean())
    return torch.stack(losses).mean()


def readout_alignment_loss(native_outputs, translated_outputs):
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
    return mse + cos, mse, cos


def q_aware_readout_losses(receiver, query_states, native_pairs, translated_pairs, prefix_len, answer_len):
    native_routes, native_outputs = offline_readout(
        query_states,
        native_pairs,
        receiver.config.num_attention_heads,
        prefix_len,
        answer_len,
    )
    translated_routes, translated_outputs = offline_readout(
        query_states,
        translated_pairs,
        receiver.config.num_attention_heads,
        prefix_len,
        answer_len,
    )
    route = route_js_loss(native_routes, translated_routes)
    readout, readout_mse, readout_cos_loss = readout_alignment_loss(native_outputs, translated_outputs)
    return {
        "route_loss": route,
        "readout_loss": readout,
        "readout_mse": readout_mse,
        "readout_cos_loss": readout_cos_loss,
    }


def q_aware_functional_loss(receiver, native_pairs, translated_pairs, example, device, aware_weight, weights):
    answer_ids = example["answer_ids"].to(device)
    answer_len = answer_ids.shape[1]
    modes = (
        ("context_aware", example["aware_tail_ids"].to(device), example["aware_prefix_len"], aware_weight),
        ("context_unaware", example["unaware_tail_ids"].to(device), example["unaware_prefix_len"], 1.0 - aware_weight),
    )
    ce_terms = []
    kl_terms = []
    route_terms = []
    readout_terms = []
    readout_mse_terms = []
    readout_cos_terms = []
    metrics = {}
    for mode, tail_ids, prefix_len, mode_weight in modes:
        if mode_weight == 0:
            continue
        with torch.no_grad():
            native_logits, native_q, _ = run_generation(
                receiver,
                native_pairs,
                tail_ids,
                prefix_len,
                answer_len,
                capture_trace=True,
            )
        translated_logits, _, _ = run_generation(
            receiver,
            translated_pairs,
            tail_ids,
            prefix_len,
            answer_len,
            capture_trace=False,
        )
        n = min(translated_logits.shape[1], answer_ids.shape[1])
        ce = F.cross_entropy(
            translated_logits[:, :n].float().reshape(-1, translated_logits.shape[-1]),
            answer_ids[:, :n].reshape(-1),
        )
        kl = logit_kl_loss(native_logits, translated_logits)
        readout = q_aware_readout_losses(receiver, native_q, native_pairs, translated_pairs, prefix_len, answer_len)
        ce_terms.append(mode_weight * ce)
        kl_terms.append(mode_weight * kl)
        route_terms.append(mode_weight * readout["route_loss"])
        readout_terms.append(mode_weight * readout["readout_loss"])
        readout_mse_terms.append(mode_weight * readout["readout_mse"])
        readout_cos_terms.append(mode_weight * readout["readout_cos_loss"])
        metrics[f"{mode}_ce"] = ce
        metrics[f"{mode}_logit_kl"] = kl
        metrics[f"{mode}_route_loss"] = readout["route_loss"]
        metrics[f"{mode}_readout_loss"] = readout["readout_loss"]
    ce_loss = torch.stack(ce_terms).sum()
    logit_kl = torch.stack(kl_terms).sum()
    route_loss = torch.stack(route_terms).sum()
    readout_loss = torch.stack(readout_terms).sum()
    readout_mse = torch.stack(readout_mse_terms).sum()
    readout_cos_loss = torch.stack(readout_cos_terms).sum()
    rec_loss = receiver_cache_reconstruction_loss(native_pairs, translated_pairs)
    total = (
        weights["ce"] * ce_loss
        + weights["logit_kl"] * logit_kl
        + weights["route"] * route_loss
        + weights["readout"] * readout_loss
        + weights["weak_rec"] * rec_loss
    )
    metrics.update(
        {
            "loss": total,
            "generation_loss": ce_loss,
            "logit_kl_loss": logit_kl,
            "route_loss": route_loss,
            "readout_loss": readout_loss,
            "readout_mse": readout_mse,
            "readout_cos_loss": readout_cos_loss,
            "receiver_cache_reconstruction_loss": rec_loss,
        }
    )
    return metrics


def normalize_answer(text):
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(prediction, gold):
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(token), g.count(token)) for token in set(p) & set(g))
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
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


def cosine_mean(a, b):
    return F.cosine_similarity(
        a.float().reshape(-1, a.shape[-1]),
        b.float().reshape(-1, b.shape[-1]),
        dim=-1,
    ).mean().item()


def topk_overlap(a, b, k):
    k = min(k, a.shape[-1], b.shape[-1])
    if k <= 0:
        return float("nan")
    ai = a.topk(k, dim=-1).indices
    bi = b.topk(k, dim=-1).indices
    return (ai.unsqueeze(-1) == bi.unsqueeze(-2)).any(dim=-1).float().mean().item()


def js_divergence(a, b):
    eps = 1e-12
    a = a.float().clamp_min(eps)
    b = b.float().clamp_min(eps)
    a = a / a.sum(dim=-1, keepdim=True)
    b = b / b.sum(dim=-1, keepdim=True)
    m = 0.5 * (a + b)
    return (0.5 * (a * (a.log() - m.log())).sum(-1) + 0.5 * (b * (b.log() - m.log())).sum(-1)).mean().item()


def distribution_metrics(native_logits, translated_logits, answer_ids):
    n = min(native_logits.shape[1], translated_logits.shape[1], answer_ids.shape[1])
    native_logits = native_logits[:, :n].float()
    translated_logits = translated_logits[:, :n].float()
    targets = answer_ids[:, :n]
    native_ce = F.cross_entropy(native_logits.reshape(-1, native_logits.shape[-1]), targets.reshape(-1))
    translated_ce = F.cross_entropy(translated_logits.reshape(-1, translated_logits.shape[-1]), targets.reshape(-1))
    return {
        "receiver_native_ce": native_ce.item(),
        "translated_ce": translated_ce.item(),
        "ce_delta": (translated_ce - native_ce).item(),
        "logit_kl": max(0.0, F.kl_div(F.log_softmax(translated_logits, -1), F.softmax(native_logits, -1), reduction="batchmean").item() / n),
        "top1_match": (native_logits.argmax(-1) == translated_logits.argmax(-1)).float().mean().item(),
    }


def cache_metric_rows(native_pairs, translated_pairs):
    rows = []
    for layer, ((nk, nv), (tk, tv)) in enumerate(zip(native_pairs, translated_pairs)):
        rows.append(
            {
                "layer": layer,
                "kv_mse": 0.5 * (F.mse_loss(tk.float(), nk.float()).item() + F.mse_loss(tv.float(), nv.float()).item()),
                "k_cos": cosine_mean(nk, tk),
                "v_cos": cosine_mean(nv, tv),
                "kv_joint_consistency": cosine_mean(torch.matmul(nk.float(), nv.float().transpose(-1, -2)), torch.matmul(tk.float(), tv.float().transpose(-1, -2))),
            }
        )
    return rows


def mean_metric(rows, key):
    values = [row[key] for row in rows if key in row and row[key] == row[key]]
    return float(sum(values) / len(values)) if values else float("nan")
