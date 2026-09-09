from __future__ import annotations

import json
import random
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


SYSTEM = (
    "You are a question answering system. Answer the question using only the provided context. "
    "Return only the shortest answer span. Do not explain your reasoning. "
    'If the answer is yes or no, return exactly "yes" or "no".'
)


# ---------------------------------------------------------------------------
# IO / env
# ---------------------------------------------------------------------------

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError("Fast tokenizer with offset mappings is required")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg):
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], local_files_only=True, dtype=dtype,
        low_cpu_mem_usage=True, device_map={"": 0},
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Prompt rendering / dataset (identical split logic to prior full-context audit)
# ---------------------------------------------------------------------------

def paragraph_text(sentences):
    return "".join(sentences)


def user_text(raw, include_context=True):
    blocks = []
    if include_context:
        blocks.append("Context:")
        for index, (title, sentences) in enumerate(raw["context"], 1):
            blocks.append(f"[{index}] {title}\n{paragraph_text(sentences)}")
    blocks.append(f"Question: {raw['question']}\n\nAnswer:")
    return "\n\n".join(blocks)


def render(tok, raw, include_context=True):
    """Single tokenization of the full chat-template string, then locate the
    ``Question:`` token boundary via offset mapping. Guarantees
    full == prefix + suffix by construction."""
    user = user_text(raw, include_context)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded.input_ids)
    template_ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    template_ids = template_ids.input_ids if hasattr(template_ids, "input_ids") else template_ids
    if ids != list(template_ids):
        raise RuntimeError("rendered-string tokenization differs from apply_chat_template")
    result = {"rendered_prompt": rendered, "input_ids": ids}
    if include_context:
        marker = f"Question: {raw['question']}\n\nAnswer:"
        marker_start = rendered.rfind(marker)
        if marker_start < 0:
            raise RuntimeError("Question boundary not found in rendered prompt")
        question_char = marker_start + len("Question: ")
        context_end = next(
            (i for i, (_, end) in enumerate(encoded.offset_mapping) if end > question_char),
            len(ids),
        )
        if context_end <= 0 or context_end >= len(ids):
            raise RuntimeError("invalid context/question token boundary")
        result.update({
            "context_end_index": context_end,
            "context_input_ids": ids[:context_end],
            "question_input_ids": ids[context_end:],
            "question_char_index": question_char,
            "boundary_token_offset": list(encoded.offset_mapping[context_end]),
            "split_equal": ids[:context_end] + ids[context_end:] == ids,
        })
    return result


def fixed_samples(cfg, tok):
    raw = load_json(cfg["hotpot_dev"])
    rng = random.Random(cfg["seed"])
    groups = {"bridge": [], "comparison": []}
    for row in raw:
        kind = row.get("type", "unknown")
        if kind in groups:
            groups[kind].append(row)
    for values in groups.values():
        rng.shuffle(values)
    half = cfg["formal_samples"] // 2
    chosen = groups["bridge"][:half] + groups["comparison"][:cfg["formal_samples"] - half]
    rng.shuffle(chosen)
    output = []
    for row in chosen:
        full = render(tok, row, True)
        question_only = render(tok, row, False)
        output.append({
            "id": row["_id"], "type": row.get("type", "unknown"),
            "answer": row["answer"], "question": row["question"],
            "titles": [title for title, _ in row["context"]],
            **full, "question_only_input_ids": question_only["input_ids"],
            "question_only_rendered_prompt": question_only["rendered_prompt"],
        })
    return output


def selected_samples(cfg, mode):
    rows = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "rendered_samples.jsonl")
    count = cfg["smoke_samples"] if mode == "smoke" else cfg["formal_samples"]
    if len(rows) < count:
        raise RuntimeError(f"rendered manifest has {len(rows)} rows, need {count}")
    return rows[:count]


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def normalize_answer(text):
    def remove_articles(value): return re.sub(r"\b(a|an|the)\b", " ", value)
    def white_space_fix(value): return " ".join(value.split())
    def remove_punc(value): return "".join(ch for ch in value if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(text.lower())))


def answer_f1(prediction, gold):
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def target_ids(tok, answer, maximum):
    ids = tok(answer, add_special_tokens=False).input_ids[:maximum]
    if not ids:
        ids = [tok.eos_token_id]
    return ids + [tok.eos_token_id]


def nll(logits, gold):
    return F.cross_entropy(logits, torch.tensor(gold, dtype=torch.long)).item()


def distribution(reference, other, gold=None):
    """reference is the ground-truth distribution (float logits), other the candidate."""
    ref_logp = reference.log_softmax(-1)
    other_logp = other.log_softmax(-1)
    kl = (ref_logp.exp() * (ref_logp - other_logp)).sum(-1)
    metrics = {
        "mean_kl": kl.mean().item(),
        "max_kl": kl.max().item(),
        "top1_match_rate": (reference.argmax(-1) == other.argmax(-1)).float().mean().item(),
        "logits_max_absolute_error": (reference - other).abs().max().item(),
        "logits_mean_absolute_error": (reference - other).abs().mean().item(),
        "logits_cosine": F.cosine_similarity(reference.flatten(), other.flatten(), 0).item(),
    }
    if gold is not None:
        metrics["reference_nll"] = nll(reference, gold)
        metrics["other_nll"] = nll(other, gold)
        metrics["nll_absolute_difference"] = abs(metrics["reference_nll"] - metrics["other_nll"])
    return metrics


# ---------------------------------------------------------------------------
# Qwen3.5 hybrid cache helpers
#
# NOTE: transformers versions differ in how linear-attention layer caches store
# state. Installed transformers 5.9.0 stores a single tensor per layer
# (``conv_states``/``recurrent_states``) plus bool init flags and a stored
# ``has_previous_state`` bool; newer main-branch versions store dicts keyed by
# ``state_idx``. Every helper below is written to handle BOTH shapes so the
# script runs on either. ``clone_cache`` copies *all* instance attributes via
# ``vars(src)``, so nothing (tensors, dicts of tensors, bools, and crucially
# ``has_previous_state``) is ever dropped.
# ---------------------------------------------------------------------------

def layer_types(config):
    """Return the per-layer attention-type list (str) for a Qwen3.5 config."""
    lt = getattr(config, "layer_types", None)
    if lt is None:
        tc = getattr(config, "text_config", None)
        lt = getattr(tc, "layer_types", None) if tc is not None else None
    if lt is None:
        n = getattr(config, "num_hidden_layers", 32)
        lt = ["full_attention"] * n
    return list(lt)


def _clone_value(value):
    """Recursively clone a cache-layer attribute (tensor / dict-of-tensors /
    scalar / dtype / device). Non-tensor objects (dtype, device, module refs)
    are shared by reference; they are immutable or stateless in the cache path."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    if isinstance(value, (bool, int, float, str)):
        return value
    return value


def _copy_layer_state(src, dst):
    for attr, value in vars(src).items():
        setattr(dst, attr, _clone_value(value))


def clone_cache(cache, config):
    """Explicit deep clone of a Qwen3.5 hybrid DynamicCache.

    Rebuilds the per-layer cache objects from ``config`` (so FA layers get a
    fresh ``DynamicLayer`` and linear layers a fresh ``LinearAttentionLayer``),
    then copies every instance attribute of the source layer verbatim. The
    source cache is never mutated. Copying via ``vars(src)`` guarantees the
    ``has_previous_state`` flag travels with the state, which the model reads
    to decide whether to resume from the copied recurrent/conv states.
    """
    new = DynamicCache(config=config)
    for src, dst in zip(cache.layers, new.layers):
        _copy_layer_state(src, dst)
    return new


def _state_iter(states):
    """Yield ``(state_idx, tensor)`` from a conv/recurrent container that is
    either a single tensor (transformers 5.9) or a dict keyed by state_idx
    (newer main-branch versions). Empty when ``states`` is None."""
    if states is None:
        return
    if isinstance(states, dict):
        for idx in sorted(states.keys()):
            tensor = states[idx]
            if tensor is not None:
                yield idx, tensor
    elif torch.is_tensor(states):
        yield 0, states


def zero_components(cache, config, zero_fa=False, zero_recurrent=False, zero_conv=False):
    """Same-shape zeroing of the cache components, per protocol section 7.

    FA layer KV is zeroed (token positions preserved); linear layers have their
    recurrent and/or conv state zeroed (every state_idx). Operates in place.
    """
    types = layer_types(config)
    for idx, layer in enumerate(cache.layers):
        is_linear = idx < len(types) and types[idx] == "linear_attention"
        if is_linear:
            if zero_conv and getattr(layer, "is_conv_states_initialized", False):
                for _, tensor in _state_iter(getattr(layer, "conv_states", None)):
                    tensor.zero_()
            if zero_recurrent and getattr(layer, "is_recurrent_states_initialized", False):
                for _, tensor in _state_iter(getattr(layer, "recurrent_states", None)):
                    tensor.zero_()
        else:
            if zero_fa and getattr(layer, "is_initialized", False) and getattr(layer, "keys", None) is not None:
                layer.keys.zero_()
                layer.values.zero_()


def collect_cache_tensors(cache):
    """Flatten every state tensor of a hybrid cache into
    ``(kind, layer_idx, state_idx, tensor)`` tuples for B-vs-C comparison."""
    out = []
    for idx, layer in enumerate(cache.layers):
        if getattr(layer, "is_initialized", False) and getattr(layer, "keys", None) is not None:
            out.append(("fa_key", idx, 0, layer.keys.detach().clone()))
            out.append(("fa_value", idx, 0, layer.values.detach().clone()))
        for sidx, tensor in _state_iter(getattr(layer, "conv_states", None)):
            out.append(("conv_state", idx, sidx, tensor.detach().clone()))
        for sidx, tensor in _state_iter(getattr(layer, "recurrent_states", None)):
            out.append(("recurrent_state", idx, sidx, tensor.detach().clone()))
    return out


def cache_tensors_equal(left, right, tol=0.0):
    """Compare two flattened cache-tensor lists. Returns (equal, max_abs, max_nmse)."""
    if len(left) != len(right):
        return False, float("inf"), float("inf")
    max_abs, max_nmse = 0.0, 0.0
    for (lk, li, ls, lt), (rk, ri, rs, rt) in zip(left, right):
        if lk != rk or li != ri or ls != rs or lt.shape != rt.shape:
            return False, float("inf"), float("inf")
        d = (lt.float() - rt.float()).abs().max().item()
        nm = (lt.float() - rt.float()).square().mean().item() / lt.float().square().mean().clamp_min(1e-12).item()
        max_abs = max(max_abs, d)
        max_nmse = max(max_nmse, nm)
    return max_abs <= tol, max_abs, max_nmse


def cache_manifest(cache, config, expected_prefix_len):
    """Per-layer structure dump used by Smoke-1 to validate every assumption,
    including the dict-vs-tensor container shape and has_previous_state."""
    types = layer_types(config)
    rows = []
    for idx, layer in enumerate(cache.layers):
        row = {
            "layer_idx": idx,
            "layer_type": types[idx] if idx < len(types) else "?",
            "layer_class": type(layer).__name__,
            "has_previous_state": getattr(layer, "has_previous_state", None),
        }
        if getattr(layer, "is_initialized", False) and getattr(layer, "keys", None) is not None and layer.keys.numel() > 0:
            row.update({
                "fa_key_shape": list(layer.keys.shape),
                "fa_value_shape": list(layer.values.shape),
                "fa_dtype": str(layer.keys.dtype),
                "fa_key_norm": layer.keys.float().norm().item(),
                "fa_value_norm": layer.values.float().norm().item(),
                "seq_len": int(layer.get_seq_length()),
            })
        if getattr(layer, "is_conv_states_initialized", False):
            conv = getattr(layer, "conv_states", None)
            row["conv_container"] = "dict" if isinstance(conv, dict) else ("tensor" if torch.is_tensor(conv) else None)
            row["conv_states"] = {
                str(sidx): {"shape": list(t.shape), "dtype": str(t.dtype), "norm": t.float().norm().item()}
                for sidx, t in _state_iter(conv)
            }
        if getattr(layer, "is_recurrent_states_initialized", False):
            rec = getattr(layer, "recurrent_states", None)
            row["recurrent_container"] = "dict" if isinstance(rec, dict) else ("tensor" if torch.is_tensor(rec) else None)
            row["recurrent_states"] = {
                str(sidx): {"shape": list(t.shape), "dtype": str(t.dtype), "norm": t.float().norm().item()}
                for sidx, t in _state_iter(rec)
            }
        rows.append(row)
    return {
        "num_layers": len(cache.layers),
        "get_seq_length": int(cache.get_seq_length()),
        "expected_prefix_len": int(expected_prefix_len),
        "has_previous_state": bool(cache.has_previous_state()),
        "layers": rows,
    }


# ---------------------------------------------------------------------------
# Forward helpers (Path A0/A1 continuous / Path B replay)
# ---------------------------------------------------------------------------

def prefill(model, ids):
    token_ids = torch.tensor([ids], dtype=torch.long, device=device())
    mask = torch.ones_like(token_ids)
    with torch.no_grad():
        output = model(input_ids=token_ids, attention_mask=mask, use_cache=True)
    return output.past_key_values


def forward_logits(model, ids_list, target_ids, prefix=0, cache=None):
    """Teacher-forced logits over the answer region.

    - Path A0 (continuous): ids_list = full_input_ids, cache=None -> use_cache=False
    - Path A1 (continuous + cache): ids_list = full_input_ids, cache=empty DynamicCache -> use_cache=True
    - Path B (replay): ids_list = question_input_ids, cache=cloned prefix cache.
    The model derives position_ids from cache.get_seq_length() on cache paths,
    so no manual cache_position is needed.
    """
    current = ids_list + target_ids[:-1]
    ids = torch.tensor([current], dtype=torch.long, device=device())
    mask = torch.ones(1, prefix + len(current), dtype=torch.long, device=device())
    with torch.no_grad():
        output = model(
            input_ids=ids, attention_mask=mask,
            past_key_values=cache, use_cache=cache is not None,
        )
    start = len(ids_list) - 1
    return output.logits[0, start:start + len(target_ids)].float().cpu()


def generate(model, tok, prompt_ids, cfg, prefix=0, cache=None):
    """Greedy decode with cache. First step feeds the whole prompt chunk
    (chunked kernel), then decodes one token at a time (fused recurrent kernel)."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device())
    mask = torch.ones(1, prefix + len(prompt_ids), dtype=torch.long, device=device())
    with torch.no_grad():
        output = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
    past = output.past_key_values
    generated = []
    next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(cfg["max_new_tokens"]):
        token = int(next_token.item())
        if token == tok.eos_token_id:
            break
        generated.append(token)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=device())], 1)
        with torch.no_grad():
            output = model(
                input_ids=next_token, attention_mask=mask,
                past_key_values=past, use_cache=True,
            )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    return tok.decode(generated, skip_special_tokens=True).strip(), generated
