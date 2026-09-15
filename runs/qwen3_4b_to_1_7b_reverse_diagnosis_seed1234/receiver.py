from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from data import cuda


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), -1)


def apply_rope(model, tensor, positions):
    position_ids = torch.tensor([positions], dtype=torch.long, device=tensor.device)
    dummy = torch.empty(1, len(positions), model.config.hidden_size, dtype=tensor.dtype, device=tensor.device)
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    return tensor * cos[0][None, :, None, :] + rotate_half(tensor) * sin[0][None, :, None, :]


def dynamic_cache(model, pre_key, value, positions=None):
    positions = list(range(pre_key.shape[1])) if positions is None else positions
    post_key = apply_rope(model, pre_key, positions)
    items = [(post_key[layer].permute(1, 0, 2).unsqueeze(0), value[layer].permute(1, 0, 2).unsqueeze(0)) for layer in range(pre_key.shape[0])]
    return DynamicCache(ddp_cache_data=items, config=model.config)


def heterogeneous_cache(model, pre_key, value, total_layers=36):
    """First layers contain context; remaining layers contain true length-zero caches."""
    post_key = apply_rope(model, pre_key, list(range(pre_key.shape[1])))
    items = []
    for layer in range(total_layers):
        if layer < pre_key.shape[0]:
            k = post_key[layer].permute(1, 0, 2).unsqueeze(0)
            v = value[layer].permute(1, 0, 2).unsqueeze(0)
        else:
            k = post_key.new_empty((1, post_key.shape[2], 0, post_key.shape[3]))
            v = value.new_empty((1, value.shape[2], 0, value.shape[3]))
        items.append((k, v))
    cache = DynamicCache(ddp_cache_data=items, config=model.config)
    lengths = [cache.get_seq_length(i) for i in range(total_layers)]
    expected = [pre_key.shape[1]] * pre_key.shape[0] + [0] * (total_layers - pre_key.shape[0])
    if lengths != expected:
        raise RuntimeError(f"heterogeneous cache lengths mismatch: {lengths} != {expected}")
    return cache


def causal_mask(hidden, query_length, past_length):
    total = past_length + query_length
    mask = torch.full((query_length, total), torch.finfo(hidden.dtype).min, dtype=hidden.dtype, device=hidden.device)
    for row in range(query_length):
        mask[row, : past_length + row + 1] = 0
    return mask[None, None]


def heterogeneous_forward(model, input_ids, position_ids, cache):
    """Qwen3 forward with a distinct causal mask/cache length for every decoder layer."""
    hidden = model.model.embed_tokens(input_ids)
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    query_length = input_ids.shape[1]
    for index, layer in enumerate(model.model.layers):
        past_length = cache.get_seq_length(index)
        mask = causal_mask(hidden, query_length, past_length)
        hidden = layer(
            hidden,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden), cache


def answer_logits(model, sample, key=None, value=None, prompt_kind="suffix", skip=False):
    prompt = sample["question_suffix_ids"] if prompt_kind == "suffix" else sample["question_only_ids"]
    target = sample["answer_token_ids"]
    current = prompt + target[:-1]
    prefix = 0 if key is None else key.shape[1]
    ids = torch.tensor([current], dtype=torch.long, device=cuda())
    positions = torch.arange(prefix, prefix + len(current), device=cuda()).unsqueeze(0)
    if skip:
        output, _ = heterogeneous_forward(model, ids, positions, heterogeneous_cache(model, key, value))
        logits_all = output
    else:
        mask = torch.ones(1, prefix + len(current), dtype=torch.long, device=cuda())
        kwargs = {"past_key_values": dynamic_cache(model, key, value)} if key is not None else {}
        logits_all = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, **kwargs).logits
    start = len(prompt) - 1
    logits = logits_all[0, start:start + len(target)].float()
    gold = torch.tensor(target, dtype=torch.long, device=cuda())
    return logits, gold


def ce_loss(model, sample, key, value, skip=False):
    logits, gold = answer_logits(model, sample, key, value, skip=skip)
    return F.cross_entropy(logits, gold), logits, gold


@torch.no_grad()
def generate(model, tok, sample, cfg, key=None, value=None, prompt_kind="suffix", skip=False):
    prompt = sample["question_suffix_ids"] if prompt_kind == "suffix" else sample["question_only_ids"]
    prefix = 0 if key is None else key.shape[1]
    ids = torch.tensor([prompt], dtype=torch.long, device=cuda())
    positions = torch.arange(prefix, prefix + len(prompt), device=cuda()).unsqueeze(0)
    if skip:
        cache = heterogeneous_cache(model, key, value)
        logits, past = heterogeneous_forward(model, ids, positions, cache)
    else:
        mask = torch.ones(1, prefix + len(prompt), dtype=torch.long, device=cuda())
        kwargs = {"past_key_values": dynamic_cache(model, key, value)} if key is not None else {}
        output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True, **kwargs)
        logits, past = output.logits, output.past_key_values
    next_token = logits[:, -1].argmax(-1, keepdim=True)
    generated = []
    next_position = prefix + len(prompt)
    for _ in range(cfg["max_new_tokens"]):
        token = int(next_token.item())
        if token == tok.eos_token_id:
            break
        generated.append(token)
        position = torch.tensor([[next_position]], device=cuda())
        if skip:
            logits, past = heterogeneous_forward(model, next_token, position, past)
        else:
            total = past.get_seq_length() + 1
            output = model(input_ids=next_token, attention_mask=torch.ones(1, total, dtype=torch.long, device=cuda()), position_ids=position, past_key_values=past, use_cache=True)
            logits, past = output.logits, output.past_key_values
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    return tok.decode(generated, skip_special_tokens=True).strip(), len(generated)
