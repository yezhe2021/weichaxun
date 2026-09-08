import copy
import math

import torch
from transformers import DynamicCache


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), -1)


def apply_rope(model, x, positions):
    positions = positions.to(x.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x.unsqueeze(0), positions)
    cos, sin = cos[0, :, None], sin[0, :, None]
    return x * cos + rotate_half(x) * sin


def make_cache(model, k, v, positions):
    dtype = next(model.parameters()).dtype
    k, v = k.to(dtype), v.to(dtype)
    if k.shape != v.shape or positions.numel() != k.shape[1]:
        raise ValueError("Invalid cache tensors/positions")
    rotated = torch.stack([apply_rope(model, layer, positions) for layer in k])
    data = [(rotated[l].permute(1, 0, 2).unsqueeze(0),
             v[l].permute(1, 0, 2).unsqueeze(0)) for l in range(k.shape[0])]
    return DynamicCache(ddp_cache_data=data, config=model.config)


def final_logits(model, ids, k=None, v=None, official=None, positions=None, suffix_start=None,
                 prefix_attention_mask=None):
    if official is not None and k is not None:
        raise ValueError("Ambiguous cache")
    cache = make_cache(model, k, v, positions) if k is not None else copy.deepcopy(official)
    prefix = int(cache.get_seq_length()) if cache is not None else 0
    start = prefix if suffix_start is None else int(suffix_start)
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    prefix_mask = (torch.ones(prefix, device="cuda", dtype=torch.long) if prefix_attention_mask is None
                   else prefix_attention_mask.to(device="cuda", dtype=torch.long))
    if prefix_mask.ndim != 1 or prefix_mask.numel() != prefix:
        raise ValueError("Invalid prefix mask")
    attention_mask = torch.cat((prefix_mask, torch.ones(len(ids), device="cuda", dtype=torch.long)))[None]
    output = model.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=torch.arange(start, start + len(ids), device="cuda")[None],
                         past_key_values=cache, use_cache=False)
    return model.lm_head(output.last_hidden_state[:, -1])[0].float()


@torch.no_grad()
def query_context_importance(model, prefix_ids, suffix_ids, pre_rope_context_key):
    """Native RoPE/GQA Query-to-Context attention, normalized over Context only."""
    prefix_len, full_ids = len(prefix_ids), prefix_ids + suffix_ids
    per_layer, handles = {}, []

    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            query = module.q_proj(hidden[:, prefix_len:]).view(
                len(suffix_ids), module.config.num_attention_heads, module.head_dim)
            if hasattr(module, "q_norm"):
                query = module.q_norm(query)
            query = apply_rope(model, query,
                               torch.arange(prefix_len, len(full_ids), device=query.device))
            key = pre_rope_context_key[index].to(query.device)
            key = apply_rope(model, key, torch.arange(prefix_len, device=query.device))
            repeat = module.config.num_attention_heads // module.config.num_key_value_heads
            key = key.permute(1, 0, 2).repeat_interleave(repeat, 0)
            query = query.permute(1, 0, 2)
            scores = torch.einsum("hjd,htd->hjt", query.float(), key.float()) / math.sqrt(module.head_dim)
            per_layer[index] = torch.softmax(scores, -1).amax(1).mean(0).cpu()
        return apply

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    position_ids=torch.arange(len(full_ids), device="cuda")[None], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(per_layer) != len(model.model.layers):
        raise RuntimeError("Incomplete Query importance capture")
    importance = torch.stack([per_layer[i] for i in range(len(per_layer))]).mean(0)
    if importance.numel() != prefix_len or not torch.isfinite(importance).all():
        raise RuntimeError("Invalid Query importance")
    return importance
