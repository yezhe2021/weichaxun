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


@torch.no_grad()
def capture_decision(model, full_ids, body_length, choice_ids):
    """One forward captures body pre-RoPE KV and decision-token attention over body candidates."""
    captured_k, captured_v, importance, handles = {}, {}, {}, []

    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            if hasattr(module, "k_norm"): key = module.k_norm(key)
            value = module.v_proj(hidden).view(shape)
            query = module.q_proj(hidden[:, -1:]).view(1, 1, module.config.num_attention_heads, module.head_dim)
            if hasattr(module, "q_norm"): query = module.q_norm(query)
            body_key = apply_rope(model, key[0, :body_length], torch.arange(body_length, device=hidden.device))
            decision_query = apply_rope(model, query[0], torch.tensor([len(full_ids) - 1], device=hidden.device))[0]
            repeat = module.config.num_attention_heads // module.config.num_key_value_heads
            repeated_key = body_key.repeat_interleave(repeat, dim=1)
            scores = torch.einsum("hd,thd->ht", decision_query.float(), repeated_key.float()) / math.sqrt(module.head_dim)
            candidate_scores = torch.softmax(scores[:, 1:], dim=-1).mean(0)
            layer_importance = torch.cat((torch.zeros(1, device=hidden.device), candidate_scores))
            captured_k[index], captured_v[index] = key[0, :body_length].cpu(), value[0, :body_length].cpu()
            importance[index] = layer_importance.cpu()
        return apply

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)
        output = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             position_ids=torch.arange(len(full_ids), device="cuda")[None], use_cache=False)
        logits = model.lm_head(output.last_hidden_state[:, -1])[0].float()
    finally:
        for handle in handles: handle.remove()
    layers = len(model.model.layers)
    if len(captured_k) != layers or len(importance) != layers: raise RuntimeError("Incomplete decision capture")
    score = torch.stack([importance[index] for index in range(layers)]).mean(0)
    score[0] = 0
    score /= score.sum().clamp_min(1e-12)
    return (torch.stack([captured_k[index] for index in range(layers)]),
            torch.stack([captured_v[index] for index in range(layers)]), score,
            logits[torch.tensor(choice_ids, device=logits.device)].cpu())


def make_cache(model, key, value, positions):
    dtype = next(model.parameters()).dtype
    key, value = key.to(dtype), value.to(dtype)
    rotated = torch.stack([apply_rope(model, layer, positions) for layer in key])
    data = [(rotated[layer].permute(1, 0, 2).unsqueeze(0),
             value[layer].permute(1, 0, 2).unsqueeze(0)) for layer in range(key.shape[0])]
    return DynamicCache(ddp_cache_data=data, config=model.config)


def final_logits(model, ids, key=None, value=None, positions=None, suffix_start=None):
    cache = make_cache(model, key, value, positions) if key is not None else None
    prefix = int(cache.get_seq_length()) if cache is not None else 0
    start = prefix if suffix_start is None else int(suffix_start)
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    attention_mask = torch.ones((1, prefix + len(ids)), device="cuda", dtype=torch.long)
    output = model.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=torch.arange(start, start + len(ids), device="cuda")[None],
                         past_key_values=cache, use_cache=False)
    return model.lm_head(output.last_hidden_state[:, -1])[0].float()
