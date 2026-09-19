import math

import torch
from transformers import DynamicCache

from common import text_backbone, text_config


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), -1)


def rotary_embeddings(model, x, positions, layer_index):
    backbone = text_backbone(model)
    positions = positions.to(x.device).unsqueeze(0)
    config = text_config(model)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None or not isinstance(getattr(backbone.rotary_emb, "rope_type", None), dict):
        return backbone.rotary_emb(x.unsqueeze(0), positions)
    return backbone.rotary_emb(x.unsqueeze(0), positions, layer_types[layer_index])


def apply_rope(model, x, positions, layer_index):
    cos, sin = rotary_embeddings(model, x, positions, layer_index)
    cos, sin = cos[0, :, None], sin[0, :, None]
    return x * cos + rotate_half(x) * sin


def apply_supplied_rope(x, cos, sin):
    return x * cos[0, :, None] + rotate_half(x) * sin[0, :, None]


@torch.no_grad()
def capture_decision(model, full_ids, body_length, choice_ids):
    """Capture pre-RoPE KV and exact layer-type-specific decision attention."""
    captured_k, captured_v, importance, handles = {}, {}, {}, []
    backbone = text_backbone(model)

    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            if hasattr(module, "k_norm"):
                try:
                    key = module.k_norm(key.transpose(1, 2)).transpose(1, 2)
                except (RuntimeError, ValueError):
                    key = module.k_norm(key)
            value = module.v_proj(hidden).view(shape)
            query = module.q_proj(hidden[:, -1:]).view(1, 1, module.config.num_attention_heads, module.head_dim)
            if hasattr(module, "q_norm"):
                try:
                    query = module.q_norm(query.transpose(1, 2)).transpose(1, 2)
                except (RuntimeError, ValueError):
                    query = module.q_norm(query)
            supplied = kwargs.get("position_embeddings")
            if supplied is None:
                body_key = apply_rope(model, key[0, :body_length],
                                      torch.arange(body_length, device=hidden.device), index)
                decision_query = apply_rope(model, query[0],
                                            torch.tensor([len(full_ids) - 1], device=hidden.device), index)[0]
            else:
                cos, sin = supplied
                body_key = apply_supplied_rope(key[0, :body_length], cos[:, :body_length], sin[:, :body_length])
                decision_query = apply_supplied_rope(query[0], cos[:, -1:], sin[:, -1:])[0]
            repeat = module.config.num_attention_heads // module.config.num_key_value_heads
            repeated_key = body_key.repeat_interleave(repeat, dim=1)
            scaling = getattr(module, "scaling", 1.0 / math.sqrt(module.head_dim))
            scores = torch.einsum("hd,thd->ht", decision_query.float(), repeated_key.float()) * scaling
            candidate_scores = torch.softmax(scores[:, 1:], dim=-1).mean(0)
            layer_importance = torch.cat((torch.zeros(1, device=hidden.device), candidate_scores))
            captured_k[index], captured_v[index] = key[0, :body_length].cpu(), value[0, :body_length].cpu()
            importance[index] = layer_importance.cpu()
        return apply

    for index, layer in enumerate(backbone.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)
        output = backbone(input_ids=ids, attention_mask=torch.ones_like(ids),
                          position_ids=torch.arange(len(full_ids), device="cuda")[None], use_cache=False)
        logits = model.lm_head(output.last_hidden_state[:, -1])[0].float()
    finally:
        for handle in handles:
            handle.remove()
    layers = len(backbone.layers)
    if len(captured_k) != layers or len(importance) != layers:
        raise RuntimeError("Incomplete decision capture")
    score = torch.stack([importance[index] for index in range(layers)]).mean(0)
    score[0] = 0
    score /= score.sum().clamp_min(1e-12)
    return (torch.stack([captured_k[index] for index in range(layers)]),
            torch.stack([captured_v[index] for index in range(layers)]), score,
            logits[torch.tensor(choice_ids, device=logits.device)].cpu())


def make_cache(model, key, value, positions):
    dtype = next(model.parameters()).dtype
    key, value = key.to(dtype), value.to(dtype)
    rotated = torch.stack([apply_rope(model, layer, positions, index)
                           for index, layer in enumerate(key)])
    data = [(rotated[layer].permute(1, 0, 2).unsqueeze(0),
             value[layer].permute(1, 0, 2).unsqueeze(0)) for layer in range(key.shape[0])]
    return DynamicCache(ddp_cache_data=data, config=text_config(model))


def final_logits(model, ids, key=None, value=None, positions=None, suffix_start=None):
    cache = make_cache(model, key, value, positions) if key is not None else None
    prefix = int(cache.get_seq_length()) if cache is not None else 0
    start = prefix if suffix_start is None else int(suffix_start)
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    attention_mask = torch.ones((1, prefix + len(ids)), device="cuda", dtype=torch.long)
    output = text_backbone(model)(input_ids=input_ids, attention_mask=attention_mask,
                                  position_ids=torch.arange(start, start + len(ids), device="cuda")[None],
                                  past_key_values=cache, use_cache=False)
    return model.lm_head(output.last_hidden_state[:, -1])[0].float()
