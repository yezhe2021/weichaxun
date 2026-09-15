import copy

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from common import digest, load_model, log, manifests, run_root, save_json, save_tensor


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def post_rope(model, k, positions=None):
    if positions is None:
        positions = torch.arange(k.shape[1], device=k.device)
    if positions.ndim != 1 or positions.numel() != k.shape[1]:
        raise ValueError('K/position length mismatch')
    positions = positions.to(k.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(k[0].unsqueeze(0), positions)
    cos, sin = cos[0][None, :, None, :], sin[0][None, :, None, :]
    return k * cos + rotate_half(k) * sin


def make_cache(model, k, v, positions=None):
    dtype = next(model.parameters()).dtype
    k, v = k.to(dtype), v.to(dtype)
    if k.shape != v.shape: raise ValueError('K/V shape mismatch')
    pk = post_rope(model, k, positions)
    data = [(pk[l].permute(1, 0, 2).unsqueeze(0), v[l].permute(1, 0, 2).unsqueeze(0)) for l in range(k.shape[0])]
    return DynamicCache(ddp_cache_data=data, config=model.config)


def final_logits(model, ids, k=None, v=None, official=None, positions=None, suffix_start=None,
                 prefix_attention_mask=None):
    if official is not None and k is not None: raise ValueError('Ambiguous cache')
    cache = make_cache(model, k, v, positions) if k is not None else copy.deepcopy(official)
    prefix = int(cache.get_seq_length()) if cache is not None else 0
    start = prefix if suffix_start is None else int(suffix_start)
    input_ids = torch.tensor([ids], device='cuda', dtype=torch.long)
    if prefix_attention_mask is None:
        prefix_mask = torch.ones(prefix, dtype=torch.long, device='cuda')
    else:
        prefix_mask = prefix_attention_mask.to(device='cuda', dtype=torch.long)
        if prefix_mask.ndim != 1 or prefix_mask.numel() != prefix:
            raise ValueError('Invalid prefix attention mask')
    attention_mask = torch.cat((prefix_mask, torch.ones(len(ids), dtype=torch.long, device='cuda'))).unsqueeze(0)
    output = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=torch.arange(start, start + len(ids), device='cuda').unsqueeze(0),
        past_key_values=cache, use_cache=False)
    # Compute only the single final-position vocabulary vector, not T x vocabulary logits.
    return model.lm_head(output.last_hidden_state[:, -1])[0].float()


@torch.no_grad()
def capture(model, ids):
    captured_k, captured_v, handles = {}, {}, []
    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            if hasattr(module, 'k_norm'): key = module.k_norm(key)
            val = module.v_proj(hidden).view(shape)
            captured_k[index], captured_v[index] = key[0].cpu(), val[0].cpu()
        return apply
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(i), with_kwargs=True))
    try:
        t = torch.tensor([ids], device='cuda', dtype=torch.long)
        output = model.model(input_ids=t, attention_mask=torch.ones_like(t),
                             position_ids=torch.arange(len(ids), device='cuda').unsqueeze(0), use_cache=True)
    finally:
        for handle in handles: handle.remove()
    if len(captured_k) != len(model.model.layers): raise RuntimeError('Incomplete pre-RoPE capture')
    k = torch.stack([captured_k[i] for i in range(len(captured_k))])
    v = torch.stack([captured_v[i] for i in range(len(captured_v))])
    return k, v, output.past_key_values


def cache_path(cfg, family, split, row):
    return run_root(cfg) / 'cache' / family / split / (row['id'] + '.pt')


def load_native(cfg, family, split, row):
    data = torch.load(cache_path(cfg, family, split, row), map_location='cpu', weights_only=True)
    if data['signature'] != cfg['signature'] or data['tokens_hash'] != digest(row['encoded'][family]):
        raise RuntimeError('Native cache does not match config/tokenization')
    return data


def build(cfg, family):
    model = load_model(cfg, family)
    audit = []
    try:
        for split in cfg['splits']:
            rows = manifests(cfg, split)
            for i, row in enumerate(rows):
                path = cache_path(cfg, family, split, row)
                need_audit = split == 'validation' and i < cfg['audit_samples']
                if path.exists() and not need_audit:
                    load_native(cfg, family, split, row)
                    continue
                fields = row['encoded'][family]
                k, v, official = capture(model, fields['prefix'])
                with torch.no_grad():
                    native = final_logits(model, fields['suffix'], k.cuda(), v.cuda())
                    if need_audit:
                        full = final_logits(model, fields['full'])
                        off = final_logits(model, fields['suffix'], official=official)
                        rebuilt = make_cache(model, k.cuda(), v.cuda())
                        cache_error = max(max((x.keys - y.keys).abs().max().item(), (x.values - y.values).abs().max().item())
                                          for x, y in zip(official.layers, rebuilt.layers))
                        cos = F.cosine_similarity(full[None], native[None]).item()
                        maxerr = (full - native).abs().max().item()
                        offerr = (off - native).abs().max().item()
                        choices = fields['choice_ids']
                        match = full[choices].argmax().item() == native[choices].argmax().item()
                        passed = cos >= .9999 and maxerr <= .3 and offerr <= 1e-5 and cache_error <= 1e-5 and match
                        audit.append({'id': row['id'], 'cosine': cos, 'max_abs_error': maxerr,
                                      'official_manual_logits_error': offerr, 'cache_error': cache_error,
                                      'choice_agreement': match, 'passed': passed})
                        save_json(run_root(cfg) / 'audit' / f'{family}.json', audit)
                        if not passed: raise RuntimeError(f'{family} native cache protocol audit failed: {audit[-1]}')
                save_tensor(path, {'signature': cfg['signature'], 'tokens_hash': digest(fields), 'k': k, 'v': v,
                                   'native_logits': native.cpu()})
                del official, k, v, native
                log(f'{family} cache {split} {i + 1}/{len(rows)}')
    finally:
        del model
        torch.cuda.empty_cache()
