import math

import torch
from torch import nn
import torch.nn.functional as F


class SlotCompressor(nn.Module):
    """One learnable query per layer, KV head and slot; shared K/V pooling weights."""
    def __init__(self, layers, slots=64, heads=8, dim=128):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(layers, heads, slots, dim))
        self.layers, self.slots, self.heads, self.dim = layers, slots, heads, dim

    def forward(self, k, v):
        if k.shape != v.shape or k.shape[0] != self.layers or tuple(k.shape[2:]) != (self.heads, self.dim):
            raise ValueError('Expected K/V [layers,tokens,heads,dim]')
        # Score-only RMS normalization keeps Llama/Qwen query learning at a similar scale.
        # Pooled K and V retain their original, pre-RoPE coordinates and magnitudes.
        keys, values = k.float().permute(0, 2, 1, 3), v.float().permute(0, 2, 1, 3)
        scale = keys.square().mean((-1, -2), keepdim=True).sqrt().clamp_min(1e-6).detach()
        weights = torch.softmax(torch.einsum('lhsd,lhtd->lhst', self.queries, keys / scale) / math.sqrt(self.dim), dim=-1)
        pooled = [torch.einsum('lhst,lhtd->lhsd', weights, x).permute(0, 2, 1, 3) for x in (keys, values)]
        return tuple(x.to(k.dtype) for x in pooled)


class RegistrationMapper(nn.Module):
    """Factorized full-depth then full-head linear maps, K/V separate and bias free."""
    def __init__(self, source=28, target=36, heads=8, dim=128):
        super().__init__()
        self.source, self.target, self.heads, self.dim = source, target, heads, dim
        self.k_depth = nn.ModuleList(nn.Linear(source * dim, dim, bias=False) for _ in range(target))
        self.v_depth = nn.ModuleList(nn.Linear(source * dim, dim, bias=False) for _ in range(target))
        # Each output row-block is the independent W of one target head.
        self.k_heads = nn.ModuleList(nn.Linear(heads * dim, heads * dim, bias=False) for _ in range(target))
        self.v_heads = nn.ModuleList(nn.Linear(heads * dim, heads * dim, bias=False) for _ in range(target))
        with torch.no_grad():
            for j in range(target):
                nearest = round(j * (source - 1) / max(target - 1, 1))
                for family in (self.k_depth, self.v_depth):
                    family[j].weight.zero_()
                    family[j].weight[:, nearest * dim:(nearest + 1) * dim].copy_(torch.eye(dim))
                for family in (self.k_heads, self.v_heads): family[j].weight.copy_(torch.eye(heads * dim))

    def one(self, x, depth, heads):
        # Concatenate layers, preserving slots and source heads; then mix heads within each slot.
        features = x.float().permute(1, 2, 0, 3).flatten(-2)
        return torch.stack([heads[j](depth[j](features).flatten(-2)).view(x.shape[1], self.heads, self.dim)
                            for j in range(self.target)]).to(x.dtype)

    def forward(self, k, v):
        return self.one(k, self.k_depth, self.k_heads), self.one(v, self.v_depth, self.v_heads)


class CrossWriter(nn.Module):
    def __init__(self, slots):
        super().__init__()
        self.compressor = SlotCompressor(28, slots)
        self.mapper = RegistrationMapper()

    def forward(self, k, v):
        return self.mapper(*self.compressor(k, v))


def full_kl(student, teacher, temperature=1.0):
    if student.shape != teacher.shape: raise ValueError('KL requires the same receiver vocabulary')
    s, t = F.log_softmax(student.float() / temperature, -1), F.log_softmax(teacher.detach().float() / temperature, -1)
    return F.kl_div(s, t, reduction='sum', log_target=True) * temperature**2


def representation_loss(pk, pv, tk, tv):
    loss, metrics = pk.new_zeros((), dtype=torch.float32), {}
    for name, pred, target in [('k', pk, tk), ('v', pv, tv)]:
        pred, target = pred.float(), target.detach().float()
        nmse = (pred - target).square().mean((1, 2, 3)) / target.square().mean((1, 2, 3)).clamp_min(1e-8)
        cosine = F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1)
        loss = loss + nmse.mean() + 1 - cosine.mean()
        metrics[name + '_nmse'] = nmse.detach().cpu().tolist()
        metrics[name + '_cosine'] = cosine.detach().cpu().tolist()
    return loss, metrics
