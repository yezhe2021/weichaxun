import math
import random

import torch
from torch import nn
import torch.nn.functional as F


class HybridChunkCompressor(nn.Module):
    """One native token and one learned remainder slot per contiguous chunk."""

    def __init__(self, layers, slots=32, heads=8, dim=128):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(layers, heads, slots, dim))
        self.layers, self.slots, self.heads, self.dim = layers, slots, heads, dim

    def chunk_bounds(self, tokens):
        if tokens < 1:
            raise ValueError("Hybrid chunking needs at least one post-token0 token")
        return [(math.floor(i * tokens / self.slots), math.floor((i + 1) * tokens / self.slots))
                for i in range(self.slots)]

    def select(self, importance, arm, sample_seed):
        tokens = int(importance.numel())
        selected = []
        rng = random.Random(sample_seed)
        for start, end in self.chunk_bounds(tokens):
            if start == end:
                selected.append(-1)
            elif arm == "query":
                selected.append(start + int(importance[start:end].argmax()))
            elif arm == "center":
                selected.append((start + end - 1) // 2)
            elif arm == "random":
                selected.append(rng.randrange(start, end))
            else:
                raise ValueError(f"Unknown selector arm: {arm}")
        return torch.tensor(selected, dtype=torch.long, device=importance.device)

    def masks(self, tokens, selected, device):
        chunk = torch.zeros(self.slots, tokens, dtype=torch.bool, device=device)
        for i, (start, end) in enumerate(self.chunk_bounds(tokens)):
            chunk[i, start:end] = True
        native_valid = selected >= 0
        remainder = chunk.clone()
        for i, index in enumerate(selected.tolist()):
            if index >= 0:
                remainder[i, index] = False
        return native_valid, remainder.any(-1), remainder

    def attention(self, key, selected):
        if key.shape[0] != self.layers or tuple(key.shape[2:]) != (self.heads, self.dim):
            raise ValueError("Expected K [layers,tokens,heads,dim]")
        native_valid, slot_valid, mask = self.masks(key.shape[1], selected, key.device)
        keys = key.float().permute(0, 2, 1, 3)
        scale = keys.square().mean((-1, -2), keepdim=True).sqrt().clamp_min(1e-6).detach()
        scores = torch.einsum("lhsd,lhtd->lhst", self.queries, keys / scale) / math.sqrt(self.dim)
        safe = mask.clone()
        safe[~slot_valid, 0] = True
        weights = torch.softmax(scores.masked_fill(~safe[None, None], float("-inf")), -1)
        return weights * slot_valid[None, None, :, None], native_valid, slot_valid

    def forward(self, key, value, selected):
        if key.shape != value.shape:
            raise ValueError("K/V shape mismatch")
        weights, native_valid, slot_valid = self.attention(key, selected)
        safe_selected = selected.clamp_min(0)
        native_key = key[:, safe_selected].clone()
        native_value = value[:, safe_selected].clone()
        native_key[:, ~native_valid] = 0
        native_value[:, ~native_valid] = 0
        keys, values = key.float().permute(0, 2, 1, 3), value.float().permute(0, 2, 1, 3)
        slot_key = torch.einsum("lhst,lhtd->lhsd", weights, keys).permute(0, 2, 1, 3)
        slot_value = torch.einsum("lhst,lhtd->lhsd", weights, values).permute(0, 2, 1, 3)
        return (native_key, native_value, slot_key.to(key.dtype), slot_value.to(value.dtype),
                native_valid, slot_valid)

    def interleave(self, token0_key, token0_value, outputs):
        nk, nv, sk, sv, native_valid, slot_valid = outputs
        key = torch.stack((nk, sk), dim=2).flatten(1, 2)
        value = torch.stack((nv, sv), dim=2).flatten(1, 2)
        valid = torch.stack((native_valid, slot_valid), dim=1).flatten()
        return (torch.cat((token0_key, key), 1), torch.cat((token0_value, value), 1),
                torch.cat((torch.ones(1, dtype=torch.bool, device=valid.device), valid)))


def full_kl(student, teacher, temperature=1.0):
    student_log = F.log_softmax(student.float() / temperature, -1)
    teacher_log = F.log_softmax(teacher.detach().float() / temperature, -1)
    return F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) * temperature**2
