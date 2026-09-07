import math

import torch
from torch import nn
import torch.nn.functional as F


class HardChunkSlotCompressor(nn.Module):
    """One learned attention-pooling slot per balanced contiguous token chunk."""

    def __init__(self, layers, slots=32, heads=8, dim=128):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(layers, heads, slots, dim))
        self.layers, self.slots, self.heads, self.dim = layers, slots, heads, dim

    def chunk_bounds(self, tokens):
        if tokens < 1:
            raise ValueError("Hard chunking needs at least one token")
        bounds = [(math.floor(index * tokens / self.slots),
                   math.floor((index + 1) * tokens / self.slots))
                  for index in range(self.slots)]
        if bounds[0][0] != 0 or bounds[-1][1] != tokens or any(end < start for start, end in bounds):
            raise RuntimeError("Invalid hard-chunk partition")
        return bounds

    def chunk_sizes(self, tokens):
        return [end - start for start, end in self.chunk_bounds(tokens)]

    def chunk_mask(self, tokens, device):
        mask = torch.zeros(self.slots, tokens, dtype=torch.bool, device=device)
        for index, (start, end) in enumerate(self.chunk_bounds(tokens)):
            mask[index, start:end] = True
        return mask

    def valid_slots(self, tokens, device):
        return self.chunk_mask(tokens, device).any(-1)

    def attention(self, key):
        if key.shape[0] != self.layers or tuple(key.shape[2:]) != (self.heads, self.dim):
            raise ValueError("Expected K [layers,tokens,heads,dim]")
        keys = key.float().permute(0, 2, 1, 3)
        # Preserve the content-score normalization used by the soft-local baseline.
        scale = keys.square().mean((-1, -2), keepdim=True).sqrt().clamp_min(1e-6).detach()
        scores = torch.einsum("lhsd,lhtd->lhst", self.queries, keys / scale) / math.sqrt(self.dim)
        mask = self.chunk_mask(key.shape[1], key.device)
        valid = mask.any(-1)
        # Empty chunks occur only when a sequence has fewer tokens than slots. Give
        # softmax a safe temporary element, then zero the whole invalid row. The
        # receiver also masks the corresponding zero-KV cache position.
        safe_mask = mask.clone()
        safe_mask[~valid, 0] = True
        weights = torch.softmax(scores.masked_fill(~safe_mask[None, None], float("-inf")), dim=-1)
        return weights * valid[None, None, :, None]

    def forward(self, key, value):
        if key.shape != value.shape:
            raise ValueError("K/V shape mismatch")
        weights = self.attention(key)
        keys = key.float().permute(0, 2, 1, 3)
        values = value.float().permute(0, 2, 1, 3)
        pooled = [torch.einsum("lhst,lhtd->lhsd", weights, tensor).permute(0, 2, 1, 3)
                  for tensor in (keys, values)]
        return tuple(tensor.to(key.dtype) for tensor in pooled)


def full_kl(student, teacher, temperature=1.0):
    student_log = F.log_softmax(student.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher.detach().float() / temperature, dim=-1)
    return F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) * temperature**2
