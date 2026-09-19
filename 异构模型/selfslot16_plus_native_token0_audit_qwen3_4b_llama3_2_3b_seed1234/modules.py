import math

import torch
from torch import nn
import torch.nn.functional as F


class SoftLocalSlotCompressor(nn.Module):
    """Ordered slots with content attention plus a normalized soft-locality bias."""

    def __init__(self, layers, slots=64, heads=8, dim=128, locality_strength=8.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(layers, heads, slots, dim))
        self.layers, self.slots, self.heads, self.dim = layers, slots, heads, dim
        self.locality_strength = float(locality_strength)

    def positional_bias(self, tokens, device):
        token_position = (torch.arange(tokens, device=device, dtype=torch.float32) + 0.5) / tokens
        slot_center = (torch.arange(self.slots, device=device, dtype=torch.float32) + 0.5) / self.slots
        return -self.locality_strength * (slot_center[:, None] - token_position[None, :]).abs()

    def attention(self, key):
        if key.shape[0] != self.layers or tuple(key.shape[2:]) != (self.heads, self.dim):
            raise ValueError("Expected K [layers,tokens,heads,dim]")
        keys = key.float().permute(0, 2, 1, 3)
        scale = keys.square().mean((-1, -2), keepdim=True).sqrt().clamp_min(1e-6).detach()
        content = torch.einsum("lhsd,lhtd->lhst", self.queries, keys / scale) / math.sqrt(self.dim)
        return torch.softmax(content + self.positional_bias(key.shape[1], key.device)[None, None], dim=-1)

    def forward(self, key, value):
        if key.shape != value.shape: raise ValueError("K/V shape mismatch")
        weights = self.attention(key)
        keys, values = key.float().permute(0, 2, 1, 3), value.float().permute(0, 2, 1, 3)
        pooled = [torch.einsum("lhst,lhtd->lhsd", weights, tensor).permute(0, 2, 1, 3)
                  for tensor in (keys, values)]
        return tuple(tensor.to(key.dtype) for tensor in pooled)


def full_kl(student, teacher, temperature=1.0):
    student_log = F.log_softmax(student.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher.detach().float() / temperature, dim=-1)
    return F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) * temperature**2
