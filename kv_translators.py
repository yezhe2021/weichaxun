import json
import math
from pathlib import Path

import torch
import torch.nn as nn


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_transform(k, rope_theta, inverse=False):
    dim = k.shape[-1]
    positions = torch.arange(k.shape[-2], device=k.device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=k.device, dtype=torch.float32) / dim))
    angles = torch.outer(positions, inv_freq)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(k.dtype).view(1, 1, k.shape[-2], dim)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(k.dtype).view(1, 1, k.shape[-2], dim)
    if inverse:
        sin = -sin
    return k * cos + rotate_half(k) * sin


class PseudoSenderTranslator(nn.Module):
    """Layer-local translator from a fixed compressed pseudo-sender representation."""

    def __init__(
        self,
        num_layers,
        num_kv_heads,
        head_dim,
        bottleneck=128,
        hidden=512,
        seed=1234,
        trainable_encoder=False,
        rope_disentangled=False,
        rope_theta=1_000_000.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.input_dim = 2 * num_kv_heads * head_dim
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.seed = seed
        self.trainable_encoder = trainable_encoder
        self.rope_disentangled = rope_disentangled
        self.rope_theta = rope_theta
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bottleneck, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.input_dim),
            )
            for _ in range(num_layers)
        ])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for layer in range(num_layers):
            projection = torch.randn(self.input_dim, bottleneck, generator=generator) / math.sqrt(self.input_dim)
            self.register_buffer(f"projection_{layer}", projection, persistent=True)
        if trainable_encoder:
            self.encoder_deltas = nn.ParameterList([
                nn.Parameter(torch.zeros(self.input_dim, bottleneck)) for _ in range(num_layers)
            ])
        else:
            self.encoder_deltas = None

    def config_dict(self):
        return {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "bottleneck": self.bottleneck,
            "hidden": self.hidden,
            "seed": self.seed,
            "trainable_encoder": self.trainable_encoder,
            "rope_disentangled": self.rope_disentangled,
            "rope_theta": self.rope_theta,
        }

    def _pack(self, k, v):
        if self.rope_disentangled:
            k = rope_transform(k, self.rope_theta, inverse=True)
        batch, heads, tokens, dim = k.shape
        k_flat = k.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim)
        v_flat = v.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim)
        return torch.cat([k_flat, v_flat], dim=-1)

    def _unpack(self, x, reference_k):
        batch, tokens, _ = x.shape
        split = self.num_kv_heads * self.head_dim
        k = x[..., :split].reshape(batch, tokens, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = x[..., split:].reshape(batch, tokens, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.rope_disentangled:
            k = rope_transform(k, self.rope_theta, inverse=False)
        return k.to(reference_k.dtype), v.to(reference_k.dtype)

    def forward(self, pairs):
        if len(pairs) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} cache layers, got {len(pairs)}")
        output = []
        for layer, (k, v) in enumerate(pairs):
            x = self._pack(k, v).float()
            scale = x.pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
            normalized = x / scale
            projection = getattr(self, f"projection_{layer}")
            if self.encoder_deltas is not None:
                projection = projection + self.encoder_deltas[layer]
            source = normalized @ projection
            decoded = self.decoders[layer](source) * scale
            output.append(self._unpack(decoded, k))
        return output


def save_translator(path, translator, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "translator_config": translator.config_dict(),
            "state_dict": translator.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_translator(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    translator = PseudoSenderTranslator(**checkpoint["translator_config"])
    translator.load_state_dict(checkpoint["state_dict"])
    return translator, checkpoint.get("metadata", {})
