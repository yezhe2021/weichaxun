from pathlib import Path

import torch
import torch.nn as nn

from real_kv_common import fixed_index_map, rope_transform


def mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class RealCrossModelKVTranslator(nn.Module):
    """Context-only KV translator: Qwen3-0.6B sender cache -> Qwen3-1.7B receiver cache shape."""

    def __init__(
        self,
        sender_layers,
        sender_kv_heads,
        sender_head_dim,
        sender_rope_theta,
        receiver_layers,
        receiver_kv_heads,
        receiver_head_dim,
        receiver_rope_theta,
        hidden=512,
        gate_init=2.0,
    ):
        super().__init__()
        self.sender_layers = int(sender_layers)
        self.sender_kv_heads = int(sender_kv_heads)
        self.sender_head_dim = int(sender_head_dim)
        self.sender_rope_theta = float(sender_rope_theta)
        self.receiver_layers = int(receiver_layers)
        self.receiver_kv_heads = int(receiver_kv_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.receiver_rope_theta = float(receiver_rope_theta)
        self.hidden = int(hidden)
        self.layer_map = fixed_index_map(self.sender_layers, self.receiver_layers)
        self.head_map = fixed_index_map(self.sender_kv_heads, self.receiver_kv_heads)
        self.k_mlps = nn.ModuleList(
            [
                nn.ModuleList(
                    [mlp(self.sender_head_dim, self.hidden, self.receiver_head_dim) for _ in range(self.receiver_kv_heads)]
                )
                for _ in range(self.receiver_layers)
            ]
        )
        self.v_mlps = nn.ModuleList(
            [
                nn.ModuleList(
                    [mlp(self.sender_head_dim, self.hidden, self.receiver_head_dim) for _ in range(self.receiver_kv_heads)]
                )
                for _ in range(self.receiver_layers)
            ]
        )
        self.k_gate_logits = nn.Parameter(torch.full((self.receiver_layers, self.receiver_kv_heads), float(gate_init)))
        self.v_gate_logits = nn.Parameter(torch.full((self.receiver_layers, self.receiver_kv_heads), float(gate_init)))

    def config_dict(self):
        return {
            "sender_layers": self.sender_layers,
            "sender_kv_heads": self.sender_kv_heads,
            "sender_head_dim": self.sender_head_dim,
            "sender_rope_theta": self.sender_rope_theta,
            "receiver_layers": self.receiver_layers,
            "receiver_kv_heads": self.receiver_kv_heads,
            "receiver_head_dim": self.receiver_head_dim,
            "receiver_rope_theta": self.receiver_rope_theta,
            "hidden": self.hidden,
            "layer_map": self.layer_map,
            "head_map": self.head_map,
            "gate_mode": "pure_translate_multiplicative",
        }

    def forward(self, sender_pairs):
        if len(sender_pairs) != self.sender_layers:
            raise ValueError(f"Expected {self.sender_layers} sender layers, got {len(sender_pairs)}")
        output = []
        for receiver_layer in range(self.receiver_layers):
            sender_layer = self.layer_map[receiver_layer]
            sender_k, sender_v = sender_pairs[sender_layer]
            sender_k_base = rope_transform(sender_k, self.sender_rope_theta, inverse=True)
            layer_k = []
            layer_v = []
            for receiver_head in range(self.receiver_kv_heads):
                sender_head = self.head_map[receiver_head]
                k_in = sender_k_base[:, sender_head].float()
                v_in = sender_v[:, sender_head].float()
                k_base = self.k_mlps[receiver_layer][receiver_head](k_in)
                v_out = self.v_mlps[receiver_layer][receiver_head](v_in)
                layer_k.append(k_base)
                layer_v.append(v_out)
            k_base = torch.stack(layer_k, dim=1)
            v_out = torch.stack(layer_v, dim=1)
            k_out = rope_transform(k_base, self.receiver_rope_theta, inverse=False)
            k_gate = torch.sigmoid(self.k_gate_logits[receiver_layer]).view(1, self.receiver_kv_heads, 1, 1)
            v_gate = torch.sigmoid(self.v_gate_logits[receiver_layer]).view(1, self.receiver_kv_heads, 1, 1)
            output.append(((k_out * k_gate).to(sender_k.dtype), (v_out * v_gate).to(sender_v.dtype)))
        return output


def save_real_translator(path, translator, metadata=None):
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


def load_real_translator(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = dict(checkpoint["translator_config"])
    config.pop("layer_map", None)
    config.pop("head_map", None)
    config.pop("gate_mode", None)
    translator = RealCrossModelKVTranslator(**config)
    translator.load_state_dict(checkpoint["state_dict"])
    return translator, checkpoint.get("metadata", {})
