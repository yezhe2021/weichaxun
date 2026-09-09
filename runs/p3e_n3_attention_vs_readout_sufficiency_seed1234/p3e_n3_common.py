from pathlib import Path

import torch
import torch.nn as nn

from p3e_n2_common import ReadoutCache, payload_to_device, probe_loss


VALID_MODES = ("attention_only", "readout_only", "attention_readout")


class SufficiencyProbe(nn.Module):
    def __init__(
        self,
        mode,
        layers=16,
        query_heads=32,
        readout_dim=2560,
        hidden_dim=128,
        max_tokens=1024,
    ):
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.layers = layers
        self.query_heads = query_heads
        self.readout_dim = readout_dim
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens

        if mode != "attention_only":
            self.readout_norm = nn.LayerNorm(readout_dim)
            self.readout_projection = nn.Linear(readout_dim, hidden_dim)
            self.layer_embedding = nn.Parameter(
                torch.zeros(1, layers, hidden_dim)
            )
            layer_block = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.layer_mixer = nn.TransformerEncoder(layer_block, num_layers=2)
            self.position_to_layer = nn.MultiheadAttention(
                hidden_dim, 8, dropout=0.0, batch_first=True
            )

        if mode != "readout_only":
            self.attention_projection = nn.Sequential(
                nn.Linear(layers * query_heads, 256),
                nn.GELU(),
                nn.Linear(256, hidden_dim),
            )

        self.position_embedding = nn.Embedding(max_tokens, hidden_dim)
        if mode == "attention_readout":
            self.fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            )

        token_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(token_block, num_layers=2)
        self.answer_pool = nn.MultiheadAttention(
            hidden_dim, 8, dropout=0.0, batch_first=True
        )
        self.answer_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.support_head = nn.Linear(hidden_dim, 1)
        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)
        self.yesno_head = nn.Linear(hidden_dim, 2)

    def forward(self, readout, attention, valid_mask):
        if readout.shape != (self.layers, self.readout_dim):
            raise RuntimeError(f"Bad readout shape {tuple(readout.shape)}")
        if attention.ndim != 3 or attention.shape[:2] != (
            self.layers,
            self.query_heads,
        ):
            raise RuntimeError(f"Bad attention shape {tuple(attention.shape)}")
        tokens = attention.shape[-1]
        if tokens > self.max_tokens:
            raise RuntimeError("Probe token length exceeds configured maximum")

        positions = self.position_embedding(
            torch.arange(tokens, device=attention.device)
        ).unsqueeze(0)
        readout_tokens = None
        attention_tokens = None

        if self.mode != "attention_only":
            layer_state = self.readout_projection(
                self.readout_norm(readout.float())
            ).unsqueeze(0)
            layer_state = self.layer_mixer(
                layer_state + self.layer_embedding
            )
            readout_tokens, _ = self.position_to_layer(
                positions, layer_state, layer_state, need_weights=False
            )

        if self.mode != "readout_only":
            token_attention = attention.permute(2, 0, 1).reshape(
                tokens, self.layers * self.query_heads
            )
            attention_tokens = self.attention_projection(
                token_attention.float()
            ).unsqueeze(0)
            attention_tokens = attention_tokens + positions

        if self.mode == "attention_only":
            token_state = attention_tokens
        elif self.mode == "readout_only":
            token_state = readout_tokens + positions
        else:
            token_state = self.fusion(
                torch.cat((attention_tokens, readout_tokens), dim=-1)
            )

        padding = (~valid_mask.bool()).unsqueeze(0)
        mixed = self.token_mixer(
            token_state, src_key_padding_mask=padding
        )
        answer_state, _ = self.answer_pool(
            self.answer_query.expand(1, -1, -1),
            mixed,
            mixed,
            key_padding_mask=padding,
            need_weights=False,
        )
        mixed = mixed[0]
        return {
            "support_logits": self.support_head(mixed).squeeze(-1),
            "start_logits": self.start_head(mixed).squeeze(-1),
            "end_logits": self.end_head(mixed).squeeze(-1),
            "yesno_logits": self.yesno_head(answer_state[:, 0]),
        }


def load_probe(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    probe = SufficiencyProbe(checkpoint["mode"]).to(device)
    probe.load_state_dict(checkpoint["probe"])
    return probe, checkpoint


def checkpoint_path(root, mode):
    return Path(root) / mode / "checkpoint_best.pt"

