from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from p3d3_common import read_json


class ReadoutCache:
    def __init__(self, index_path, capacity=8):
        self.path = Path(index_path)
        self.root = self.path.parent
        self.index = read_json(index_path)
        self.entries = self.index["entries"]
        self.capacity = capacity
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index not in self.loaded:
            self.loaded[index] = torch.load(
                self.root / self.entries[index]["file"],
                map_location="cpu",
                weights_only=False,
            )
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]


def contiguous_spans(mask):
    spans = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(mask) - 1):
            end = index if active and index == len(mask) - 1 else index - 1
            spans.append((start, end))
            start = None
    return spans


class ReadoutContentProbe(nn.Module):
    def __init__(
        self,
        layers=16,
        query_heads=32,
        readout_dim=2560,
        hidden_dim=128,
        max_tokens=1024,
    ):
        super().__init__()
        self.layers = layers
        self.query_heads = query_heads
        self.readout_dim = readout_dim
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.readout_norm = nn.LayerNorm(readout_dim)
        self.readout_projection = nn.Linear(readout_dim, 64)
        self.layer_embedding = nn.Parameter(torch.zeros(layers, 64))
        self.global_projection = nn.Sequential(
            nn.Linear(layers * 64, 512),
            nn.GELU(),
            nn.Linear(512, hidden_dim),
        )
        self.attention_projection = nn.Sequential(
            nn.Linear(layers * query_heads, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )
        self.position_embedding = nn.Embedding(max_tokens, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(encoder_layer, num_layers=2)
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
        layer_state = self.readout_projection(self.readout_norm(readout.float()))
        layer_state = layer_state + self.layer_embedding
        global_state = self.global_projection(layer_state.reshape(1, -1))
        token_attention = attention.permute(2, 0, 1).reshape(
            tokens, self.layers * self.query_heads
        )
        token_state = self.attention_projection(token_attention.float())
        positions = self.position_embedding(
            torch.arange(tokens, device=attention.device)
        )
        token_state = token_state + positions + global_state
        mixed = self.token_mixer(
            token_state.unsqueeze(0),
            src_key_padding_mask=(~valid_mask.bool()).unsqueeze(0),
        )[0]
        return {
            "support_logits": self.support_head(mixed).squeeze(-1),
            "start_logits": self.start_head(mixed).squeeze(-1),
            "end_logits": self.end_head(mixed).squeeze(-1),
            "yesno_logits": self.yesno_head(global_state),
        }


def marginal_span_loss(start_logits, end_logits, spans):
    if not spans:
        return None
    start_log_probs = F.log_softmax(start_logits.float(), dim=-1)
    end_log_probs = F.log_softmax(end_logits.float(), dim=-1)
    scores = torch.stack(
        [start_log_probs[start] + end_log_probs[end] for start, end in spans]
    )
    return -torch.logsumexp(scores, dim=0)


def probe_loss(outputs, labels):
    valid = labels["valid_mask"].bool()
    support = labels["support_token_mask"].float()
    positives = support[valid].sum().clamp_min(1.0)
    negatives = valid.sum().float() - positives
    pos_weight = (negatives / positives).clamp(1.0, 20.0)
    support_loss = F.binary_cross_entropy_with_logits(
        outputs["support_logits"][valid],
        support[valid],
        pos_weight=pos_weight,
    )
    answer = str(labels["source_answer"]).strip().lower()
    span_loss = marginal_span_loss(
        outputs["start_logits"].masked_fill(~valid, -1e9),
        outputs["end_logits"].masked_fill(~valid, -1e9),
        labels["answer_spans"],
    )
    yesno_loss = None
    if answer in ("yes", "no"):
        target = torch.tensor(
            [0 if answer == "yes" else 1],
            dtype=torch.long,
            device=outputs["yesno_logits"].device,
        )
        yesno_loss = F.cross_entropy(outputs["yesno_logits"], target)
    active = [support_loss]
    if span_loss is not None:
        active.append(span_loss)
    if yesno_loss is not None:
        active.append(yesno_loss)
    return sum(active), {
        "support": support_loss,
        "span": span_loss,
        "yesno": yesno_loss,
    }


def payload_to_device(payload, condition, device, zero=False):
    state = payload["conditions"][condition]
    readout = state["readout"].float().to(device)
    attention = state["attention"].float().to(device)
    if zero:
        readout = torch.zeros_like(readout)
        attention = torch.zeros_like(attention)
    metadata = state["metadata"]
    labels = {
        "valid_mask": torch.as_tensor(
            metadata["valid_mask"], dtype=torch.bool, device=device
        ),
        "support_token_mask": torch.as_tensor(
            metadata["support_token_mask"], dtype=torch.bool, device=device
        ),
        "answer_spans": contiguous_spans(metadata["answer_token_mask"]),
        "source_id": state["source_id"],
        "source_answer": state["source_answer"],
    }
    return readout, attention, labels

