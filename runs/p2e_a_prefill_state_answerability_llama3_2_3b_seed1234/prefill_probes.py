import torch
import torch.nn as nn
import torch.nn.functional as F


class EndLinearProbe(nn.Module):
    def __init__(self, hidden_size, classes):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, states, mask=None):
        return self.classifier(self.norm(states))


class SummaryLinearProbe(nn.Module):
    def __init__(self, hidden_size, classes):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, states, mask=None):
        if mask is None:
            pooled = states.mean(dim=1)
        else:
            weights = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = torch.einsum("bt,btd->bd", weights, states)
        return self.classifier(self.norm(pooled))


class AttentionPoolProbe(nn.Module):
    def __init__(self, hidden_size, classes, attention_rank=128, value_rank=256):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.key = nn.Linear(hidden_size, attention_rank, bias=False)
        self.value = nn.Linear(hidden_size, value_rank, bias=False)
        self.query = nn.Parameter(torch.empty(attention_rank))
        self.output = nn.Sequential(
            nn.LayerNorm(value_rank),
            nn.Linear(value_rank, value_rank),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(value_rank, classes),
        )
        nn.init.normal_(self.query, std=attention_rank**-0.5)

    def forward(self, states, mask=None):
        states = self.input_norm(states)
        scores = torch.einsum("btr,r->bt", self.key(states), self.query)
        scores = scores / self.query.numel() ** 0.5
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        pooled = torch.einsum("bt,btv->bv", weights, self.value(states))
        return self.output(pooled)


def build_probe(config, hidden_size, classes, attention_rank=128, value_rank=256):
    kind = config["kind"]
    if kind == "end_linear":
        return EndLinearProbe(hidden_size, classes)
    if kind == "summary_linear":
        return SummaryLinearProbe(hidden_size, classes)
    if kind in {"summary_attention", "raw_evidence_attention"}:
        return AttentionPoolProbe(hidden_size, classes, attention_rank, value_rank)
    raise ValueError(f"Unknown probe kind: {kind}")


__all__ = [
    "AttentionPoolProbe",
    "EndLinearProbe",
    "SummaryLinearProbe",
    "build_probe",
]
