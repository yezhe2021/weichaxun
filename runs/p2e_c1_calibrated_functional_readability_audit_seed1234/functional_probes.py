import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedClassifier(nn.Module):
    def __init__(self, latent_dim, classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, classes),
        )

    def forward(self, state):
        return self.network(state)


class HiddenTokenProbe(nn.Module):
    def __init__(self, input_dim, latent_dim, classes):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.key = nn.Linear(input_dim, latent_dim, bias=False)
        self.value = nn.Linear(input_dim, latent_dim, bias=False)
        self.query = nn.Parameter(torch.empty(latent_dim))
        self.classifier = SharedClassifier(latent_dim, classes)
        nn.init.normal_(self.query, std=latent_dim**-0.5)

    def forward(self, hidden):
        hidden = self.norm(hidden.float())
        key = self.key(hidden)
        value = self.value(hidden)
        weights = (key @ self.query / self.query.numel() ** 0.5).softmax(dim=0)
        return self.classifier(torch.einsum("t,tr->r", weights, value))


class ReusedAttentionPoolProbe(nn.Module):
    """Exact Experiment-A probe architecture, with unbatched inference support."""

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
        unbatched = states.ndim == 2
        if unbatched:
            states = states.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        states = self.input_norm(states.float())
        scores = torch.einsum("btr,r->bt", self.key(states), self.query)
        scores = scores / self.query.numel() ** 0.5
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        pooled = torch.einsum("bt,btv->bv", weights, self.value(states))
        logits = self.output(pooled)
        return logits[0] if unbatched else logits


class KVFunctionalProbe(nn.Module):
    def __init__(
        self,
        layers,
        heads,
        head_dim,
        key_rank,
        value_rank,
        classes,
    ):
        super().__init__()
        self.layers = int(layers)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        token_dim = self.heads * self.head_dim
        self.key_norm = nn.ModuleList(
            [nn.LayerNorm(token_dim, elementwise_affine=False) for _ in range(layers)]
        )
        self.value_norm = nn.ModuleList(
            [nn.LayerNorm(token_dim, elementwise_affine=False) for _ in range(layers)]
        )
        self.key_projection = nn.ModuleList(
            [nn.Linear(token_dim, key_rank, bias=False) for _ in range(layers)]
        )
        self.value_projection = nn.ModuleList(
            [nn.Linear(token_dim, value_rank, bias=False) for _ in range(layers)]
        )
        self.token_queries = nn.Parameter(torch.empty(layers, key_rank))
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.classifier = SharedClassifier(value_rank, classes)
        nn.init.normal_(self.token_queries, std=key_rank**-0.5)

    def token_matrix(self, tensor):
        if tensor.ndim != 3:
            raise ValueError(f"Expected [heads, tokens, dim], got {tuple(tensor.shape)}")
        if tensor.shape[0] != self.heads or tensor.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected heads/dim {self.heads}/{self.head_dim}, got "
                f"{tensor.shape[0]}/{tensor.shape[-1]}"
            )
        return tensor.float().permute(1, 0, 2).reshape(tensor.shape[1], -1)

    def forward(self, memory):
        if len(memory["keys"]) != self.layers:
            raise ValueError(f"Expected {self.layers} layers, got {len(memory['keys'])}")
        layer_states = []
        for layer, (key, value) in enumerate(zip(memory["keys"], memory["values"])):
            if key.shape[1] != value.shape[1]:
                raise ValueError("K/V token counts differ")
            key = self.key_projection[layer](
                self.key_norm[layer](self.token_matrix(key))
            )
            value = self.value_projection[layer](
                self.value_norm[layer](self.token_matrix(value))
            )
            scores = torch.einsum("tr,r->t", key, self.token_queries[layer])
            weights = (scores / self.token_queries.shape[-1] ** 0.5).softmax(dim=0)
            pooled = torch.einsum("t,tv->v", weights, value)
            layer_states.append(pooled)
        layer_weights = self.layer_logits.softmax(dim=0)
        state = torch.einsum("l,lr->r", layer_weights, torch.stack(layer_states))
        return self.classifier(state)


class LayerStateProbe(nn.Module):
    def __init__(self, input_dim, latent_dim, classes):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.projection = nn.Linear(input_dim, latent_dim, bias=False)
        self.layer_query = nn.Parameter(torch.empty(latent_dim))
        self.classifier = SharedClassifier(latent_dim, classes)
        nn.init.normal_(self.layer_query, std=latent_dim**-0.5)

    def forward(self, states):
        states = self.projection(self.norm(states.float()))
        weights = (states @ self.layer_query / self.layer_query.numel() ** 0.5).softmax(dim=0)
        return self.classifier(torch.einsum("l,lr->r", weights, states))


class VectorStateProbe(nn.Module):
    def __init__(self, input_dim, latent_dim, classes):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.projection = nn.Linear(input_dim, latent_dim, bias=False)
        self.classifier = SharedClassifier(latent_dim, classes)

    def forward(self, state):
        return self.classifier(self.projection(self.norm(state.float())))


__all__ = [
    "HiddenTokenProbe",
    "KVFunctionalProbe",
    "LayerStateProbe",
    "ReusedAttentionPoolProbe",
    "VectorStateProbe",
]
