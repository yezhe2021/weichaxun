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


class KVFunctionalProbe(nn.Module):
    def __init__(self, layers, head_dim, latent_dim, classes):
        super().__init__()
        self.layers = int(layers)
        self.key_norm = nn.ModuleList(
            [nn.LayerNorm(head_dim, elementwise_affine=False) for _ in range(layers)]
        )
        self.value_norm = nn.ModuleList(
            [nn.LayerNorm(head_dim, elementwise_affine=False) for _ in range(layers)]
        )
        self.key_projection = nn.ModuleList(
            [nn.Linear(head_dim, latent_dim, bias=False) for _ in range(layers)]
        )
        self.value_projection = nn.ModuleList(
            [nn.Linear(head_dim, latent_dim, bias=False) for _ in range(layers)]
        )
        self.token_queries = nn.Parameter(torch.empty(layers, latent_dim))
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.classifier = SharedClassifier(latent_dim, classes)
        nn.init.normal_(self.token_queries, std=latent_dim**-0.5)

    def forward(self, memory):
        if len(memory["keys"]) != self.layers:
            raise ValueError(f"Expected {self.layers} layers, got {len(memory['keys'])}")
        layer_states = []
        for layer, (key, value) in enumerate(zip(memory["keys"], memory["values"])):
            key = self.key_projection[layer](self.key_norm[layer](key.float()))
            value = self.value_projection[layer](self.value_norm[layer](value.float()))
            scores = torch.einsum("htr,r->ht", key, self.token_queries[layer])
            weights = (scores.reshape(-1) / self.token_queries.shape[-1] ** 0.5).softmax(dim=0)
            pooled = torch.einsum("n,nr->r", weights, value.reshape(-1, value.shape[-1]))
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
    "VectorStateProbe",
]
