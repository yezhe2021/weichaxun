import torch
from torch import nn
import torch.nn.functional as F


class NativeKVTranslator(nn.Module):
    """Full28 MLP with diagonal head adapters; tokens/heads never mix and K/V are separate."""

    def __init__(self, architecture="full28_mlp", source_layers=28, target_layers=36,
                 heads=8, dim=128, hidden_dim=1024):
        super().__init__()
        if architecture != "full28_mlp": raise ValueError(architecture)
        self.architecture = architecture
        self.source_layers, self.target_layers = source_layers, target_layers
        self.heads, self.dim, self.hidden_dim = heads, dim, hidden_dim
        width = source_layers * dim
        def mlp():
            return nn.Sequential(
                nn.Linear(width, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, dim, bias=False),
            )
        self.k_depth = nn.ModuleList([mlp() for _ in range(target_layers)])
        self.v_depth = nn.ModuleList([mlp() for _ in range(target_layers)])
        self.k_head = nn.ModuleList([nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(heads)])
                                     for _ in range(target_layers)])
        self.v_head = nn.ModuleList([nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(heads)])
                                     for _ in range(target_layers)])

    def _depth(self, tensor, mappings):
        x = tensor.permute(0, 2, 3, 1, 4).flatten(-2)
        return torch.stack([mapping(x) for mapping in mappings], 1)

    def _heads(self, tensor, mappings):
        return torch.stack([torch.stack([mappings[layer][head](tensor[:, layer, :, head])
                                         for head in range(self.heads)], 2)
                            for layer in range(self.target_layers)], 1)

    def forward(self, key, value):
        expected = (self.source_layers, 32, self.heads, self.dim)
        if key.shape != value.shape or key.ndim != 5 or tuple(key.shape[1:]) != expected:
            raise ValueError(f"Expected [batch,{','.join(map(str, expected))}]")
        return (self._heads(self._depth(key, self.k_depth), self.k_head),
                self._heads(self._depth(value, self.v_depth), self.v_head))


class ResidualKVAdapter(nn.Module):
    """Independent token-wise receiver-space adapters for every target layer/head and K/V."""

    def __init__(self, layers=36, heads=8, dim=128, rank=64):
        super().__init__()
        self.layers, self.heads, self.dim, self.rank = layers, heads, dim, rank

        def branch():
            module = nn.Sequential(
                nn.Linear(dim, rank, bias=False),
                nn.GELU(),
                nn.Linear(rank, dim, bias=False),
            )
            nn.init.zeros_(module[-1].weight)
            return module

        self.k = nn.ModuleList([nn.ModuleList([branch() for _ in range(heads)]) for _ in range(layers)])
        self.v = nn.ModuleList([nn.ModuleList([branch() for _ in range(heads)]) for _ in range(layers)])

    def _map_branches(self, tensor, mappings):
        return torch.stack([
            torch.stack([mappings[layer][head](tensor[:, layer, :, head])
                         for head in range(self.heads)], dim=2)
            for layer in range(self.layers)], dim=1)

    def forward(self, key, value):
        expected = (self.layers, 32, self.heads, self.dim)
        if key.shape != value.shape or key.ndim != 5 or tuple(key.shape[1:]) != expected:
            raise ValueError(f"Expected [batch,{','.join(map(str, expected))}]")
        delta_k, delta_v = self._map_branches(key, self.k), self._map_branches(value, self.v)
        return key + delta_k, value + delta_v, delta_k, delta_v


def component_loss(prediction, target):
    prediction, target = prediction.float(), target.float()
    nmse = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)
    cosine = F.cosine_similarity(prediction, target, dim=-1).mean()
    return nmse + 1 - cosine, nmse, cosine


def representation_loss(pred_k, pred_v, target_k, target_v):
    kl, kn, kc = component_loss(pred_k, target_k)
    vl, vn, vc = component_loss(pred_v, target_v)
    return kl + vl, {"k_nmse": kn, "v_nmse": vn, "k_cosine": kc, "v_cosine": vc}
