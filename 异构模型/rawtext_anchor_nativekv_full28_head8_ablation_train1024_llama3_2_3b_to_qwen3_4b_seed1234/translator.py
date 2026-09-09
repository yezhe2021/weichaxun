import torch
from torch import nn
import torch.nn.functional as F


def depth_indices(source_layers=28, target_layers=36, window=5):
    radius = window // 2
    result = []
    for layer in range(target_layers):
        center = round(layer / (target_layers - 1) * (source_layers - 1))
        result.append([min(max(center + delta, 0), source_layers - 1)
                       for delta in range(-radius, radius + 1)])
    return result


class NativeKVTranslator(nn.Module):
    """Bias-free, token-preserving translators. K and V are always independent."""

    def __init__(self, architecture, source_layers=28, target_layers=36, heads=8, dim=128):
        super().__init__()
        if architecture not in {"local5_samehead", "full28_samehead", "full28_diagonal", "full28_head8"}:
            raise ValueError(architecture)
        self.architecture = architecture
        self.source_layers, self.target_layers, self.heads, self.dim = source_layers, target_layers, heads, dim
        self.indices = depth_indices(source_layers, target_layers, 5) if architecture == "local5_samehead" else [list(range(source_layers)) for _ in range(target_layers)]
        width = (5 if architecture == "local5_samehead" else source_layers) * dim
        self.k_depth = nn.ModuleList([nn.Linear(width, dim, bias=False) for _ in range(target_layers)])
        self.v_depth = nn.ModuleList([nn.Linear(width, dim, bias=False) for _ in range(target_layers)])
        if architecture == "full28_diagonal":
            self.k_head = nn.ModuleList([nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(heads)]) for _ in range(target_layers)])
            self.v_head = nn.ModuleList([nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(heads)]) for _ in range(target_layers)])
        elif architecture == "full28_head8":
            self.k_head = nn.ModuleList([nn.ModuleList([nn.Linear(heads * dim, dim, bias=False) for _ in range(heads)]) for _ in range(target_layers)])
            self.v_head = nn.ModuleList([nn.ModuleList([nn.Linear(heads * dim, dim, bias=False) for _ in range(heads)]) for _ in range(target_layers)])

    def _depth(self, tensor, maps):
        outputs = []
        for indices, mapping in zip(self.indices, maps):
            x = tensor[:, indices].permute(0, 2, 3, 1, 4).flatten(-2)
            outputs.append(mapping(x))
        return torch.stack(outputs, 1)

    def _heads(self, z, maps):
        if self.architecture == "full28_diagonal":
            return torch.stack([torch.stack([maps[layer][head](z[:, layer, :, head])
                                              for head in range(self.heads)], dim=2)
                                for layer in range(self.target_layers)], dim=1)
        flattened = z.flatten(-2)
        return torch.stack([torch.stack([maps[layer][head](flattened[:, layer])
                                          for head in range(self.heads)], dim=2)
                            for layer in range(self.target_layers)], dim=1)

    def forward(self, key, value):
        expected = (self.source_layers, 32, self.heads, self.dim)
        if key.shape != value.shape or key.ndim != 5 or tuple(key.shape[1:]) != expected:
            raise ValueError(f"Expected paired content KV [batch,{','.join(map(str, expected))}]")
        k, v = self._depth(key, self.k_depth), self._depth(value, self.v_depth)
        if hasattr(self, "k_head"):
            k, v = self._heads(k, self.k_head), self._heads(v, self.v_head)
        return k, v


def component_loss(prediction, target):
    prediction, target = prediction.float(), target.float()
    nmse = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)
    cosine_error = 1 - F.cosine_similarity(prediction, target, dim=-1).mean()
    return nmse + cosine_error, nmse, 1 - cosine_error


def representation_loss(pred_k, pred_v, target_k, target_v):
    k_loss, k_nmse, k_cosine = component_loss(pred_k, target_k)
    v_loss, v_nmse, v_cosine = component_loss(pred_v, target_v)
    return k_loss + v_loss, {"k_nmse": k_nmse, "v_nmse": v_nmse,
                             "k_cosine": k_cosine, "v_cosine": v_cosine}
