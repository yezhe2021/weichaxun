import torch
from torch import nn
import torch.nn.functional as F


def depth_indices(source_layers=28, target_layers=36, window=1):
    if window not in (1, 5): raise ValueError("Only nearest and Local5 are supported")
    radius = window // 2; result = []
    for layer in range(target_layers):
        center = round(layer / (target_layers - 1) * (source_layers - 1))
        result.append([min(max(center + delta, 0), source_layers - 1)
                       for delta in range(-radius, radius + 1)])
    return result


class NativeKVTranslator(nn.Module):
    """Independent bias-free K/V map per target layer; heads/entries never mix."""

    def __init__(self, window, source_layers=28, target_layers=36, dim=128):
        super().__init__(); self.window, self.dim = window, dim
        self.indices = depth_indices(source_layers, target_layers, window)
        self.k_maps = nn.ModuleList([nn.Linear(window * dim, dim, bias=False) for _ in self.indices])
        self.v_maps = nn.ModuleList([nn.Linear(window * dim, dim, bias=False) for _ in self.indices])

    def project(self, tensor, maps):
        # tensor: [batch,source_layer,entry,head,dim]
        outputs = []
        for indices, mapping in zip(self.indices, maps):
            x = tensor[:, indices].permute(0, 2, 3, 1, 4).flatten(-2)
            outputs.append(mapping(x))
        return torch.stack(outputs, 1)

    def forward(self, key, value):
        if key.shape != value.shape or key.ndim != 5 or key.shape[1] != 28:
            raise ValueError("Expected paired Llama KV [batch,28,33,8,128]")
        return self.project(key, self.k_maps), self.project(value, self.v_maps)


def direct_nearest(key, value):
    indices = torch.tensor([x[0] for x in depth_indices(window=1)], device=key.device)
    return key[:, indices], value[:, indices]


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
