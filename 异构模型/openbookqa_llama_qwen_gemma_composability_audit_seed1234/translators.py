import torch
from torch import nn


class LlamaToQwen(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        width = 28 * 128

        def mlp():
            return nn.Sequential(nn.Linear(width, hidden_dim, bias=False), nn.GELU(),
                                 nn.Linear(hidden_dim, 128, bias=False))

        self.k_depth = nn.ModuleList([mlp() for _ in range(36)])
        self.v_depth = nn.ModuleList([mlp() for _ in range(36)])
        self.k_head = nn.ModuleList([nn.ModuleList([nn.Linear(128, 128, bias=False) for _ in range(8)])
                                     for _ in range(36)])
        self.v_head = nn.ModuleList([nn.ModuleList([nn.Linear(128, 128, bias=False) for _ in range(8)])
                                     for _ in range(36)])

    def _depth(self, tensor, mappings):
        x = tensor.permute(0, 2, 3, 1, 4).flatten(-2)
        return torch.stack([mapping(x) for mapping in mappings], dim=1)

    def _heads(self, tensor, mappings):
        return torch.stack([torch.stack([mappings[layer][head](tensor[:, layer, :, head])
                                         for head in range(8)], dim=2)
                            for layer in range(36)], dim=1)

    def forward(self, key, value):
        return self._heads(self._depth(key, self.k_depth), self.k_head), \
               self._heads(self._depth(value, self.v_depth), self.v_head)


class QwenToGemma(nn.Module):
    def __init__(self, hidden_dim=1024, depth_output_dim=128):
        super().__init__()
        width = 36 * 128

        def depth_mlp():
            return nn.Sequential(nn.Linear(width, hidden_dim, bias=False), nn.GELU(),
                                 nn.Linear(hidden_dim, depth_output_dim, bias=False))

        self.k_depth = nn.ModuleList([depth_mlp() for _ in range(34)])
        self.v_depth = nn.ModuleList([depth_mlp() for _ in range(34)])
        flat = 8 * depth_output_dim
        self.k_head = nn.ModuleList([nn.Linear(flat, flat, bias=False) for _ in range(34)])
        self.v_head = nn.ModuleList([nn.Linear(flat, flat, bias=False) for _ in range(34)])

    def _depth(self, tensor, mappings):
        x = tensor.permute(0, 2, 3, 1, 4).flatten(-2)
        return torch.stack([mapping(x) for mapping in mappings], dim=1)

    def _heads(self, tensor, mappings):
        outputs = []
        for layer in range(34):
            current = mappings[layer](tensor[:, layer].flatten(-2))
            outputs.append(current.reshape(current.shape[0], current.shape[1], 4, 256))
        return torch.stack(outputs, dim=1)

    def forward(self, key, value):
        return self._heads(self._depth(key, self.k_depth), self.k_head), \
               self._heads(self._depth(value, self.v_depth), self.v_head)


class ResidualAdapter(nn.Module):
    def __init__(self, layers, heads, dim, rank=64):
        super().__init__()
        self.layers, self.heads, self.dim = layers, heads, dim

        def branch():
            module = nn.Sequential(nn.Linear(dim, rank, bias=False), nn.GELU(),
                                   nn.Linear(rank, dim, bias=False))
            nn.init.zeros_(module[-1].weight)
            return module

        self.k = nn.ModuleList([nn.ModuleList([branch() for _ in range(heads)]) for _ in range(layers)])
        self.v = nn.ModuleList([nn.ModuleList([branch() for _ in range(heads)]) for _ in range(layers)])

    def _map(self, tensor, mappings):
        return torch.stack([torch.stack([mappings[layer][head](tensor[:, layer, :, head])
                                         for head in range(self.heads)], dim=2)
                            for layer in range(self.layers)], dim=1)

    def forward(self, key, value):
        delta_k, delta_v = self._map(key, self.k), self._map(value, self.v)
        return key + delta_k, value + delta_v
