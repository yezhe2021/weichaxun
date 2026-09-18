import torch
from torch import nn


class DepthHeadTranslator(nn.Module):
    def __init__(self, source_layers, target_layers, source_heads, target_heads,
                 source_dim, target_dim, hidden_dim, depth_output_dim,
                 head_mode="per_head"):
        super().__init__()
        self.source_layers, self.target_layers = source_layers, target_layers
        self.source_heads, self.target_heads = source_heads, target_heads
        self.source_dim, self.target_dim = source_dim, target_dim
        self.depth_output_dim, self.head_mode = depth_output_dim, head_mode
        width = source_layers * source_dim

        def depth_branch():
            return nn.Sequential(nn.Linear(width, hidden_dim, bias=False), nn.GELU(),
                                 nn.Linear(hidden_dim, depth_output_dim, bias=False))

        self.k_depth = nn.ModuleList([depth_branch() for _ in range(target_layers)])
        self.v_depth = nn.ModuleList([depth_branch() for _ in range(target_layers)])
        if head_mode == "per_head":
            if source_heads != target_heads or depth_output_dim != target_dim:
                raise ValueError("per_head requires matching head geometry")
            self.k_head = nn.ModuleList([nn.ModuleList([
                nn.Linear(target_dim, target_dim, bias=False) for _ in range(target_heads)
            ]) for _ in range(target_layers)])
            self.v_head = nn.ModuleList([nn.ModuleList([
                nn.Linear(target_dim, target_dim, bias=False) for _ in range(target_heads)
            ]) for _ in range(target_layers)])
        elif head_mode == "full_head":
            source_width = source_heads * depth_output_dim
            target_width = target_heads * target_dim
            if source_width != target_width:
                raise ValueError("full_head requires equal flattened widths")
            self.k_head = nn.ModuleList([nn.Linear(source_width, target_width, bias=False)
                                         for _ in range(target_layers)])
            self.v_head = nn.ModuleList([nn.Linear(source_width, target_width, bias=False)
                                         for _ in range(target_layers)])
        else:
            raise ValueError(head_mode)

    def _depth(self, tensor, mappings):
        x = tensor.permute(0, 2, 3, 1, 4).flatten(-2)
        return torch.stack([mapping(x) for mapping in mappings], dim=1)

    def _heads(self, tensor, mappings):
        if self.head_mode == "per_head":
            return torch.stack([torch.stack([
                mappings[layer][head](tensor[:, layer, :, head])
                for head in range(self.target_heads)], dim=2)
                for layer in range(self.target_layers)], dim=1)
        outputs = []
        for layer in range(self.target_layers):
            current = mappings[layer](tensor[:, layer].flatten(-2))
            outputs.append(current.reshape(current.shape[0], current.shape[1],
                                           self.target_heads, self.target_dim))
        return torch.stack(outputs, dim=1)

    def forward(self, key, value):
        return (self._heads(self._depth(key, self.k_depth), self.k_head),
                self._heads(self._depth(value, self.v_depth), self.v_head))


def llama_to_qwen():
    return DepthHeadTranslator(28, 36, 8, 8, 128, 128, 1024, 128, "per_head")


def qwen_to_gemma():
    return DepthHeadTranslator(36, 34, 8, 4, 128, 256, 1024, 128, "full_head")


def gemma_to_qwen():
    return DepthHeadTranslator(34, 36, 4, 8, 256, 128, 1024, 256, "full_head")


def qwen_to_llama():
    return DepthHeadTranslator(36, 28, 8, 8, 128, 128, 1024, 128, "per_head")


class ResidualAdapter(nn.Module):
    def __init__(self, layers, heads, dim, rank=64):
        super().__init__()
        self.layers, self.heads = layers, heads

        def branch():
            module = nn.Sequential(nn.Linear(dim, rank, bias=False), nn.GELU(),
                                   nn.Linear(rank, dim, bias=False))
            nn.init.zeros_(module[-1].weight)
            return module

        self.k = nn.ModuleList([nn.ModuleList([branch() for _ in range(heads)])
                                for _ in range(layers)])
        self.v = nn.ModuleList([nn.ModuleList([branch() for _ in range(heads)])
                                for _ in range(layers)])

    def _map(self, tensor, mappings):
        return torch.stack([torch.stack([
            mappings[layer][head](tensor[:, layer, :, head])
            for head in range(self.heads)], dim=2)
            for layer in range(self.layers)], dim=1)

    def forward(self, key, value):
        return key + self._map(key, self.k), value + self._map(value, self.v)
