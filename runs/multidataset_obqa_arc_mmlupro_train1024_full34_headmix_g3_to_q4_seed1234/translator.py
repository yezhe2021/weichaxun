import torch
from torch import nn
import torch.nn.functional as F


class NativeKVTranslator(nn.Module):
    """Full34 Gemma3-4B [34,T,4,256] -> Qwen3-4B [36,T,8,128]."""

    def __init__(self, architecture="full34_headmix256", source_layers=34, target_layers=36,
                 source_heads=4, target_heads=8, source_dim=256, target_dim=128,
                 hidden_dim=1024, depth_output_dim=256, head_mapping="full_head"):
        super().__init__()
        if architecture != "full34_headmix256":
            raise ValueError(architecture)
        if head_mapping not in ("fixed_split", "block_head", "full_head"):
            raise ValueError(head_mapping)
        if source_heads * depth_output_dim != target_heads * target_dim:
            raise ValueError("Head mapper requires equal flattened source/target widths")
        self.architecture = architecture
        self.source_layers, self.target_layers = source_layers, target_layers
        self.source_heads, self.target_heads = source_heads, target_heads
        self.source_dim, self.target_dim = source_dim, target_dim
        self.hidden_dim, self.depth_output_dim = hidden_dim, depth_output_dim
        self.head_mapping = head_mapping
        width = source_layers * source_dim

        def depth_mlp():
            return nn.Sequential(
                nn.Linear(width, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, depth_output_dim, bias=False),
            )

        self.k_depth = nn.ModuleList([depth_mlp() for _ in range(target_layers)])
        self.v_depth = nn.ModuleList([depth_mlp() for _ in range(target_layers)])
        flat_width = source_heads * depth_output_dim
        if head_mapping == "full_head":
            self.k_head = nn.ModuleList([nn.Linear(flat_width, flat_width, bias=False)
                                         for _ in range(target_layers)])
            self.v_head = nn.ModuleList([nn.Linear(flat_width, flat_width, bias=False)
                                         for _ in range(target_layers)])
        elif head_mapping == "block_head":
            self.k_head = nn.ModuleList([nn.ModuleList([
                nn.Linear(depth_output_dim, 2 * target_dim, bias=False) for _ in range(source_heads)
            ]) for _ in range(target_layers)])
            self.v_head = nn.ModuleList([nn.ModuleList([
                nn.Linear(depth_output_dim, 2 * target_dim, bias=False) for _ in range(source_heads)
            ]) for _ in range(target_layers)])
        else:
            self.k_head = nn.ModuleList([nn.Identity() for _ in range(target_layers)])
            self.v_head = nn.ModuleList([nn.Identity() for _ in range(target_layers)])

    def _depth(self, tensor, mappings):
        x = tensor.permute(0, 2, 3, 1, 4).flatten(-2)
        return torch.stack([mapping(x) for mapping in mappings], dim=1)

    def _heads(self, tensor, mappings):
        outputs = []
        for layer in range(self.target_layers):
            current = tensor[:, layer]
            if self.head_mapping == "block_head":
                current = torch.cat([
                    mappings[layer][head](current[:, :, head])
                    for head in range(self.source_heads)
                ], dim=-1)
            else:
                current = mappings[layer](current.flatten(-2))
            outputs.append(current.reshape(current.shape[0], current.shape[1],
                                           self.target_heads, self.target_dim))
        return torch.stack(outputs, dim=1)

    def forward(self, key, value):
        expected = (self.source_layers, self.source_heads, self.source_dim)
        if (key.shape != value.shape or key.ndim != 5 or key.shape[1] != self.source_layers
                or tuple(key.shape[3:]) != (self.source_heads, self.source_dim)):
            raise ValueError(f"Expected [batch,{expected[0]},tokens,{expected[1]},{expected[2]}]")
        return (
            self._heads(self._depth(key, self.k_depth), self.k_head),
            self._heads(self._depth(value, self.v_depth), self.v_head),
        )

    @torch.no_grad()
    def correspondence(self):
        if self.head_mapping != "full_head":
            raise RuntimeError("Correspondence statistics require full_head")
        head = {}
        for name, mappings in (("k", self.k_head), ("v", self.v_head)):
            head[name] = torch.stack([
                module.weight.float().reshape(self.target_heads, self.target_dim,
                                              self.source_heads, self.depth_output_dim)
                .square().sum(dim=(1, 3)).sqrt()
                for module in mappings
            ]).cpu()
        depth = {}
        for name, mappings in (("k", self.k_depth), ("v", self.v_depth)):
            depth[name] = torch.stack([
                module[0].weight.float().reshape(self.hidden_dim, self.source_layers, self.source_dim)
                .square().sum(dim=(0, 2)).sqrt()
                for module in mappings
            ]).cpu()
        return head, depth


class ResidualKVAdapter(nn.Module):
    def __init__(self, layers=36, heads=8, dim=128, rank=64):
        super().__init__()
        self.layers, self.heads, self.dim, self.rank = layers, heads, dim, rank

        def branch():
            module = nn.Sequential(
                nn.Linear(dim, rank, bias=False), nn.GELU(), nn.Linear(rank, dim, bias=False))
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
        expected = (self.layers, self.heads, self.dim)
        if (key.shape != value.shape or key.ndim != 5 or key.shape[1] != self.layers
                or tuple(key.shape[3:]) != (self.heads, self.dim)):
            raise ValueError(f"Expected [batch,{expected[0]},tokens,{expected[1]},{expected[2]}]")
        delta_k = self._map_branches(key, self.k)
        delta_v = self._map_branches(value, self.v)
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
