import torch
import torch.nn as nn
import torch.nn.functional as F

from p3e_b_common import SenderNativeHeadwiseCache


class PerCanonicalHeadResidual(nn.Module):
    def __init__(self, layers=16, heads=16, dimension=128, rank=32):
        super().__init__(); self.layers, self.heads, self.dimension, self.rank = layers, heads, dimension, rank
        self.down = nn.Parameter(torch.empty(layers, heads, rank, dimension))
        self.up = nn.Parameter(torch.zeros(layers, heads, dimension, rank))
        for layer in range(layers):
            for head in range(heads): nn.init.orthogonal_(self.down[layer, head])

    def forward(self, value):
        hidden = torch.einsum("ltcd,lcrd->ltcr", value.float(), self.down)
        update = torch.einsum("ltcr,lcdr->ltcd", F.silu(hidden), self.up)
        return value.float() + update


class HeadStructuredWriter(nn.Module):
    def __init__(self, layers=16, native_heads=8, canonical_heads=16, head_dim=128, rank=32, temperature=1.0):
        super().__init__(); self.layers, self.native_heads, self.canonical_heads = int(layers), int(native_heads), int(canonical_heads)
        self.head_dim, self.rank, self.temperature = int(head_dim), int(rank), float(temperature)
        logits = torch.full((self.layers, self.canonical_heads, self.native_heads), -6.0)
        for canonical_head in range(self.canonical_heads): logits[:, canonical_head, canonical_head // 2] = 0.0
        self.head_router_logits = nn.Parameter(logits)
        self.k_adapter = PerCanonicalHeadResidual(self.layers, self.canonical_heads, self.head_dim, self.rank)
        self.v_adapter = PerCanonicalHeadResidual(self.layers, self.canonical_heads, self.head_dim, self.rank)

    def routing_weights(self):
        soft = (self.head_router_logits.float() / self.temperature).softmax(dim=-1)
        indices = soft.argmax(dim=-1, keepdim=True); hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
        return hard.detach() - soft.detach() + soft if self.training else hard

    def forward(self, keys, values):
        if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != self.layers or keys.shape[-2:] != (self.native_heads, self.head_dim):
            raise RuntimeError(f"Expected Native K/V [16,T,8,128], got {tuple(keys.shape)}")
        route = self.routing_weights()
        mixed_keys = torch.einsum("lch,lthd->ltcd", route, keys.float())
        mixed_values = torch.einsum("lch,lthd->ltcd", route, values.float())
        return self.k_adapter(mixed_keys), self.v_adapter(mixed_values), route

    def metadata(self):
        return {"layers": self.layers, "native_heads": self.native_heads, "canonical_heads": self.canonical_heads,
                "head_dim": self.head_dim, "rank": self.rank, "routing": "shared_KV_16x8_straight_through_top1",
                "initialization": "exact_fixed_duplicate_writer16", "k_v_route_shared": True,
                "k_v_residual_adapters_separate": True, "cross_token_mixing": False, "cross_layer_mixing": False,
                "dimension_compression": False}


def native_payload_to(payload, device):
    keys, values = payload["keys"].float().to(device), payload["values"].float().to(device)
    valid = torch.as_tensor(payload["metadata"]["valid_mask"], dtype=torch.bool, device=device)
    support = torch.as_tensor(payload["metadata"]["support_token_mask"], dtype=torch.bool, device=device)
    return keys, values, valid, support


def writer_memory(writer, payload, device, oracle_support=False, no_grad=False):
    keys, values, valid, support = native_payload_to(payload, device)
    if no_grad:
        with torch.no_grad(): canonical_keys, canonical_values, route = writer(keys, values)
    else: canonical_keys, canonical_values, route = writer(keys, values)
    mask = valid & support if oracle_support else valid
    if not mask.any(): raise RuntimeError("Writer memory mask is empty")
    return {"keys": canonical_keys, "values": canonical_values, "mask": mask, "support_mask": support, "writer_route": route}


def head_diversity_loss(memory):
    mask = memory["mask"]
    losses = []
    for name in ("keys", "values"):
        pooled = memory[name][:, mask].mean(dim=1)
        normalized = F.normalize(pooled.float(), dim=-1)
        similarity = torch.einsum("lcd,led->lce", normalized, normalized)
        off_diagonal = ~torch.eye(similarity.shape[-1], dtype=torch.bool, device=similarity.device)[None]
        losses.append(similarity[off_diagonal.expand_as(similarity)].square().mean())
    return sum(losses) / len(losses)


def load_writer(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); metadata = checkpoint["writer_metadata"]
    writer = HeadStructuredWriter(metadata["layers"], metadata["native_heads"], metadata["canonical_heads"], metadata["head_dim"], metadata["rank"]).to(device)
    writer.load_state_dict(checkpoint["writer"]); return writer, checkpoint


__all__ = ["HeadStructuredWriter", "SenderNativeHeadwiseCache", "head_diversity_loss", "load_writer", "native_payload_to", "writer_memory"]
