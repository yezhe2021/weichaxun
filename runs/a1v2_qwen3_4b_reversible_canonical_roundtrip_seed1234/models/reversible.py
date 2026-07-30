import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_branch(x, norm, first, second):
    normalized = F.rms_norm(
        x,
        (x.shape[-1],),
        norm.weight.to(x.dtype),
        norm.eps,
    )
    return F.linear(
        F.silu(F.linear(normalized, first.weight.to(x.dtype), None)),
        second.weight.to(x.dtype),
        None,
    )


class BiasFreeMLP(nn.Module):
    def __init__(self, dimension, hidden):
        super().__init__()
        self.norm = nn.RMSNorm(dimension)
        self.first = nn.Linear(dimension, hidden, bias=False)
        self.second = nn.Linear(hidden, dimension, bias=False)

    def forward(self, x):
        return rms_branch(x, self.norm, self.first, self.second)


class StreamWriter(nn.Module):
    def __init__(self, cfg, scale, rotation):
        super().__init__()
        layers, heads, dim = (
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
        )
        self.register_buffer("scale", scale.float().clone())
        self.register_buffer("rotation", rotation.float().clone())
        self.head = nn.Parameter(torch.eye(heads).repeat(layers, 1, 1))
        self.trunk = BiasFreeMLP(dim, cfg["trunk_hidden_dim"])
        self.gamma = nn.Parameter(torch.zeros(layers))
        self.adapters = nn.ModuleList(
            [
                BiasFreeMLP(dim, cfg["adapter_rank"])
                for _ in range(layers)
            ]
        )
        self.eta = nn.Parameter(torch.zeros(layers))

    def forward(self, x):
        x = x.float() / self.scale[:, None, :, :]
        x = torch.einsum("loi,ltid->ltod", self.head, x)
        z = torch.matmul(x, self.rotation)
        trunk = self.trunk(z)
        z = z + self.gamma[:, None, None, None] * trunk
        adapter = torch.stack(
            [module(z[layer]) for layer, module in enumerate(self.adapters)]
        )
        return z + self.eta[:, None, None, None] * adapter


class StreamDecoder(nn.Module):
    def __init__(self, cfg, scale, rotation):
        super().__init__()
        layers, heads, dim = (
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
        )
        self.register_buffer("scale", scale.float().clone())
        self.register_buffer("rotation_t", rotation.float().T.contiguous())
        self.correction = BiasFreeMLP(dim, cfg["trunk_hidden_dim"])
        self.beta = nn.Parameter(torch.zeros(layers))
        self.head = nn.Parameter(torch.eye(heads).repeat(layers, 1, 1))

    def forward(self, canonical):
        z = canonical.float()
        z = z + self.beta[:, None, None, None] * self.correction(z)
        x = torch.matmul(z, self.rotation_t)
        x = torch.einsum("loi,ltid->ltod", self.head, x)
        return x * self.scale[:, None, :, :]


class ReversibleCanonical4B(nn.Module):
    def __init__(self, cfg, protocol):
        super().__init__()
        self.writer_k = StreamWriter(cfg, protocol["scale_k"], protocol["rotation_k"])
        self.writer_v = StreamWriter(cfg, protocol["scale_v"], protocol["rotation_v"])
        self.decoder_k = StreamDecoder(cfg, protocol["scale_k"], protocol["rotation_k"])
        self.decoder_v = StreamDecoder(cfg, protocol["scale_v"], protocol["rotation_v"])

    def write(self, key, value):
        return self.writer_k(key), self.writer_v(value)

    def decode(self, canonical_key, canonical_value):
        return self.decoder_k(canonical_key), self.decoder_v(canonical_value)

    def forward(self, key, value):
        return self.decode(*self.write(key, value))

    def writer_parameters(self):
        return list(self.writer_k.parameters()) + list(self.writer_v.parameters())

    def decoder_parameters(self):
        return list(self.decoder_k.parameters()) + list(self.decoder_v.parameters())

    def orthogonality_loss(self):
        values = []
        for head in (
            self.writer_k.head,
            self.writer_v.head,
            self.decoder_k.head,
            self.decoder_v.head,
        ):
            identity = torch.eye(head.shape[-1], device=head.device)
            gram = torch.matmul(head.transpose(-1, -2), head)
            values.append((gram - identity).square().mean())
        return torch.stack(values).sum()

    def gate_loss(self):
        gates = (
            self.writer_k.gamma,
            self.writer_v.gamma,
            self.writer_k.eta,
            self.writer_v.eta,
            self.decoder_k.beta,
            self.decoder_v.beta,
        )
        return torch.stack([gate.square().mean() for gate in gates]).sum()

    def zero_check(self, tokens=3):
        device = next(self.parameters()).device
        shape = (
            self.writer_k.scale.shape[0],
            tokens,
            self.writer_k.scale.shape[1],
            self.writer_k.scale.shape[2],
        )
        zero = torch.zeros(shape, device=device)
        canonical = self.write(zero, zero)
        decoded = self.decode(*canonical)
        return {
            "writer_k_max_abs": canonical[0].abs().max().item(),
            "writer_v_max_abs": canonical[1].abs().max().item(),
            "decoder_k_max_abs": decoded[0].abs().max().item(),
            "decoder_v_max_abs": decoded[1].abs().max().item(),
        }
