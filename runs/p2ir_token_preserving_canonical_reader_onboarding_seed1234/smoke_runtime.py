import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn

from p2ir_reader import TokenCanonicalReader


class FakeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, hidden_states, **kwargs):
        return (hidden_states + self.linear(hidden_states),)


class FakeBase(nn.Module):
    def __init__(self, dim, layers):
        super().__init__(); self.layers = nn.ModuleList([FakeLayer(dim) for _ in range(layers)])


class FakeModel(nn.Module):
    def __init__(self, dim=32, layers=3):
        super().__init__(); self.model = FakeBase(dim, layers); self.config = SimpleNamespace(hidden_size=dim)
    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = layer(hidden_states=hidden)[0]
        return hidden


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    device = torch.device(args.device); model = FakeModel().to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    reader = TokenCanonicalReader(model, canonical_dim=256, rank=16, active_layers=[0, 1, 2]).to(device)
    memory = {
        "keys": torch.randn(11, 256, device=device), "values": torch.randn(11, 256, device=device),
        "mask": torch.tensor([1] * 9 + [0, 0], dtype=torch.bool, device=device),
        "answer_token_mask": torch.tensor([0] * 7 + [1, 1] + [0, 0], dtype=torch.bool, device=device),
    }
    hidden = torch.randn(1, 5, 32, device=device)
    with reader.inject(model, memory): output = model(hidden)
    loss = output.square().mean(); loss.backward()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen fake backbone received gradients")
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in reader.parameters()):
        raise RuntimeError("Reader gradient smoke failed")
    print("P2-I-R runtime smoke passed")


if __name__ == "__main__":
    main()
