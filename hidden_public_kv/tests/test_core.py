import torch
import torch.nn as nn

from hidden_public_kv.core import PublicReader, PublicWriter


class Layer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)


class Backbone(nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        self.layers = nn.ModuleList([Layer(width) for _ in range(layers)])

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden_states=hidden)
        return hidden


class Model(nn.Module):
    def __init__(self, width=16, layers=4):
        super().__init__()
        self.model = Backbone(width, layers)

    def forward(self, hidden):
        return self.model(hidden)


def test_writer_shapes_and_detach_guard():
    writer = PublicWriter(16, layers=2, kv_heads=2, head_dim=4)
    taps = [torch.randn(1, 5, 16), torch.randn(1, 5, 16)]
    memory = writer(taps, torch.ones(1, 5, dtype=torch.bool))
    assert memory.keys[0].shape == (1, 2, 5, 4)
    bad = [torch.randn(1, 5, 16, requires_grad=True), taps[1]]
    try:
        writer(bad, torch.ones(1, 5, dtype=torch.bool))
        raise AssertionError("detach guard did not fire")
    except RuntimeError:
        pass


def test_reader_zero_and_reader_off_match():
    torch.manual_seed(7)
    model = Model()
    writer = PublicWriter(16, layers=2, kv_heads=2, head_dim=4)
    reader = PublicReader(16, active_layers=[1, 3], query_heads=4, kv_heads=2, head_dim=4)
    hidden = torch.randn(1, 3, 16)
    taps = [torch.randn(1, 5, 16), torch.randn(1, 5, 16)]
    memory = writer(taps, torch.ones(1, 5, dtype=torch.bool))
    baseline = model(hidden)
    with reader.inject(model, memory.zero()):
        zero = model(hidden)
    assert torch.allclose(baseline, zero, atol=1e-6)
    with reader.inject(model, memory):
        public = model(hidden)
    assert not torch.allclose(baseline, public)


if __name__ == "__main__":
    test_writer_shapes_and_detach_guard()
    test_reader_zero_and_reader_off_match()
    print("core tests passed")
