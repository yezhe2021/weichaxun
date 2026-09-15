import torch

from anchors import build_anchors
from offsets import TokenSpan
from translator import NativeKVTranslator


def main():
    module = NativeKVTranslator("full28_mlp", source_layers=2, target_layers=3,
                                heads=2, dim=4, hidden_dim=8)
    x = torch.randn(1, 2, 32, 2, 4)
    key, value = module(x, x)
    assert key.shape == value.shape == (1, 3, 32, 2, 4)
    zero = torch.zeros_like(x); zk, zv = module(zero, zero)
    assert torch.count_nonzero(zk) == 0 and torch.count_nonzero(zv) == 0
    assert all(layer.bias is None for layer in module.modules() if isinstance(layer, torch.nn.Linear))
    assert module.k_depth[0] is not module.k_depth[1]
    assert module.k_depth[0] is not module.v_depth[0]
    assert module.k_head[0][0] is not module.k_head[0][1]
    assert module.k_head[0][0] is not module.v_head[0][0]
    spans = [TokenSpan(0, 1, 0, 0)] + [TokenSpan(i + 1, i + 2, i, i + 1) for i in range(64)]
    importance = torch.arange(65, dtype=torch.float32)
    anchors = build_anchors("x" * 64, spans, spans, importance, 32)
    assert len(anchors) == 32
    assert all(anchor["source_index"] != 0 for anchor in anchors)
    assert [anchor["source_index"] for anchor in anchors] == sorted(anchor["source_index"] for anchor in anchors)
    print("All Full28-MLP zero/bias/shape/parameter-independence and ordered-region tests passed", flush=True)


if __name__ == "__main__": main()
