from pathlib import Path

import torch

from translator import NativeKVTranslator


def main():
    module = NativeKVTranslator("full28_diagonal")
    source = torch.randn(2, 28, 32, 8, 128)
    key, value = module(source, source)
    assert key.shape == value.shape == (2, 36, 32, 8, 128)
    zero = torch.zeros_like(source)
    zk, zv = module(zero, zero)
    assert torch.count_nonzero(zk) == 0 and torch.count_nonzero(zv) == 0
    assert all(layer.bias is None for layer in module.modules() if isinstance(layer, torch.nn.Linear))
    data_source = (Path(__file__).parent / "data.py").read_text(encoding="utf-8")
    assert 'raw["target_k"][:, :1]' in data_source and 'raw["target_v"][:, :1]' in data_source
    assert 'raw["target_k"][:, 1:]' not in data_source and 'raw["target_v"][:, 1:]' not in data_source
    print("NoAlign architecture and target-KV isolation tests passed", flush=True)


if __name__ == "__main__": main()
