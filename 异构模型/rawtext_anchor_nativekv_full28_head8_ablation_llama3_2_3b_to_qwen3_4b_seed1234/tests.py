import torch

from translator import NativeKVTranslator, depth_indices


def main():
    assert len(depth_indices()) == 36
    x = torch.randn(2, 28, 32, 8, 128)
    for architecture in ("local5_samehead", "full28_samehead", "full28_diagonal", "full28_head8"):
        module = NativeKVTranslator(architecture)
        k, v = module(x, x)
        assert k.shape == v.shape == (2, 36, 32, 8, 128)
        zero = torch.zeros_like(x)
        zk, zv = module(zero, zero)
        assert torch.count_nonzero(zk) == 0 and torch.count_nonzero(zv) == 0, f"{architecture}: Writer(0) != 0"
        assert all(layer.bias is None for layer in module.modules() if isinstance(layer, torch.nn.Linear))
    print("All translator shape/bias/zero tests passed", flush=True)


if __name__ == "__main__":
    main()
