import torch

from translator import NativeKVTranslator, ResidualKVAdapter


if __name__ == "__main__":
    adapter = ResidualKVAdapter(rank=64)
    x = torch.randn(1, 34, 32, 4, 256)
    y = torch.randn_like(x)
    fk, fv, dk, dv = adapter(x, y)
    assert torch.equal(fk, x) and torch.equal(fv, y)
    assert torch.count_nonzero(dk) == 0 and torch.count_nonzero(dv) == 0
    z = torch.zeros_like(x)
    fk, fv, _, _ = adapter(z, z)
    assert torch.count_nonzero(fk) == 0 and torch.count_nonzero(fv) == 0
    assert all(layer.bias is None for layer in adapter.modules() if isinstance(layer, torch.nn.Linear))
    assert sum(p.numel() for p in adapter.parameters()) == 8912896

    kwargs = dict(source_layers=3, target_layers=2, source_heads=4, target_heads=2,
                  source_dim=4, target_dim=8, hidden_dim=5, depth_output_dim=4)
    translator = NativeKVTranslator("full36_headmix128", head_mapping="full_head", **kwargs)
    source_k = torch.randn(1, 3, 32, 4, 4)
    source_v = torch.randn_like(source_k)
    target_k, target_v = translator(source_k, source_v)
    assert target_k.shape == (1, 2, 32, 2, 8) and target_v.shape == target_k.shape
    assert all(layer.bias is None for layer in translator.modules() if isinstance(layer, torch.nn.Linear))
    head, depth = translator.correspondence()
    assert head["k"].shape == (2, 2, 4) and depth["k"].shape == (2, 3)
    production_parameters = 2 * 34 * (4608 * 1024 + 1024 * 128 + 1024 * 1024)
    assert production_parameters == 401080320

    for mapping in ("fixed_merge", "block_head"):
        ablation = NativeKVTranslator("full36_headmix128", head_mapping=mapping, **kwargs)
        ak, av = ablation(source_k, source_v)
        assert ak.shape == target_k.shape and av.shape == target_v.shape
    print("Reverse Full36 depth/head geometry, ablation switches, and Gemma Residual64 tests passed", flush=True)
