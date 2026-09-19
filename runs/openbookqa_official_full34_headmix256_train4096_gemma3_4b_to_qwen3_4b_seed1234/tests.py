import torch

from translator import NativeKVTranslator, ResidualKVAdapter


if __name__ == "__main__":
    adapter = ResidualKVAdapter(rank=64)
    x = torch.randn(1, 36, 32, 8, 128)
    y = torch.randn_like(x)
    fk, fv, dk, dv = adapter(x, y)
    assert torch.equal(fk, x) and torch.equal(fv, y)
    assert torch.count_nonzero(dk) == 0 and torch.count_nonzero(dv) == 0
    z = torch.zeros_like(x)
    fk, fv, _, _ = adapter(z, z)
    assert torch.count_nonzero(fk) == 0 and torch.count_nonzero(fv) == 0
    assert all(layer.bias is None for layer in adapter.modules() if isinstance(layer, torch.nn.Linear))
    assert sum(p.numel() for p in adapter.parameters()) == 9437184

    # Exercise identical tensor logic at reduced dimensions; the production model is 736M parameters.
    kwargs = dict(source_layers=3, target_layers=2, source_heads=2, target_heads=4,
                  source_dim=4, target_dim=2, hidden_dim=5, depth_output_dim=4)
    translator = NativeKVTranslator("full34_headmix256", head_mapping="full_head", **kwargs)
    source_k = torch.randn(1, 3, 32, 2, 4)
    source_v = torch.randn_like(source_k)
    target_k, target_v = translator(source_k, source_v)
    assert target_k.shape == (1, 2, 32, 4, 2) and target_v.shape == target_k.shape
    assert all(layer.bias is None for layer in translator.modules() if isinstance(layer, torch.nn.Linear))
    head, depth = translator.correspondence()
    assert head["k"].shape == (2, 4, 2) and depth["k"].shape == (2, 3)
    production_parameters = 2 * 36 * (8704 * 1024 + 1024 * 256 + 1024 * 1024)
    assert production_parameters == 736100352

    for mapping in ("fixed_split", "block_head"):
        ablation = NativeKVTranslator("full34_headmix256", head_mapping=mapping, **kwargs)
        ak, av = ablation(source_k, source_v)
        assert ak.shape == target_k.shape and av.shape == target_v.shape
    print("Full34 depth/head geometry, ablation switches, and Residual64 tests passed", flush=True)
