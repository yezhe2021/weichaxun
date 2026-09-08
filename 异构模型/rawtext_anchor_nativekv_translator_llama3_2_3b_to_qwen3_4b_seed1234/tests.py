import torch

from translator import NativeKVTranslator, depth_indices, direct_nearest, representation_loss


def run_tests():
    nearest, local5 = depth_indices(window=1), depth_indices(window=5)
    assert len(nearest) == len(local5) == 36 and nearest[0] == [0] and nearest[-1] == [27]
    assert local5[0] == [0, 0, 0, 1, 2] and local5[-1] == [25, 26, 27, 27, 27]
    key, value = torch.randn(2, 28, 3, 8, 4), torch.randn(2, 28, 3, 8, 4)
    module = NativeKVTranslator(5, dim=4)
    pk, pv = module(key, value)
    assert pk.shape == pv.shape == (2, 36, 3, 8, 4)
    loss, detail = representation_loss(pk, pv, torch.randn_like(pk), torch.randn_like(pv))
    loss.backward()
    assert all(layer.bias is None for layer in list(module.k_maps) + list(module.v_maps))
    assert module.k_maps[0].weight.grad is not None and module.v_maps[0].weight.grad is not None
    zk, zv = module(torch.zeros_like(key), torch.zeros_like(value))
    assert torch.count_nonzero(zk) == torch.count_nonzero(zv) == 0
    dk, dv = direct_nearest(key, value)
    assert dk.shape == dv.shape == pk.shape
    torch.testing.assert_close(dk[:, 0], key[:, 0]); torch.testing.assert_close(dk[:, -1], key[:, -1])
    assert set(detail) == {"k_nmse", "v_nmse", "k_cosine", "v_cosine"}
    return {"passed": True, "checks": ["normalized_depth", "local5_clamp", "36_layer_output",
            "per_target_layer_maps", "separate_kv", "no_head_mix", "bias_false", "Writer(0)=0",
            "direct_nearest", "representation_metrics", "gradient"]}


if __name__ == "__main__": print(run_tests())
