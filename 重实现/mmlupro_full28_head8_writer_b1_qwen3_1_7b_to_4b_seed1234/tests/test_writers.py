from __future__ import annotations

import tempfile

import torch

from writers import FullKVWriter, Full28Head8Writer, load_writer, save_writer


def config():
    return {
        "source_layers": 28,
        "target_layers": 36,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rms_epsilon": 1e-6,
    }


def scales():
    return {
        "source_k": torch.rand(28, 8, 128) + 0.5,
        "source_v": torch.rand(28, 8, 128) + 0.5,
        "target_k": torch.rand(36, 8, 128) + 0.5,
        "target_v": torch.rand(36, 8, 128) + 0.5,
    }


def inputs(tokens=19):
    return torch.randn(28, tokens, 8, 128), torch.randn(28, tokens, 8, 128)


def test_shapes_zero_and_no_bias():
    for kind in ("d0", "d1", "d2", "full28"):
        writer = FullKVWriter(kind, scales(), config())
        key = torch.zeros(28, 11, 8, 128)
        value = torch.zeros_like(key)
        output_k, output_v = writer(key, value)
        assert output_k.shape == (36, 11, 8, 128)
        assert output_v.shape == output_k.shape
        assert torch.count_nonzero(output_k) == 0
        assert torch.count_nonzero(output_v) == 0
        assert all(layer.bias is None for layer in writer.k_linears)
        assert all(layer.bias is None for layer in writer.v_linears)


def test_parameters_are_independent_by_layer_and_family():
    writer = FullKVWriter("full28", scales(), config())
    pointers = [layer.weight.data_ptr() for layer in list(writer.k_linears) + list(writer.v_linears)]
    assert len(pointers) == len(set(pointers))
    assert not any(writer.k_linears[i].weight is writer.v_linears[i].weight for i in range(36))


def test_full28_shape_and_layer_concat_order():
    writer = FullKVWriter("full28", scales(), config())
    assert writer.k_linears[0].in_features == 28 * 128
    normalized = torch.zeros(28, 1, 8, 128)
    for layer in range(28):
        normalized[layer].fill_(layer + 1)
    features = writer._features(normalized, 0)
    assert features.shape == (1, 8, 28 * 128)
    for layer in range(28):
        block = features[..., layer * 128:(layer + 1) * 128]
        assert torch.equal(block, torch.full_like(block, layer + 1))


def test_full28_initialization_equals_calibrated_d0():
    fixed_scales = scales()
    d0 = FullKVWriter("d0", fixed_scales, config())
    full28 = FullKVWriter("full28", fixed_scales, config())
    key, value = inputs(tokens=3)
    d0_k, d0_v = d0(key, value)
    full_k, full_v = full28(key, value)
    assert torch.allclose(d0_k, full_k, atol=1e-6, rtol=1e-6)
    assert torch.allclose(d0_v, full_v, atol=1e-6, rtol=1e-6)


def test_d2_initialization_equals_calibrated_d0():
    fixed_scales = scales()
    d0 = FullKVWriter("d0", fixed_scales, config())
    d2 = FullKVWriter("d2", fixed_scales, config())
    key, value = inputs()
    d0_k, d0_v = d0(key, value)
    d2_k, d2_v = d2(key, value)
    assert torch.allclose(d0_k, d2_k, atol=1e-6, rtol=1e-6)
    assert torch.allclose(d0_v, d2_v, atol=1e-6, rtol=1e-6)


def test_head_independence():
    writer = FullKVWriter("full28", scales(), config())
    key, value = inputs()
    reference_k, reference_v = writer(key, value)
    changed_key, changed_value = key.clone(), value.clone()
    changed_key[:, :, 3] += 10
    changed_value[:, :, 3] -= 10
    changed_k, changed_v = writer(changed_key, changed_value)
    unchanged = [0, 1, 2, 4, 5, 6, 7]
    assert torch.equal(reference_k[:, :, unchanged], changed_k[:, :, unchanged])
    assert torch.equal(reference_v[:, :, unchanged], changed_v[:, :, unchanged])
    assert not torch.equal(reference_k[:, :, 3], changed_k[:, :, 3])
    assert not torch.equal(reference_v[:, :, 3], changed_v[:, :, 3])


def test_token_independence():
    writer = FullKVWriter("full28", scales(), config())
    key, value = inputs(tokens=23)
    reference_k, reference_v = writer(key, value)
    changed_key, changed_value = key.clone(), value.clone()
    changed_key[:, 17] += 10
    changed_value[:, 17] -= 10
    changed_k, changed_v = writer(changed_key, changed_value)
    unchanged = list(range(17)) + list(range(18, 23))
    assert torch.equal(reference_k[:, unchanged], changed_k[:, unchanged])
    assert torch.equal(reference_v[:, unchanged], changed_v[:, unchanged])
    assert not torch.equal(reference_k[:, 17], changed_k[:, 17])
    assert not torch.equal(reference_v[:, 17], changed_v[:, 17])


def test_neighbor_order_and_boundary_rules():
    writer = FullKVWriter("d2", scales(), config())
    assert writer.local_source_indices[0].tolist() == [0, 1, 2, 3, 4]
    assert writer.local_source_indices[-1].tolist() == [23, 24, 25, 26, 27]
    for target, neighbors in enumerate(writer.local_source_indices.tolist()):
        assert len(neighbors) == 5
        assert len(set(neighbors)) == 5
        assert neighbors == sorted(neighbors)
        assert int(writer.nearest_indices[target]) in neighbors


def test_checkpoint_round_trip_and_metadata():
    writer = FullKVWriter("d2", scales(), config())
    key, value = inputs()
    expected = writer(key, value)
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/writer.pt"
        save_writer(path, writer)
        restored = FullKVWriter("d2", scales(), config())
        load_writer(path, restored)
        actual = restored(key, value)
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])


def small_config():
    return {"source_layers": 5, "target_layers": 4, "num_kv_heads": 2, "head_dim": 4, "rms_epsilon": 1e-6}


def small_scales():
    return {
        "source_k": torch.ones(5, 2, 4), "source_v": torch.ones(5, 2, 4),
        "target_k": torch.ones(4, 2, 4), "target_v": torch.ones(4, 2, 4),
    }


def test_head8_identity_extension_zero_and_freeze_policy():
    base = FullKVWriter("full28", small_scales(), small_config())
    extended = Full28Head8Writer(small_scales(), small_config())
    with torch.no_grad():
        for target in range(4):
            extended.k_linears[target].weight.copy_(base.k_linears[target].weight)
            extended.v_linears[target].weight.copy_(base.v_linears[target].weight)
    key, value = torch.randn(5, 5, 2, 4), torch.randn(5, 5, 2, 4)
    expected, actual = base(key, value), extended(key, value)
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])
    zero = extended(torch.zeros_like(key), torch.zeros_like(value))
    assert torch.count_nonzero(zero[0]) == 0 and torch.count_nonzero(zero[1]) == 0
    assert all(not layer.weight.requires_grad for layer in list(extended.k_linears) + list(extended.v_linears))
    assert all(layer.weight.requires_grad for layer in list(extended.k_head_linears) + list(extended.v_head_linears))
    assert all(layer.bias is None for layer in list(extended.k_head_linears) + list(extended.v_head_linears))


def test_head8_cross_head_blocks_are_independent_and_receive_gradients():
    writer = Full28Head8Writer(small_scales(), small_config())
    key, value = torch.randn(5, 2, 2, 4), torch.randn(5, 2, 2, 4)
    output = writer(key, value)
    loss = output[0].float().square().mean() + output[1].float().square().mean()
    loss.backward()
    assert all(layer.weight.grad is not None for layer in list(writer.k_head_linears) + list(writer.v_head_linears))
    assert all(layer.weight.grad is None for layer in list(writer.k_linears) + list(writer.v_linears))
    pointers = [layer.weight.data_ptr() for layer in list(writer.k_head_linears) + list(writer.v_head_linears)]
    assert len(pointers) == len(set(pointers))
