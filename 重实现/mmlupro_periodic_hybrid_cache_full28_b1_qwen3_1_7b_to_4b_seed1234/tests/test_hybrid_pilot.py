import torch

from hybrid_pilot import PATTERNS, distances, hybrid_cache

def run():
    native_k = torch.zeros(36, 3, 2, 4)
    native_v = torch.ones_like(native_k)
    writer_k = torch.full_like(native_k, 7)
    writer_v = torch.full_like(native_k, 9)
    for name, layers in PATTERNS.items():
        key, value = hybrid_cache(native_k, native_v, writer_k, writer_v, layers)
        assert len(layers) == (0 if name == "native" else 36 if name == "full_writer" else 12)
        for layer in range(36):
            assert torch.equal(key[layer], writer_k[layer] if layer in layers else native_k[layer])
            assert torch.equal(value[layer], writer_v[layer] if layer in layers else native_v[layer])
    logits = torch.randn(101)
    metrics = distances(logits, logits.clone(), torch.tensor([1, 3, 5, 7]))
    assert abs(metrics["final_position_full_vocab_kl_vs_native"]) < 1e-7
    assert abs(metrics["choice_only_kl_vs_native"]) < 1e-7
    assert metrics["native_choice_top1_agreement"] == 1.0

if __name__ == "__main__":
    run()
