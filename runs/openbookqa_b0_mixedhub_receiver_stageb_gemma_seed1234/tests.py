import torch

from translators import LlamaToQwen, QwenToGemma, ResidualAdapter


def main():
    with torch.device("meta"):
        first, second = LlamaToQwen(), QwenToGemma()
        adapter = ResidualAdapter(34, 4, 256, 64)
        key = torch.empty(1, 28, 32, 8, 128, device="meta")
        value = torch.empty_like(key)
        hub_k, hub_v = first(key, value)
        assert hub_k.shape == (1, 36, 32, 8, 128)
        receiver_k, receiver_v = second(hub_k, hub_v)
        assert receiver_k.shape == (1, 34, 32, 4, 256)
        final_k, final_v = adapter(receiver_k, receiver_v)
        assert final_k.shape == receiver_k.shape and final_v.shape == receiver_v.shape
    assert all(module.bias is None for module in adapter.modules() if isinstance(module, torch.nn.Linear))
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 8912896
    print("PASS: B0 paired Hub geometry and Gemma Residual64")


if __name__ == "__main__":
    main()
