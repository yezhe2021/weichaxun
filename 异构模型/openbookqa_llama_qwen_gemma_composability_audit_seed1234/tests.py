import torch

from translators import LlamaToQwen, QwenToGemma, ResidualAdapter


def main():
    with torch.device("meta"):
        first = LlamaToQwen()
        first_adapter = ResidualAdapter(36, 8, 128)
        second = QwenToGemma()
        second_adapter = ResidualAdapter(34, 4, 256)
        key = torch.empty(1, 28, 32, 8, 128, device="meta")
        value = torch.empty_like(key)
        key, value = first(key, value)
        assert key.shape == (1, 36, 32, 8, 128)
        key, value = first_adapter(key, value)
        key, value = second(key, value)
        assert key.shape == (1, 34, 32, 4, 256)
        key, value = second_adapter(key, value)
        assert value.shape == (1, 34, 32, 4, 256)
    assert all(module.bias is None for model in (first, second, first_adapter, second_adapter)
               for module in model.modules() if isinstance(module, torch.nn.Linear))
    print("PASS: Llama->Qwen->Gemma shape chain and bias-free writers")


if __name__ == "__main__":
    main()
