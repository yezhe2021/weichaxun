import torch

from translators import GemmaToQwen, QwenToLlama, ResidualAdapter


def main():
    with torch.device("meta"):
        forward = GemmaToQwen()
        reverse = QwenToLlama()
        forward_adapter = ResidualAdapter(36)
        reverse_adapter = ResidualAdapter(28)
        gemma_k = torch.empty(1, 34, 32, 4, 256, device="meta")
        gemma_v = torch.empty_like(gemma_k)
        qwen_k, qwen_v = forward(gemma_k, gemma_v)
        assert qwen_k.shape == (1, 36, 32, 8, 128)
        qwen_k, qwen_v = forward_adapter(qwen_k, qwen_v)
        llama_k, llama_v = reverse(qwen_k, qwen_v)
        assert llama_k.shape == (1, 28, 32, 8, 128)
        llama_k, llama_v = reverse_adapter(llama_k, llama_v)
        assert llama_v.shape == (1, 28, 32, 8, 128)
    assert all(module.bias is None for model in (forward, reverse)
               for module in model.modules() if isinstance(module, torch.nn.Linear))
    print("PASS: shape chain, separate K/V and bias-free writer checks")


if __name__ == "__main__":
    main()
