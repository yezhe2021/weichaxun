import torch

from experiment import receiver_suffix
from translator import NativeKVTranslator


def main():
    row = {"id": "unit", "encoded": {"qwen": {
        "question_prefix": [99, 10, 11, 12], "body": [99, 10, 11, 12, 20],
        "full": [99, 10, 11, 12, 20, 30], "receiver_answer": [40, 41]}}}
    assert receiver_suffix(row) == [10, 11, 12, 40, 41]
    module = NativeKVTranslator("full28_diagonal")
    x = torch.zeros(1, 28, 32, 8, 128)
    key, value = module(x, x)
    assert key.shape == value.shape == (1, 36, 32, 8, 128)
    assert torch.count_nonzero(key) == 0 and torch.count_nonzero(value) == 0
    assert all(layer.bias is None for layer in module.modules() if isinstance(layer, torch.nn.Linear))
    print("Memory-first layout and Full28-Diagonal zero/bias/shape tests passed", flush=True)


if __name__ == "__main__": main()
