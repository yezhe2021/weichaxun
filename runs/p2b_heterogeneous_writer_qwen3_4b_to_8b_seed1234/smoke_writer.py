import torch

from p2b_writer import HeterogeneousNativeKVWriter, shape_only_memory


def main():
    writer = HeterogeneousNativeKVWriter(
        sender_layers=4,
        sender_heads=2,
        sender_head_dim=8,
        receiver_layers=6,
        receiver_heads=3,
        receiver_head_dim=5,
        layer_width=3,
    )
    memory = {
        "keys": [torch.randn(2, 7, 8) for _ in range(4)],
        "values": [torch.randn(2, 7, 8) for _ in range(4)],
        "answer_token_mask": torch.tensor([False, False, True, False, False, False, False]),
    }
    output = writer(memory)
    assert len(output["keys"]) == 6
    assert output["keys"][0].shape == (3, 7, 5)
    loss = sum(tensor.square().mean() for tensor in output["keys"] + output["values"])
    loss.backward()
    assert all(parameter.grad is not None for parameter in writer.parameters())
    raw = shape_only_memory(memory, receiver_layers=6, receiver_heads=3, receiver_head_dim=5)
    assert raw["keys"][0].shape == (3, 7, 5)
    print("p2b_writer_smoke=passed")


if __name__ == "__main__":
    main()
