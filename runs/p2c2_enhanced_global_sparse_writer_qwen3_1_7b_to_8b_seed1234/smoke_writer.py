import torch

from p2c2_writer import EnhancedGlobalNativeKVWriter, shape_only_memory


def main():
    writer = EnhancedGlobalNativeKVWriter(
        sender_layers=4,
        sender_heads=2,
        sender_head_dim=8,
        receiver_layers=6,
        receiver_heads=3,
        receiver_head_dim=5,
        top_k=3,
        adapter_mode="per_head",
        adapter_rank=4,
        teacher_k_rms=torch.ones(6, 3),
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
    routing = writer.routing_diagnostics()
    assert len(routing) == 12 and all(len(row["sender_layers"]) == 3 for row in routing)
    print("p2c2_writer_smoke=passed")


if __name__ == "__main__":
    main()
