from types import SimpleNamespace

import torch
import torch.nn as nn

from llama_specific_reader import LlamaSpecificExternalReader


class FakeAttention(nn.Module):
    def __init__(self, hidden_size=32, query_heads=4, kv_heads=2, head_dim=8):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=query_heads,
            num_key_value_heads=kv_heads,
        )
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, query_heads * head_dim, bias=False)
        self.q_norm = nn.Identity()
        self.o_proj = nn.Linear(query_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states):
        return hidden_states


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()

    def forward(self, hidden_states):
        return self.self_attn(hidden_states=hidden_states)


class FakeBase(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer() for _ in range(layers)])


class FakeReceiver(nn.Module):
    def __init__(self, layers=4):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        self.model = FakeBase(layers)

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def main():
    for variant in ("minimal_reader", "routed_reader"):
        receiver = FakeReceiver()
        for parameter in receiver.parameters():
            parameter.requires_grad_(False)
        reader = LlamaSpecificExternalReader(
            receiver,
            sender_layers=3,
            sender_kv_heads=2,
            sender_head_dim=8,
            variant=variant,
            top_k=2,
            query_rank=8,
            output_rank=8,
        )
        memory = {
            "keys": [torch.randn(2, 7, 8) for _ in range(3)],
            "values": [torch.randn(2, 7, 8) for _ in range(3)],
            "answer_token_mask": torch.tensor([0, 0, 0, 0, 0, 1, 0], dtype=torch.bool),
        }
        diagnostics = {}
        with reader.inject(receiver, memory, diagnostics):
            output = receiver(torch.randn(2, 5, 32))
        output.square().mean().backward()
        if not any(parameter.grad is not None for parameter in reader.parameters()):
            raise RuntimeError(f"No Reader gradients for {variant}")
        if any(parameter.grad is not None for parameter in receiver.parameters()):
            raise RuntimeError("Frozen receiver received gradients")
        if len(reader.routing_diagnostics()) != 4:
            raise RuntimeError("Missing routing diagnostics")
    print("Experiment B synthetic Reader smoke passed")


if __name__ == "__main__":
    main()
