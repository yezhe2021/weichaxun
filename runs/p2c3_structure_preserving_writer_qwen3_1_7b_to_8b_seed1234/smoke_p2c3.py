import torch

from p2c3_structure import structure_preservation_losses
from p2c3_writer import StructurePreservingNativeKVWriter
from train_p2c3_writer import value_mismatched_memory


def diagnostics(layers, query_heads, tokens):
    output = {}
    for layer in range(layers):
        route = torch.rand(1, query_heads, tokens)
        route = route / route.sum(dim=-1, keepdim=True)
        output[str(layer)] = {
            "route_tensor": route,
            "readout_tensor": torch.randn(1, query_heads * 8),
            "route_entropy_tensor": torch.tensor(1.0),
            "target_mass_tensor": route[..., -1].mean(),
        }
    return output


def run(shared_routing):
    torch.manual_seed(1234)
    writer = StructurePreservingNativeKVWriter(
        sender_layers=4,
        sender_heads=2,
        sender_head_dim=8,
        receiver_layers=3,
        receiver_heads=4,
        receiver_head_dim=8,
        top_k=2,
        adapter_rank=4,
        shared_routing=shared_routing,
        teacher_k_rms=torch.ones(3, 4),
    )
    source = {
        "keys": [torch.randn(2, 7, 8) for _ in range(4)],
        "values": [torch.randn(2, 7, 8) for _ in range(4)],
        "answer_token_mask": torch.tensor([False, False, False, False, False, False, True]),
    }
    teacher = {
        "keys": [torch.randn(4, 7, 8) for _ in range(3)],
        "values": [torch.randn(4, 7, 8) for _ in range(3)],
        "answer_token_mask": source["answer_token_mask"],
    }
    output = writer(source)
    assert len(output["keys"]) == 3
    assert output["keys"][0].shape == output["values"][0].shape == (4, 7, 8)
    writer_diag = diagnostics(3, 8, 7)
    teacher_diag = diagnostics(3, 8, 7)
    losses = structure_preservation_losses(
        output, teacher, writer_diag, teacher_diag, True, max_tokens=5
    )
    objective = sum(output["keys"]).mean() + sum(output["values"]).mean()
    objective = objective + losses["binding"] + losses["key_relation"]
    objective = objective + losses["value_relation"] + losses["readout_relation"]
    objective.backward()
    assert torch.isfinite(objective)
    assert any(parameter.grad is not None for parameter in writer.parameters())
    if shared_routing:
        difference = writer.routing_difference_tensors()
        assert float(difference["layer_support_disagreement"]) == 0.0
    unrelated = writer(
        {
            "keys": [torch.randn(2, 5, 8) for _ in range(4)],
            "values": [torch.randn(2, 5, 8) for _ in range(4)],
        }
    )
    mismatch = value_mismatched_memory(output, unrelated)
    assert mismatch["keys"][0].shape[1] == mismatch["values"][0].shape[1] == 7


run(shared_routing=False)
run(shared_routing=True)
print("p2c3_smoke=passed")
