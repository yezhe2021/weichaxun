import torch

from p2e_structure import span_relation_losses, transport_alignment_losses
from p2e_writer import StructurePreservingNativeKVWriter


def masks(length):
    names = (
        "evidence_a",
        "evidence_b",
        "target_person",
        "organization_a",
        "organization_b",
        "answer",
    )
    output = {}
    for index, name in enumerate(names):
        mask = torch.zeros(length, dtype=torch.bool)
        mask[index % length] = True
        output[name] = mask
    output["evidence_a"][: length // 2] = True
    output["evidence_b"][length // 2 :] = True
    return output


def diagnostics(layers, query_heads, tokens, hidden):
    output = {}
    for layer in range(layers):
        route = torch.rand(1, query_heads, tokens)
        route = route / route.sum(dim=-1, keepdim=True)
        output[str(layer)] = {
            "route_tensor": route,
            "readout_tensor": torch.rand(1, query_heads, hidden // query_heads),
        }
    return output


def main():
    sender_layers, receiver_layers = 4, 5
    heads, dim = 2, 8
    sender_tokens, teacher_tokens = 9, 8
    writer = StructurePreservingNativeKVWriter(
        sender_layers, heads, dim, receiver_layers, heads, dim,
        top_k=2, adapter_rank=4, shared_routing=True,
        teacher_k_rms=torch.ones(receiver_layers, heads),
    )
    source = {
        "keys": [torch.randn(heads, sender_tokens, dim) for _ in range(sender_layers)],
        "values": [torch.randn(heads, sender_tokens, dim) for _ in range(sender_layers)],
        "answer_token_mask": masks(sender_tokens)["answer"],
    }
    teacher = {
        "keys": [torch.randn(heads, teacher_tokens, dim) for _ in range(receiver_layers)],
        "values": [torch.randn(heads, teacher_tokens, dim) for _ in range(receiver_layers)],
    }
    output = writer(source)
    overlap = torch.rand(teacher_tokens, sender_tokens)
    pool = overlap / overlap.sum(dim=1, keepdim=True)
    mass = overlap.T / overlap.T.sum(dim=1, keepdim=True)
    transport = {
        "teacher_pool_from_sender": pool,
        "sender_mass_to_teacher": mass,
        "sender_span_masks": masks(sender_tokens),
        "teacher_span_masks": masks(teacher_tokens),
    }
    writer_diag = diagnostics(receiver_layers, heads * 2, sender_tokens, heads * 2 * dim)
    teacher_diag = diagnostics(receiver_layers, heads * 2, teacher_tokens, heads * 2 * dim)
    route, readout = transport_alignment_losses(
        writer_diag, teacher_diag, transport, output["keys"][0]
    )
    structure = span_relation_losses(output, teacher, writer_diag, teacher_diag, transport)
    loss = route + readout + structure["binding"] + structure["key_relation"]
    loss.backward()
    if not any(parameter.grad is not None for parameter in writer.parameters()):
        raise RuntimeError("Writer did not receive gradients")
    print("P2-E synthetic smoke passed")


if __name__ == "__main__":
    main()
