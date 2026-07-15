import torch
import torch.nn.functional as F


def _sample_token_indices(length, max_tokens, answer_mask, device):
    if length <= max_tokens:
        return torch.arange(length, device=device)
    answer_positions = []
    if answer_mask is not None and answer_mask.numel() >= length:
        answer_positions = torch.nonzero(answer_mask[:length], as_tuple=False).flatten().tolist()
    if len(answer_positions) >= max_tokens:
        selection = torch.linspace(0, len(answer_positions) - 1, max_tokens).round().long().tolist()
        return torch.tensor([answer_positions[index] for index in selection], device=device)
    required = set(answer_positions)
    candidates = torch.linspace(0, length - 1, max_tokens * 2).round().long().tolist()
    for index in candidates:
        required.add(index)
        if len(required) >= max_tokens:
            break
    return torch.tensor(sorted(required), device=device, dtype=torch.long)


def _token_vectors(tensor, indices):
    selected = tensor.float()[:, indices, :].permute(1, 0, 2).reshape(indices.numel(), -1)
    return F.normalize(selected, dim=-1, eps=1e-6)


def _cosine_graph(vectors):
    return vectors @ vectors.transpose(0, 1)


def _query_contribution_graph(route, values, indices):
    route = route.float()[0]
    groups = route.shape[0] // values.shape[0]
    if groups * values.shape[0] != route.shape[0]:
        raise ValueError("Query heads are not divisible by KV heads")
    expanded_values = values.float().repeat_interleave(groups, dim=0)
    contribution = route[:, indices].unsqueeze(-1) * expanded_values[:, indices, :]
    vectors = contribution.permute(1, 0, 2).reshape(indices.numel(), -1)
    return _cosine_graph(F.normalize(vectors, dim=-1, eps=1e-6))


def structure_preservation_losses(
    writer_memory,
    teacher_memory,
    writer_diagnostics,
    teacher_diagnostics,
    token_aligned,
    max_tokens=64,
):
    zero = writer_memory["keys"][0].new_zeros((), dtype=torch.float32)
    output = {
        "binding": zero,
        "key_relation": zero,
        "value_relation": zero,
        "readout_relation": zero,
        "layers": 0,
        "sampled_tokens": 0,
        "token_aligned": bool(token_aligned),
    }
    if not token_aligned:
        return output

    binding_terms = []
    key_terms = []
    value_terms = []
    readout_terms = []
    sampled_tokens = []
    answer_mask = writer_memory.get("answer_token_mask")
    for layer, (writer_key, writer_value, teacher_key, teacher_value) in enumerate(
        zip(
            writer_memory["keys"],
            writer_memory["values"],
            teacher_memory["keys"],
            teacher_memory["values"],
        )
    ):
        length = min(
            writer_key.shape[1],
            writer_value.shape[1],
            teacher_key.shape[1],
            teacher_value.shape[1],
        )
        if length < 2:
            continue
        indices = _sample_token_indices(length, max_tokens, answer_mask, writer_key.device)
        sampled_tokens.append(int(indices.numel()))
        wk = _token_vectors(writer_key, indices)
        wv = _token_vectors(writer_value, indices)
        tk = _token_vectors(teacher_key.detach(), indices)
        tv = _token_vectors(teacher_value.detach(), indices)

        writer_binding = wk @ wv.transpose(0, 1)
        teacher_binding = tk @ tv.transpose(0, 1)
        binding_terms.append(F.mse_loss(writer_binding, teacher_binding))
        binding_terms.append(F.mse_loss(writer_binding.diagonal(), teacher_binding.diagonal()))
        key_terms.append(F.mse_loss(_cosine_graph(wk), _cosine_graph(tk)))
        value_terms.append(F.mse_loss(_cosine_graph(wv), _cosine_graph(tv)))

        writer_slot = writer_diagnostics[str(layer)]
        teacher_slot = teacher_diagnostics[str(layer)]
        writer_graph = _query_contribution_graph(
            writer_slot["route_tensor"], writer_value, indices
        )
        teacher_graph = _query_contribution_graph(
            teacher_slot["route_tensor"].detach(), teacher_value.detach(), indices
        )
        readout_terms.append(F.mse_loss(writer_graph, teacher_graph))

    if not binding_terms:
        return output
    output.update(
        {
            "binding": torch.stack(binding_terms).mean(),
            "key_relation": torch.stack(key_terms).mean(),
            "value_relation": torch.stack(value_terms).mean(),
            "readout_relation": torch.stack(readout_terms).mean(),
            "layers": len(key_terms),
            "sampled_tokens": int(sum(sampled_tokens) / len(sampled_tokens)),
        }
    )
    return output


def detached_structure_metrics(losses):
    return {
        key: float(value.detach().cpu()) if torch.is_tensor(value) else value
        for key, value in losses.items()
    }


__all__ = ["detached_structure_metrics", "structure_preservation_losses"]
