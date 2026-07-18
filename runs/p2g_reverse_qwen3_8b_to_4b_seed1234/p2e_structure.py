import torch
import torch.nn.functional as F


SPAN_NAMES = (
    "evidence_a",
    "evidence_b",
    "target_person",
    "organization_a",
    "organization_b",
    "answer",
)


def identity_token_transport(sender_row, teacher_row):
    sender_ids = sender_row.get("evidence_token_ids")
    teacher_ids = teacher_row.get("evidence_token_ids")
    if sender_ids is None or teacher_ids is None:
        raise ValueError("Identity transport requires evidence_token_ids in both caches")
    if sender_ids != teacher_ids:
        raise ValueError(
            f"Evidence token mismatch for pair {sender_row.get('pair_id')}: "
            f"sender={len(sender_ids)} teacher={len(teacher_ids)}"
        )
    return {"sender_mass_to_teacher": torch.eye(len(sender_ids), dtype=torch.float32)}


def _normalized_distribution(tensor):
    tensor = tensor.clamp_min(1e-8)
    return tensor / tensor.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def project_route_to_teacher(route, sender_mass_to_teacher):
    mass = sender_mass_to_teacher.to(device=route.device, dtype=route.dtype)
    return _normalized_distribution(torch.einsum("bhs,st->bht", route, mass))


def transport_alignment_losses(writer_diag, teacher_diag, transport, reference):
    route_terms = []
    readout_terms = []
    mass = transport["sender_mass_to_teacher"]
    for layer in sorted(int(key) for key in writer_diag if key.isdigit()):
        writer_slot = writer_diag[str(layer)]
        teacher_slot = teacher_diag[str(layer)]
        projected = project_route_to_teacher(writer_slot["route_tensor"].float(), mass)
        teacher_route = _normalized_distribution(teacher_slot["route_tensor"].detach().float())
        route_terms.append(
            F.kl_div(projected.clamp_min(1e-8).log(), teacher_route, reduction="batchmean")
            / projected.shape[1]
        )
        writer_readout = writer_slot["readout_tensor"].float()
        teacher_readout = teacher_slot["readout_tensor"].detach().float()
        cosine = 1.0 - F.cosine_similarity(writer_readout, teacher_readout, dim=-1).mean()
        norm = (
            writer_readout.norm(dim=-1).clamp_min(1e-6).log()
            - teacher_readout.norm(dim=-1).clamp_min(1e-6).log()
        ).square().mean()
        readout_terms.append(cosine + 0.1 * norm)
    if not route_terms:
        zero = reference.new_zeros((), dtype=torch.float32)
        return zero, zero
    return torch.stack(route_terms).mean(), torch.stack(readout_terms).mean()


def _span_vectors(tensor, masks):
    vectors = []
    for name in SPAN_NAMES:
        mask = masks[name].to(tensor.device)
        if not mask.any():
            raise ValueError(f"Empty token span: {name}")
        vector = tensor.float()[:, mask, :].mean(dim=1).reshape(-1)
        vectors.append(F.normalize(vector, dim=0, eps=1e-6))
    return torch.stack(vectors)


def _contribution_vectors(route, values, masks):
    route = route.float()[0]
    groups = route.shape[0] // values.shape[0]
    if groups * values.shape[0] != route.shape[0]:
        raise ValueError("Query heads must be divisible by KV heads")
    expanded_values = values.float().repeat_interleave(groups, dim=0)
    vectors = []
    for name in SPAN_NAMES:
        mask = masks[name].to(values.device)
        contribution = (
            route[:, mask].unsqueeze(-1) * expanded_values[:, mask, :]
        ).sum(dim=1).reshape(-1)
        vectors.append(F.normalize(contribution, dim=0, eps=1e-6))
    return torch.stack(vectors)


def span_relation_losses(writer_memory, teacher_memory, writer_diag, teacher_diag, transport):
    sender_masks = transport["sender_span_masks"]
    teacher_masks = transport["teacher_span_masks"]
    binding_terms = []
    key_terms = []
    value_terms = []
    contribution_terms = []
    for layer, (writer_key, writer_value, teacher_key, teacher_value) in enumerate(
        zip(
            writer_memory["keys"],
            writer_memory["values"],
            teacher_memory["keys"],
            teacher_memory["values"],
        )
    ):
        writer_k = _span_vectors(writer_key, sender_masks)
        writer_v = _span_vectors(writer_value, sender_masks)
        teacher_k = _span_vectors(teacher_key.detach(), teacher_masks)
        teacher_v = _span_vectors(teacher_value.detach(), teacher_masks)
        key_terms.append(F.mse_loss(writer_k @ writer_k.T, teacher_k @ teacher_k.T))
        value_terms.append(F.mse_loss(writer_v @ writer_v.T, teacher_v @ teacher_v.T))
        binding_terms.append(F.mse_loss(writer_k @ writer_v.T, teacher_k @ teacher_v.T))

        writer_contribution = _contribution_vectors(
            writer_diag[str(layer)]["route_tensor"], writer_value, sender_masks
        )
        teacher_contribution = _contribution_vectors(
            teacher_diag[str(layer)]["route_tensor"].detach(),
            teacher_value.detach(),
            teacher_masks,
        )
        contribution_terms.append(
            F.mse_loss(
                writer_contribution @ writer_contribution.T,
                teacher_contribution @ teacher_contribution.T,
            )
        )
    return {
        "binding": torch.stack(binding_terms).mean(),
        "key_relation": torch.stack(key_terms).mean(),
        "value_relation": torch.stack(value_terms).mean(),
        "readout_relation": torch.stack(contribution_terms).mean(),
        "layers": len(binding_terms),
        "spans": len(SPAN_NAMES),
        "transport_coverage": 1.0,
    }


def detached_metrics(metrics):
    return {
        key: float(value.detach().cpu()) if torch.is_tensor(value) else value
        for key, value in metrics.items()
    }


__all__ = [
    "SPAN_NAMES",
    "detached_metrics",
    "identity_token_transport",
    "project_route_to_teacher",
    "span_relation_losses",
    "transport_alignment_losses",
]
