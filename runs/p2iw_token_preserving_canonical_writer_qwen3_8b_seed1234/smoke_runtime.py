import argparse

import torch
import torch.nn.functional as F

from p2iw_common import TokenCanonicalWriter, VariableAttentionProbe
from train_writer import one_pair


def fake_projection(input_dim, output_dim):
    components = torch.linalg.qr(torch.randn(input_dim, output_dim), mode="reduced").Q
    return {"mean": torch.zeros(input_dim), "components": components, "scale": torch.ones(output_dim)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    device = torch.device(args.device)
    projections = {"key": fake_projection(1024, 256), "value": fake_projection(1024, 256)}
    writer = TokenCanonicalWriter(projections, rank=32).to(device)
    probe = VariableAttentionProbe(40).to(device)
    key, value = torch.randn(17, 1024, device=device), torch.randn(17, 1024, device=device)
    output = writer(key, value)
    mask = torch.ones(1, 17, dtype=torch.bool, device=device)
    logits = probe(output["keys"][None], output["values"][None], mask)
    loss = F.cross_entropy(logits, torch.tensor([3], device=device)) + output["shared"].square().mean()
    loss.backward()
    if not all(torch.isfinite(parameter.grad).all() for parameter in list(writer.parameters()) + list(probe.parameters()) if parameter.requires_grad and parameter.grad is not None):
        raise RuntimeError("Smoke test found non-finite gradients")
    writer.eval(); probe.eval()
    with torch.inference_mode():
        first = probe(output["keys"][None], output["values"][None], mask)
        order = torch.randperm(17, device=device)
        second = probe(output["keys"][None, order], output["values"][None, order], mask)
    if not torch.allclose(first, second, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Synchronous token permutation changed set-attention output")
    hidden_projection = fake_projection(4096, 256)
    full_projection = {"key": projections["key"], "value": projections["value"], "hidden": hidden_projection}
    answer_base = torch.zeros(17, dtype=torch.bool); answer_base[-1] = True
    answer_cf = torch.zeros(18, dtype=torch.bool); answer_cf[-1] = True
    pair = {
        "base": {"key_flat": torch.randn(17, 1024), "value_flat": torch.randn(17, 1024), "hidden": torch.randn(17, 4096), "answer_mask": answer_base, "answer": "city0"},
        "counterfactual": {"key_flat": torch.randn(18, 1024), "value_flat": torch.randn(18, 1024), "hidden": torch.randn(18, 4096), "answer_mask": answer_cf, "answer": "city1"},
        "_stable_alignment": torch.tensor([[index, index] for index in range(16)]),
    }
    writer.zero_grad(set_to_none=True); probe.zero_grad(set_to_none=True)
    weights = {"content": 1.0, "stable": 0.2, "change": 0.2, "classification": 0.1, "switch": 0.05}
    paired_loss, _, _, _, _ = one_pair(writer, probe, pair, {"city0": 0, "city1": 1}, full_projection, device, weights, 0.35)
    paired_loss.backward()
    if not torch.isfinite(paired_loss):
        raise RuntimeError("Paired Writer objective is non-finite")
    print("P2-I-W runtime smoke passed")


if __name__ == "__main__":
    main()
