import argparse

import torch

from p2is_common import Sender4CanonicalWriter, anchor_terms


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); args = parser.parse_args(); device = torch.device(args.device)
    ridge = {
        "key": {"weight": torch.randn(1024, 256) / 32, "bias": torch.zeros(256)},
        "value": {"weight": torch.randn(1024, 256) / 32, "bias": torch.zeros(256)},
    }
    writer = Sender4CanonicalWriter(ridge, rank=32, freeze_base=True).to(device)
    output = writer(torch.randn(13, 1024, device=device), torch.randn(13, 1024, device=device))
    target = {"keys": torch.randn(13, 256, device=device), "values": torch.randn(13, 256, device=device)}
    loss = anchor_terms(output, target)["total"]; loss.backward(retain_graph=True)
    if any(parameter.grad is not None for parameter in list(writer.key_base.parameters()) + list(writer.value_base.parameters())): raise RuntimeError("Frozen ridge base received gradients")
    trainable = [parameter for parameter in writer.parameters() if parameter.requires_grad]
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable): raise RuntimeError("Writer anchor gradient smoke failed")
    for parameter in trainable: parameter.grad = None
    torch.autograd.backward((output["keys"], output["values"]), (torch.randn_like(output["keys"]), torch.randn_like(output["values"])))
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable): raise RuntimeError("Functional memory-gradient smoke failed")
    print("P2-I-S runtime smoke passed")


if __name__ == "__main__": main()
