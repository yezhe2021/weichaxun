from __future__ import annotations

import os

import torch

from common import cuda, load_json, load_model
from training import functional_loss, load_training_split
from writers import make_writer


def test_pure_functional_gradient_path(config: str, mode: str) -> None:
    cfg = load_json(config)
    sample = load_training_split(cfg, "train")[0]
    writer = make_writer(cfg["writer_kind"], cfg).to(cuda()).train()
    receiver = load_model(cfg[cfg["receiver_model_key"]], cfg, frozen=True)
    try:
        loss = functional_loss(cfg, receiver, writer, sample, "train", mode)
        loss.backward()
        assert all(parameter.grad is None for parameter in receiver.parameters())
        for name, parameter in writer.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                assert parameter.grad.abs().max() > 0, name
    finally:
        del writer, receiver
        torch.cuda.empty_cache()


def main() -> None:
    config = os.environ.get("MMLUPRO_FUNCTIONAL_CONFIG", "config.json")
    for mode in ("final",):
        test_pure_functional_gradient_path(config, mode)
        print(f"PASS GPU pure-functional gradient path: {mode}", flush=True)


if __name__ == "__main__":
    main()
