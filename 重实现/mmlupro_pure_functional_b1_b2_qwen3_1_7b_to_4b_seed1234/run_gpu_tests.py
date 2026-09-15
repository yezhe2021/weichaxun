from __future__ import annotations

import os
from pathlib import Path

import torch

from common import cuda, load_json, load_model, read_jsonl
from training import functional_loss, training_sample
from writers import make_writer


def test_pure_functional_gradient_path(config: str, mode: str) -> None:
    cfg = load_json(config)
    sample = training_sample(read_jsonl(Path(cfg["work_dir"]) / "artifacts/manifests/train.jsonl")[0])
    writer = make_writer("d2", cfg).to(cuda()).train()
    receiver = load_model(cfg["model_4b"], cfg, frozen=True)
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
    for mode in ("final", "all"):
        test_pure_functional_gradient_path(config, mode)
        print(f"PASS GPU pure-functional gradient path: {mode}", flush=True)


if __name__ == "__main__":
    main()
