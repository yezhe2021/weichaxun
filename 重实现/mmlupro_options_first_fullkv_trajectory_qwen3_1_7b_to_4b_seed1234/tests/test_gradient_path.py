from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from common import cuda, load_json, load_model, read_jsonl
from training import functional_components, weighted_total
from writers import make_writer


CONFIG = os.environ.get("MMLUPRO_TRAJECTORY_CONFIG")
pytestmark = pytest.mark.skipif(not CONFIG, reason="set MMLUPRO_TRAJECTORY_CONFIG to run the GPU integration test")


def test_trajectory_loss_reaches_writer_but_not_receiver_and_does_not_require_gold():
    cfg = load_json(CONFIG)
    sample = read_jsonl(Path(cfg["work_dir"]) / "artifacts/manifests/train.jsonl")[0]
    sample = {key: value for key, value in sample.items() if key not in {"gold_index", "gold_label"}}
    writer = make_writer("d2", cfg).to(cuda()).train()
    receiver = load_model(cfg["model_4b"], cfg, frozen=True)
    components, _ = functional_components(cfg, receiver, writer, sample, "train", cfg["seed"])
    loss = weighted_total(components, cfg["stage_b"])
    loss.backward()
    assert all(parameter.grad is None for parameter in receiver.parameters())
    for name, parameter in writer.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().max() > 0, name
