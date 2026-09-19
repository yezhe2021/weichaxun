from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from cache_store import cache_path, load_cache
from common import cuda, load_json, load_model, read_jsonl
from receiver import answer_ce
from writers import make_writer


CONFIG = os.environ.get("FULLKV_CONFIG")
pytestmark = pytest.mark.skipif(not CONFIG, reason="set FULLKV_CONFIG to run the GPU integration test explicitly")


def test_ce_reaches_every_writer_parameter_but_not_receiver_parameters():
    cfg = load_json(CONFIG)
    sample = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "manifests" / "train.jsonl")[0]
    source = load_cache(cache_path(cfg, "source17", "train", sample["id"]), sample)
    writer = make_writer("d2", cfg).to(cuda()).train()
    receiver = load_model(cfg["model_4b"], cfg, frozen=True)
    predicted_k, predicted_v = writer(source["pre_key"].to(cuda()), source["value"].to(cuda()))
    assert predicted_k.requires_grad and predicted_v.requires_grad
    loss, _, _ = answer_ce(receiver, sample, predicted_k, predicted_v)
    loss.backward()
    assert all(parameter.grad is None for parameter in receiver.parameters())
    for name, parameter in writer.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().max() > 0, name

