from __future__ import annotations

import torch

from receiver import DecoderLayerCapture
from training import hidden_trajectory_loss, logit_kl_loss


def test_identical_36_layer_hidden_loss_is_zero():
    teacher = tuple(torch.randn(7, 32) for _ in range(36))
    student = tuple(value.clone() for value in teacher)
    loss, metrics, summary = hidden_trajectory_loss(student, teacher, collect_metrics=True)
    assert loss.item() < 1e-7
    assert len(metrics) == 36
    assert all(row["cosine"] > 0.999999 for row in metrics)
    assert summary["hidden_nmse"] == 0.0


def test_hidden_loss_averages_layers_instead_of_flattening():
    teacher = tuple(torch.ones(3, 4) * (layer + 1) for layer in range(36))
    student = tuple(value.clone() for value in teacher)
    student = (student[0] * 2,) + student[1:]
    loss, metrics, _ = hidden_trajectory_loss(student, teacher, collect_metrics=True)
    expected = sum(row["nmse"] + 1.0 - row["cosine"] for row in metrics) / 36
    assert abs(loss.item() - expected) < 1e-6


def test_identical_full_vocabulary_kl_is_zero():
    logits = torch.randn(9, 101)
    settings = {"kl_token_mode": "all", "kl_last_n_tokens": 4, "kl_temperature": 1.0}
    assert abs(logit_kl_loss(logits, logits.clone(), settings).item()) < 1e-6


def test_last_n_kl_ignores_earlier_tokens():
    teacher = torch.randn(9, 101)
    student = teacher.clone()
    student[:5] += torch.randn_like(student[:5])
    settings = {"kl_token_mode": "last_n", "kl_last_n_tokens": 4, "kl_temperature": 1.0}
    assert abs(logit_kl_loss(student, teacher, settings).item()) < 1e-6


def test_decoder_layer_capture_preserves_autograd():
    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(5, 5, bias=False) for _ in range(3)])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()

    model = Model()
    capture = DecoderLayerCapture(model)
    value = torch.randn(2, 5, requires_grad=True)
    output = value
    try:
        for layer in model.model.layers:
            output = layer(output)
        states = capture.result(3)
    finally:
        capture.close()
    assert len(states) == 3
    sum(state.square().mean() for state in states).backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert value.grad.abs().max() > 0
