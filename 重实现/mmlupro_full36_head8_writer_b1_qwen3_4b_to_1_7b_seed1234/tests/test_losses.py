from __future__ import annotations

import torch

from receiver import DecoderLayerCapture
from evaluate import matched_control_comparison
from training import behavioral_kl_loss, epoch_permutation, gradient_was_clipped, hidden_trajectory_loss, training_sample


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
    settings = {"functional_last_n_tokens": 4, "temperature": 1.0}
    assert abs(behavioral_kl_loss(logits, logits.clone(), settings, "all").item()) < 1e-6


def test_last_n_kl_ignores_earlier_tokens():
    teacher = torch.randn(9, 101)
    student = teacher.clone()
    student[:5] += torch.randn_like(student[:5])
    settings = {"functional_last_n_tokens": 4, "temperature": 1.0}
    assert abs(behavioral_kl_loss(student, teacher, settings, "last_n").item()) < 1e-6


def test_final_kl_ignores_all_earlier_suffix_positions():
    teacher = torch.randn(9, 101)
    student = teacher.clone()
    student[:-1] += torch.randn_like(student[:-1])
    settings = {"functional_last_n_tokens": 4, "temperature": 1.0}
    assert abs(behavioral_kl_loss(student, teacher, settings, "final").item()) < 1e-6
    assert behavioral_kl_loss(student, teacher, settings, "all").item() > 0


def test_training_sample_physically_removes_gold_fields():
    sample = {
        "id": "x", "gold_index": 2, "gold_label": "C", "answer": "C",
        "answer_index": 2, "cot_content": "forbidden", "question_suffix_ids": [1, 2],
    }
    sanitized = training_sample(sample)
    assert sanitized == {"id": "x", "question_suffix_ids": [1, 2]}


def test_shuffled_comparison_uses_only_matched_correct_samples():
    rows = [
        {"sample_id": "a", "condition": "correct", "accuracy": 1.0, "native_agreement": 1.0, "native_to_condition_choice_kl": 0.1},
        {"sample_id": "b", "condition": "correct", "accuracy": 0.0, "native_agreement": 0.0, "native_to_condition_choice_kl": 1.0},
        {"sample_id": "b", "condition": "shuffled", "accuracy": 1.0, "native_agreement": 1.0, "native_to_condition_choice_kl": 2.0},
    ]
    result = matched_control_comparison(rows, "correct", "shuffled")
    assert result["matched_count"] == 1
    assert result["correct_accuracy"] == 0.0
    assert result["shuffled_accuracy"] == 1.0
    assert result["correct_minus_shuffled_accuracy"] == -1.0


def test_epoch_sampler_is_without_replacement_and_changes_each_epoch():
    first = epoch_permutation(1024, 1234, 1)
    second = epoch_permutation(1024, 1234, 2)
    assert len(first) == len(set(first)) == 1024
    assert set(first) == set(range(1024))
    assert first != second


def test_clip_rate_decision_uses_pre_clip_norm():
    assert not gradient_was_clipped(1.0, 1.0)
    assert gradient_was_clipped(1.000001, 1.0)


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
