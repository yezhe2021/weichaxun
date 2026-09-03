from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from common import LABELS, cuda
from kv_protocol import differentiable_cache


@dataclass
class ReceiverTrajectory:
    hidden: tuple[torch.Tensor, ...]
    logits: torch.Tensor
    choice_logits: torch.Tensor


class DecoderLayerCapture:
    """Capture exact outputs of all frozen Qwen3 decoder blocks without detaching student autograd."""

    def __init__(self, model):
        self.states: dict[int, torch.Tensor] = {}
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(model.model.layers)
        ]

    def _hook(self, index):
        def hook(module, args, output):
            self.states[index] = output[0] if isinstance(output, tuple) else output
        return hook

    def result(self, expected_layers):
        if len(self.states) != expected_layers:
            raise RuntimeError(f"captured {len(self.states)} decoder layers; expected {expected_layers}")
        return tuple(self.states[index] for index in range(expected_layers))

    def close(self):
        for handle in self.handles:
            handle.remove()


def _condition_inputs(sample, condition, pre_key, value, official_cache, model):
    if condition == "options_first_full_text":
        return sample["full_prompt_ids"], 0, None, sample["context_length"]
    if condition == "standard_full_text":
        return sample["standard_prompt_ids"], 0, None, 0
    if condition == "question_only":
        return sample["question_only_ids"], 0, None, 0
    if condition != "split_cache":
        raise ValueError(condition)
    if official_cache is not None and pre_key is not None:
        raise ValueError("provide official_cache or manual pre-RoPE KV, not both")
    if official_cache is not None:
        return sample["question_suffix_ids"], sample["context_length"], copy.deepcopy(official_cache), 0
    if pre_key is None or value is None:
        raise ValueError("split_cache requires official_cache or pre_key/value")
    return sample["question_suffix_ids"], int(pre_key.shape[1]), differentiable_cache(model, pre_key, value), 0


def trajectory(
    model,
    sample: dict[str, Any],
    condition: str = "split_cache",
    pre_key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    official_cache=None,
    output_hidden_states: bool = True,
) -> ReceiverTrajectory:
    token_ids, prefix_length, cache, slice_start = _condition_inputs(
        sample, condition, pre_key, value, official_cache, model
    )
    ids = torch.tensor([token_ids], dtype=torch.long, device=cuda())
    positions = torch.arange(prefix_length, prefix_length + len(token_ids), device=cuda()).unsqueeze(0)
    mask = torch.ones(1, prefix_length + len(token_ids), dtype=torch.long, device=cuda())
    capture = DecoderLayerCapture(model) if output_hidden_states else None
    try:
        output = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=False,
            output_hidden_states=False,
        )
        captured_states = capture.result(len(model.model.layers)) if capture is not None else ()
    finally:
        if capture is not None:
            capture.close()
    suffix_length = sample["suffix_length"] if condition == "options_first_full_text" else len(token_ids)
    logits = output.logits[0, slice_start:slice_start + suffix_length]
    label_ids = torch.tensor(sample["label_token_ids"], dtype=torch.long, device=logits.device)
    choice_logits = output.logits[0, -1].index_select(0, label_ids).float()
    hidden: tuple[torch.Tensor, ...] = ()
    if output_hidden_states:
        hidden = tuple(state[0, slice_start:slice_start + suffix_length] for state in captured_states)
    return ReceiverTrajectory(hidden=hidden, logits=logits, choice_logits=choice_logits)


def prediction(choice_logits: torch.Tensor) -> tuple[int, str]:
    index = int(choice_logits.argmax().item())
    return index, LABELS[index]


def choice_distribution(choice_logits: torch.Tensor) -> torch.Tensor:
    return choice_logits.float().softmax(dim=-1)
