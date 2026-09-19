from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F

from common import cuda
from kv_protocol import differentiable_cache


def answer_logits(
    model,
    sample: dict[str, Any],
    pre_key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    official_cache=None,
    question_only: bool = False,
):
    if official_cache is not None and pre_key is not None:
        raise ValueError("provide either official_cache or manual KV, not both")
    target = sample["answer_token_ids"]
    if question_only:
        prompt = sample["question_only_ids"]
        prefix_length = 0
        cache = None
    elif official_cache is not None:
        prompt = sample["question_suffix_ids"]
        prefix_length = sample["context_length"]
        cache = official_cache
    elif pre_key is not None:
        prompt = sample["question_suffix_ids"]
        prefix_length = pre_key.shape[1]
        cache = differentiable_cache(model, pre_key, value)
    else:
        prompt = sample["full_prompt_ids"]
        prefix_length = 0
        cache = None

    current = prompt + target[:-1]
    ids = torch.tensor([current], dtype=torch.long, device=cuda())
    positions = torch.arange(prefix_length, prefix_length + len(current), device=cuda()).unsqueeze(0)
    mask = torch.ones(1, prefix_length + len(current), dtype=torch.long, device=cuda())
    output = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        past_key_values=cache,
        use_cache=False,
    )
    start = len(prompt) - 1
    logits = output.logits[0, start:start + len(target)].float()
    gold = torch.tensor(target, dtype=torch.long, device=logits.device)
    return logits, gold


def answer_ce(model, sample, pre_key, value):
    logits, gold = answer_logits(model, sample, pre_key=pre_key, value=value)
    return F.cross_entropy(logits, gold), logits, gold


def suffix_continuation_logits(model, sample, cache=None, pre_key=None, value=None):
    """Logits produced after consuming each Question-suffix token.

    The first row predicts the token after the first consumed suffix token. The
    full-text condition uses the identical prefix+suffix token sequence.
    """
    suffix = sample["question_suffix_ids"]
    if cache is not None:
        past = cache
        prefix = sample["context_length"]
        prompt = suffix
    elif pre_key is not None:
        past = differentiable_cache(model, pre_key, value)
        prefix = pre_key.shape[1]
        prompt = suffix
    else:
        past = None
        prefix = 0
        prompt = sample["full_prompt_ids"]
    ids = torch.tensor([prompt], dtype=torch.long, device=cuda())
    positions = torch.arange(prefix, prefix + len(prompt), device=cuda()).unsqueeze(0)
    mask = torch.ones(1, prefix + len(prompt), dtype=torch.long, device=cuda())
    output = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        past_key_values=past,
        use_cache=False,
    )
    if prefix == 0:
        start = sample["context_length"]
        return output.logits[0, start:start + len(suffix)].float()
    return output.logits[0, :len(suffix)].float()


@torch.no_grad()
def generate(
    model,
    tokenizer,
    sample: dict[str, Any],
    cfg: dict[str, Any],
    pre_key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    official_cache=None,
    question_only: bool = False,
):
    if question_only:
        prompt, prefix, cache = sample["question_only_ids"], 0, None
    elif official_cache is not None:
        prompt, prefix, cache = sample["question_suffix_ids"], sample["context_length"], copy.deepcopy(official_cache)
    elif pre_key is not None:
        prompt, prefix, cache = sample["question_suffix_ids"], pre_key.shape[1], differentiable_cache(model, pre_key, value)
    else:
        prompt, prefix, cache = sample["full_prompt_ids"], 0, None
    ids = torch.tensor([prompt], dtype=torch.long, device=cuda())
    positions = torch.arange(prefix, prefix + len(prompt), device=cuda()).unsqueeze(0)
    mask = torch.ones(1, prefix + len(prompt), dtype=torch.long, device=cuda())
    output = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
    )
    past = output.past_key_values
    next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    generated = []
    next_position = prefix + len(prompt)
    stop_reason = "max_new_tokens"
    for _ in range(cfg["max_new_tokens"]):
        token = int(next_token.item())
        if token == tokenizer.eos_token_id:
            stop_reason = "eos"
            break
        generated.append(token)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=cuda())], dim=1)
        output = model(
            input_ids=next_token,
            attention_mask=mask,
            position_ids=torch.tensor([[next_position]], dtype=torch.long, device=cuda()),
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), generated, stop_reason

