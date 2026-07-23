from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_MODULES = ("q_proj", "v_proj", "o_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA target must be nn.Linear, got {type(base)}")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        self.enabled = True

    def forward(self, value):
        base_output = self.base(value)
        if not self.enabled:
            return base_output
        update = F.linear(F.linear(self.dropout(value.float()), self.lora_a), self.lora_b)
        return base_output + update.to(base_output.dtype) * self.scaling


def install_lora(model, layer_indices, rank=8, alpha=16.0, dropout=0.0):
    modules = {}
    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]
        targets = {
            "q_proj": (layer.self_attn, "q_proj"),
            "v_proj": (layer.self_attn, "v_proj"),
            "o_proj": (layer.self_attn, "o_proj"),
            "down_proj": (layer.mlp, "down_proj"),
        }
        for name, (parent, attribute) in targets.items():
            base = getattr(parent, attribute)
            module = LoRALinear(base, rank, alpha, dropout).to(device=base.weight.device)
            setattr(parent, attribute, module)
            modules[f"layers.{layer_index}.{name}"] = module
    return modules


def lora_parameters(modules):
    return [parameter for module in modules.values()
            for parameter in (module.lora_a, module.lora_b)]


def lora_state_dict(modules):
    return {
        name: {"lora_a": module.lora_a.detach().cpu(),
               "lora_b": module.lora_b.detach().cpu()}
        for name, module in modules.items()
    }


def load_lora_state(modules, state):
    if set(modules) != set(state):
        raise RuntimeError("LoRA module set does not match checkpoint")
    with torch.no_grad():
        for name, module in modules.items():
            module.lora_a.copy_(state[name]["lora_a"])
            module.lora_b.copy_(state[name]["lora_b"])


@contextmanager
def lora_enabled(modules, enabled=True):
    previous = {name: module.enabled for name, module in modules.items()}
    for module in modules.values():
        module.enabled = bool(enabled)
    try:
        yield
    finally:
        for name, module in modules.items():
            module.enabled = previous[name]


def lora_diagnostics(modules):
    result = {}
    for name, module in modules.items():
        result[name] = {
            "rank": module.rank, "alpha": module.alpha, "scaling": module.scaling,
            "a_norm": float(module.lora_a.detach().float().norm()),
            "b_norm": float(module.lora_b.detach().float().norm()),
            "effective_update_frobenius": float(
                (module.lora_b.detach().float() @ module.lora_a.detach().float()).norm()
                * module.scaling
            ),
            "base_weight_frobenius": float(module.base.weight.detach().float().norm()),
        }
        result[name]["effective_to_base_norm_ratio"] = (
            result[name]["effective_update_frobenius"] /
            max(result[name]["base_weight_frobenius"], 1e-8)
        )
    return result


def assert_optimizer_boundary(model, reader, modules, optimizer):
    allowed = {id(parameter) for parameter in lora_parameters(modules)}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if allowed != actual:
        raise RuntimeError("Optimizer must contain exactly LoRA A/B parameters")
    if any(parameter.requires_grad for parameter in reader.parameters()):
        raise RuntimeError("C1 Reader is not frozen")
    for parameter in model.parameters():
        if id(parameter) not in allowed and parameter.requires_grad:
            raise RuntimeError("A non-LoRA Receiver parameter is trainable")


def assert_frozen_gradients(model, reader, modules):
    allowed = {id(parameter) for parameter in lora_parameters(modules)}
    for parameter in model.parameters():
        if id(parameter) not in allowed and parameter.grad is not None:
            raise RuntimeError("Gradient reached frozen Receiver parameter")
    if any(parameter.grad is not None for parameter in reader.parameters()):
        raise RuntimeError("Gradient reached frozen C1 Reader")
