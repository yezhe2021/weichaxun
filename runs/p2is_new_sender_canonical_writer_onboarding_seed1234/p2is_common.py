import hashlib
import json
import math
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
P2IW_ROOT = Path(os.environ.get("P2IW_ROOT", ROOT.parent / "p2iw_token_preserving_canonical_writer_qwen3_8b_seed1234"))
P2IR_ROOT = Path(os.environ.get("P2IR_ROOT", ROOT.parent / "p2ir_token_preserving_canonical_reader_onboarding_seed1234"))
for dependency in (P2IW_ROOT, P2IR_ROOT):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from p2iw_common import PairCache, file_sha256
from p2ir_common import (
    canonical_to, extract_answer, fixed_negative, generate, load_receiver, normalize_answer,
    pack_answer, parse_dtype, resolve_device, seed_everything, state_sha256,
    student_prefixed_prompt, write_json, write_jsonl,
)
from p2ir_reader import TokenCanonicalReader, full_attention_layers


class LowRankResidual(nn.Module):
    def __init__(self, dim=256, rank=64):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False); self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5)); nn.init.zeros_(self.up.weight)
    def forward(self, value):
        return self.up(F.silu(self.down(value)))


class Sender4CanonicalWriter(nn.Module):
    def __init__(self, ridge_state, dim=256, rank=64, freeze_base=True):
        super().__init__(); self.dim = int(dim); self.rank = int(rank)
        self.key_base = nn.Linear(1024, dim); self.value_base = nn.Linear(1024, dim)
        with torch.no_grad():
            self.key_base.weight.copy_(ridge_state["key"]["weight"].T)
            self.key_base.bias.copy_(ridge_state["key"]["bias"])
            self.value_base.weight.copy_(ridge_state["value"]["weight"].T)
            self.value_base.bias.copy_(ridge_state["value"]["bias"])
        self.freeze_base = bool(freeze_base)
        if freeze_base:
            for parameter in list(self.key_base.parameters()) + list(self.value_base.parameters()):
                parameter.requires_grad_(False)
        self.key_input_norm = nn.LayerNorm(dim); self.value_input_norm = nn.LayerNorm(dim)
        self.fusion = nn.Linear(2 * dim, dim)
        with torch.no_grad():
            self.fusion.weight.zero_(); self.fusion.bias.zero_()
            self.fusion.weight[:, :dim].copy_(0.5 * torch.eye(dim)); self.fusion.weight[:, dim:].copy_(0.5 * torch.eye(dim))
        self.shared_adapter = LowRankResidual(dim, rank); self.shared_norm = nn.LayerNorm(dim)
        self.key_adapter = LowRankResidual(dim, rank); self.value_adapter = LowRankResidual(dim, rank)
        self.key_norm = nn.LayerNorm(dim); self.value_norm = nn.LayerNorm(dim)
        self.key_log_scale = nn.Parameter(torch.zeros(dim)); self.value_log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, key_flat, value_flat):
        key0_raw = self.key_base(key_flat.float()); value0_raw = self.value_base(value_flat.float())
        key0 = self.key_input_norm(key0_raw); value0 = self.value_input_norm(value0_raw)
        shared0 = self.fusion(torch.cat((key0, value0), dim=-1))
        shared = self.shared_norm(shared0 + self.shared_adapter(shared0))
        keys = self.key_norm(key0_raw + self.key_adapter(shared)) * self.key_log_scale.exp()
        values = self.value_norm(value0_raw + self.value_adapter(shared)) * self.value_log_scale.exp()
        return {"keys": keys, "values": values, "shared": shared}

    def trainable_nonbase_parameters(self):
        return [parameter for name, parameter in self.named_parameters() if not name.startswith(("key_base.", "value_base."))]


def writer_from_checkpoint(ridge_path, checkpoint_path, device):
    ridge = torch.load(ridge_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    writer = Sender4CanonicalWriter(ridge, **checkpoint["writer_config"]).to(device)
    writer.load_state_dict(checkpoint["writer"])
    return writer, checkpoint


def anchor_terms(output, target):
    losses = {}
    losses["cosine"] = 0.5 * (
        (1.0 - F.cosine_similarity(output["keys"], target["keys"].float(), dim=-1)).mean()
        + (1.0 - F.cosine_similarity(output["values"], target["values"].float(), dim=-1)).mean()
    )
    losses["mse"] = 0.5 * (
        F.mse_loss(F.layer_norm(output["keys"], (256,)), F.layer_norm(target["keys"].float(), (256,)))
        + F.mse_loss(F.layer_norm(output["values"], (256,)), F.layer_norm(target["values"].float(), (256,)))
    )
    def graph(value):
        normalized = F.normalize(value.float(), dim=-1); return normalized @ normalized.T
    losses["relation"] = 0.5 * (
        F.mse_loss(graph(output["keys"]), graph(target["keys"]))
        + F.mse_loss(graph(output["values"]), graph(target["values"]))
    )
    losses["total"] = losses["cosine"] + losses["mse"] + 0.05 * losses["relation"]
    return losses


def load_aligned_pair(q4_cache, old_cache, index):
    q4, old = q4_cache.load(index), old_cache.load(index)
    if q4["base"]["pair_id"] != old["base"]["pair_id"]:
        raise RuntimeError("Qwen3-4B Native cache and old Canonical cache are not aligned")
    return q4, old


def memory_from_output(output, answer_mask):
    return {
        "keys": output["keys"], "values": output["values"],
        "mask": torch.ones(output["keys"].shape[0], dtype=torch.bool, device=output["keys"].device),
        "answer_token_mask": answer_mask.to(device=output["keys"].device, dtype=torch.bool),
    }


def gradient_vector(parameters):
    chunks = []
    for parameter in parameters:
        chunks.append(torch.zeros_like(parameter, device="cpu").flatten() if parameter.grad is None else parameter.grad.detach().float().cpu().flatten())
    return torch.cat(chunks)


def assign_gradient(parameters, vector, device):
    offset = 0
    for parameter in parameters:
        size = parameter.numel(); parameter.grad = vector[offset:offset + size].view_as(parameter).to(device=device, dtype=parameter.dtype); offset += size
    if offset != vector.numel():
        raise RuntimeError("Gradient vector length mismatch")


def cosine_between(left, right):
    return float(F.cosine_similarity(left.float(), right.float(), dim=0).cpu()) if left.norm() > 0 and right.norm() > 0 else 0.0

