from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from cache_store import cache_path, load_cache
from common import cuda, load_json, load_model, read_jsonl, save_json, seed_all
from kv_protocol import apply_receiver_rope, differentiable_cache
from writers import load_writer, make_writer


class FirstQueryCapture:
    def __init__(self, model, expected_layers):
        self.values, self.handles = {}, []
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._hook(index), with_kwargs=True))
        self.expected_layers = expected_layers

    def _hook(self, index):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:2], -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden).view(shape))
            self.values[index] = query[0, 0].detach()
        return hook

    def result(self):
        if len(self.values) != self.expected_layers:
            raise RuntimeError("incomplete Receiver query capture")
        return torch.stack([self.values[index] for index in range(self.expected_layers)])

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def capture_first_question_query(model, sample, native_key, native_value, cfg):
    capture = FirstQueryCapture(model, cfg["target_layers"])
    first = torch.tensor([[sample["question_suffix_ids"][0]]], dtype=torch.long, device=cuda())
    prefix = native_key.shape[1]
    try:
        model(
            input_ids=first,
            attention_mask=torch.ones(1, prefix + 1, dtype=torch.long, device=cuda()),
            position_ids=torch.tensor([[prefix]], dtype=torch.long, device=cuda()),
            past_key_values=differentiable_cache(model, native_key, native_value),
            use_cache=False,
        )
        return capture.result()
    finally:
        capture.close()


@torch.no_grad()
def attention_output_diagnostics(model, query_pre, predicted_k, predicted_v, native_k, native_v):
    prefix = native_k.shape[1]
    query = apply_receiver_rope(model, query_pre[:, None], [prefix])[:, 0]
    predicted_k = apply_receiver_rope(model, predicted_k)
    native_k = apply_receiver_rope(model, native_k)
    repeat = query.shape[1] // predicted_k.shape[2]
    predicted_k = predicted_k.repeat_interleave(repeat, dim=2)
    predicted_v = predicted_v.repeat_interleave(repeat, dim=2)
    native_k = native_k.repeat_interleave(repeat, dim=2)
    native_v = native_v.repeat_interleave(repeat, dim=2)
    predicted_logits = torch.einsum("lhd,lthd->lht", query, predicted_k) / math.sqrt(query.shape[-1])
    native_logits = torch.einsum("lhd,lthd->lht", query, native_k) / math.sqrt(query.shape[-1])
    predicted_probability = predicted_logits.float().softmax(-1)
    native_probability = native_logits.float().softmax(-1)
    predicted_output = torch.einsum("lht,lthd->lhd", predicted_probability.to(predicted_v.dtype), predicted_v)
    native_output = torch.einsum("lht,lthd->lhd", native_probability.to(native_v.dtype), native_v)
    route_kl = (
        native_probability * (native_probability.clamp_min(1e-12).log() - predicted_probability.clamp_min(1e-12).log())
    ).sum(-1).mean(-1)
    cosine = F.cosine_similarity(predicted_output.float().flatten(1), native_output.float().flatten(1), dim=1)
    return [
        {"layer": layer, "route_kl": route_kl[layer].item(), "attention_output_cosine": cosine[layer].item()}
        for layer in range(native_k.shape[0])
    ]


def run(cfg, writer_kind, checkpoint):
    sample = read_jsonl(Path(cfg["work_dir"]) / "artifacts" / "manifests" / "validation.jsonl")[0]
    source = load_cache(cache_path(cfg, "source17", "validation", sample["id"]), sample)
    target = load_cache(cache_path(cfg, "target4", "validation", sample["id"]), sample)
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    writer = make_writer(writer_kind, cfg).to(cuda()).eval()
    load_writer(checkpoint, writer)
    source_k, source_v = source["pre_key"].to(cuda()), source["value"].to(cuda())
    native_k, native_v = target["pre_key"].to(cuda()), target["value"].to(cuda())
    with torch.no_grad():
        predicted_k, predicted_v = writer(source_k, source_v)
        query = capture_first_question_query(model, sample, native_k, native_v, cfg)
        rows = attention_output_diagnostics(model, query, predicted_k, predicted_v, native_k, native_v)
    destination = Path(cfg["work_dir"]) / "artifacts" / "diagnostics" / writer_kind
    save_json(destination / "attention_output.json", {
        "sample_id": sample["id"],
        "checkpoint": checkpoint,
        "diagnostic_only_not_in_loss": True,
        "hard_gate_used": False,
        "layers": rows,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--writer", choices=("d0", "d1", "d2"), required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    run(cfg, args.writer, args.checkpoint)


if __name__ == "__main__":
    main()
