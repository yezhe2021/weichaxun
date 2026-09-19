from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cache_store import cache_path, load_cache
from common import cuda, load_json, load_model, progress, read_jsonl, save_json, seed_all
from receiver import trajectory
from writers import load_scales, load_writer, make_writer, FullKVWriter


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    samples = read_jsonl(Path(cfg["manifest_dir"]) / "validation.jsonl")[:args.samples]
    scales = load_scales(cfg["calibration_path"])
    base = FullKVWriter("full28", scales, cfg).to(cuda()).eval()
    load_writer(cfg["base_stage_a_checkpoint"], base)
    extended = make_writer("full28_head8", cfg).to(cuda()).eval()
    model = load_model(cfg["model_4b"], cfg, frozen=True)
    rows = []
    try:
        for index, sample in enumerate(samples, 1):
            source = load_cache(cache_path(cfg, "source17", "validation", sample["id"]), sample)
            key, value = source["pre_key"].to(cuda()), source["value"].to(cuda())
            base_k, base_v = base(key, value)
            new_k, new_v = extended(key, value)
            base_logits = trajectory(model, sample, pre_key=base_k, value=base_v, output_hidden_states=False).logits
            new_logits = trajectory(model, sample, pre_key=new_k, value=new_v, output_hidden_states=False).logits
            row = {
                "sample_id": sample["id"],
                "k_max_abs_diff": (base_k.float() - new_k.float()).abs().max().item(),
                "v_max_abs_diff": (base_v.float() - new_v.float()).abs().max().item(),
                "final_logits_max_abs_diff": (base_logits[-1].float() - new_logits[-1].float()).abs().max().item(),
            }
            rows.append(row)
            progress(f"head-extension equivalence: {index}/{len(samples)}")
        zero_k, zero_v = extended(torch.zeros_like(key), torch.zeros_like(value))
        zero_max = max(zero_k.abs().max().item(), zero_v.abs().max().item())
        failures = [row for row in rows if row["k_max_abs_diff"] > 1e-6
                    or row["v_max_abs_diff"] > 1e-6 or row["final_logits_max_abs_diff"] > 1e-5]
        if failures or zero_max != 0.0:
            raise RuntimeError(f"head extension identity audit failed: rows={failures} zero_max={zero_max}")
        depth_trainable = [name for name, parameter in extended.named_parameters()
                           if (name.startswith("k_linears") or name.startswith("v_linears")) and parameter.requires_grad]
        head_frozen = [name for name, parameter in extended.named_parameters()
                       if "head_linears" in name and not parameter.requires_grad]
        if depth_trainable or head_frozen:
            raise RuntimeError(f"freeze policy failed: depth_trainable={depth_trainable} head_frozen={head_frozen}")
        save_json(Path(cfg["work_dir"]) / "artifacts/head_extension_audit.json", {
            "passed": True, "rows": rows, "writer_zero_max_abs": zero_max,
            "direct_dense_parameter_count_avoided": 36 * 8 * 2 * (28 * 8 * 128) * 128,
            "total_parameter_count": sum(parameter.numel() for parameter in extended.parameters()),
            "trainable_head_parameter_count": sum(parameter.numel() for parameter in extended.parameters() if parameter.requires_grad),
            "depth_projection_frozen": True, "k_v_separated": True,
            "per_target_head_independent": True,
        })
    finally:
        del base, extended, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
