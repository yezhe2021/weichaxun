from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cache_store import cache_path, load_cache
from common import cuda, load_json, load_model, progress, read_jsonl, save_json, seed_all
from receiver import trajectory
from writers import FullKVWriter, load_scales, make_writer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    cfg = load_json(args.config)
    seed_all(cfg["seed"])
    model_configs = {}
    expected_structures = {
        "model_4b": {"num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936},
        "model_1_7b": {"num_hidden_layers": 28, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936},
    }
    for name, expected in expected_structures.items():
        with open(Path(cfg[name]) / "config.json", encoding="utf-8") as handle:
            model_configs[name] = json.load(handle)
        actual = model_configs[name]
        mismatch = {key: (actual.get(key), value) for key, value in expected.items() if actual.get(key) != value}
        if mismatch:
            raise RuntimeError(f"{name} structure mismatch: {mismatch}")
    samples = read_jsonl(Path(cfg["manifest_dir"]) / "validation.jsonl")[:args.samples]
    tokenizer17 = AutoTokenizer.from_pretrained(cfg["model_1_7b"], local_files_only=True, trust_remote_code=True)
    tokenizer4 = AutoTokenizer.from_pretrained(cfg["model_4b"], local_files_only=True, trust_remote_code=True)
    for sample in samples:
        ids17 = tokenizer17(sample["prefix_text"], add_special_tokens=False).input_ids
        ids4 = tokenizer4(sample["prefix_text"], add_special_tokens=False).input_ids
        if ids17 != ids4 or ids17 != sample["prefix_token_ids"]:
            raise RuntimeError(f"tokenizer/persisted prefix mismatch: {sample['id']}")
    scales = load_scales(cfg["calibration_path"])
    base = FullKVWriter("full36", scales, cfg).to(cuda()).eval()
    writer = make_writer("full36_head8", cfg).to(cuda()).eval()
    receiver = load_model(cfg["receiver_model"], cfg, frozen=True)
    rows = []
    try:
        for index, sample in enumerate(samples, 1):
            source = load_cache(cache_path(cfg, cfg["sender_cache_family"], "validation", sample["id"]), sample)
            key, value = source["pre_key"].to(cuda()), source["value"].to(cuda())
            base_k, base_v = base(key, value)
            head_k, head_v = writer(key, value)
            base_logits = trajectory(receiver, sample, pre_key=base_k, value=base_v, output_hidden_states=False).logits[-1]
            head_logits = trajectory(receiver, sample, pre_key=head_k, value=head_v, output_hidden_states=False).logits[-1]
            rows.append({
                "sample_id": sample["id"],
                "k_max_abs_diff": (base_k.float() - head_k.float()).abs().max().item(),
                "v_max_abs_diff": (base_v.float() - head_v.float()).abs().max().item(),
                "final_logits_max_abs_diff": (base_logits.float() - head_logits.float()).abs().max().item(),
            })
            progress(f"direct Head-8 identity audit: {index}/{len(samples)}")
        zero_k, zero_v = writer(torch.zeros_like(key), torch.zeros_like(value))
        zero_max = max(zero_k.abs().max().item(), zero_v.abs().max().item())
        failures = [row for row in rows if max(row["k_max_abs_diff"], row["v_max_abs_diff"]) > 1e-6
                    or row["final_logits_max_abs_diff"] > 1e-5]
        if failures or zero_max != 0.0:
            raise RuntimeError(f"identity/zero audit failed: failures={failures} zero={zero_max}")
        save_json(Path(cfg["work_dir"]) / "artifacts/direct_head8_audit.json", {
            "passed": True, "rows": rows, "writer_zero_max_abs": zero_max,
            "model_configs": model_configs,
            "tokenizer_prefix_ids_identical": True,
            "full_context_pre_rope_k_and_v": True,
            "total_parameter_count": sum(p.numel() for p in writer.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in writer.parameters() if p.requires_grad),
            "direct_dense_parameter_count_avoided": cfg["target_layers"] * cfg["num_kv_heads"] * 2 * (cfg["source_layers"] * cfg["num_kv_heads"] * cfg["head_dim"]) * cfg["head_dim"],
        })
    finally:
        del base, writer, receiver
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
