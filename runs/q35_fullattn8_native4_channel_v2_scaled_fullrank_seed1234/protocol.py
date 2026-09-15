from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from v2_common import load_json, progress, rows_for, save_json
from writer_v2 import ScaledFullRankWriter


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--action", choices=("selftest", "audit"), required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.action == "selftest":
        scales = {
            "source_k": torch.ones(8, 1024),
            "source_v": torch.ones(8, 1024),
            "target_k": torch.ones(36, 1024),
            "target_v": torch.ones(36, 1024),
        }
        for variant in ("v0", "v1", "v2"):
            writer = ScaledFullRankWriter(cfg, scales, variant)
            assert not any(
                isinstance(module, torch.nn.Linear) and module.bias is not None
                for module in writer.modules()
            )
            key = torch.randn(8, 3, 4, 256)
            output = writer(key, key)
            assert output[0].shape == (36, 3, 8, 128)
            zero = writer(torch.zeros_like(key), torch.zeros_like(key))
            assert max(x.abs().max().item() for x in zero) == 0
        progress("v2 CPU selftest passed")
        return
    q35 = AutoConfig.from_pretrained(
        cfg["model_q35"], local_files_only=True
    ).text_config
    q4 = AutoConfig.from_pretrained(cfg["model_4b"], local_files_only=True)
    full = [i for i, kind in enumerate(q35.layer_types) if kind == "full_attention"]
    assert full == cfg["q35_full_attention_layers"]
    assert (q35.num_key_value_heads, q35.head_dim) == (4, 256)
    assert (q4.num_hidden_layers, q4.num_key_value_heads, q4.head_dim) == (36, 8, 128)
    tok35 = AutoTokenizer.from_pretrained(cfg["model_q35"], local_files_only=True)
    tok4 = AutoTokenizer.from_pretrained(cfg["model_4b"], local_files_only=True)
    assert len(tok35) != len(tok4)
    reader = Path(cfg["r1_dir"]) / "artifacts" / (
        "smoke" if args.mode == "smoke" else "formal"
    ) / "sparse_reader" / "best.pt"
    report = {
        "samples": {k: len(v) for k, v in rows_for(cfg, args.mode).items()},
        "sender_input": "complete context only; question excluded",
        "source_layers": full,
        "source_state": "k_proj -> k_norm -> pre-RoPE K; native V",
        "excluded": ["DeltaNet pseudo-KV", "DeltaNet recurrent state",
                     "DeltaNet convolution state", "question", "Selector"],
        "reader_checkpoint": str(reader),
        "reader_checkpoint_sha256": digest(reader),
        "receiver_translator": "identity",
        "q_o_lora": "enabled and frozen during training",
        "hard_gate": None,
    }
    save_json(Path(cfg["work_dir"]) / "artifacts" / args.mode / "protocol_audit.json", report)
    progress(f"{args.mode}: v2 protocol audit completed")


if __name__ == "__main__":
    main()
