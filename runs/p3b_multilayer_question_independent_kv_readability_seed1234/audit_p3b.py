import argparse
import json
import shutil
from pathlib import Path

import torch
import transformers

from p3b_common import LAYER_SETS, SOURCES, SENDER_MODES, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config_path = Path(args.model) / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    free_bytes = shutil.disk_usage(args.root).free
    expected = {
        "layers": int(config.get("num_hidden_layers", -1)),
        "attention_heads": int(config.get("num_attention_heads", -1)),
        "kv_heads": int(config.get("num_key_value_heads", -1)),
        "hidden_size": int(config.get("hidden_size", -1)),
        "head_dim": int(config.get("head_dim", config.get("hidden_size", 0) // max(1, config.get("num_attention_heads", 1)))),
    }
    errors = []
    if expected != {"layers": 36, "attention_heads": 32, "kv_heads": 8, "hidden_size": 4096, "head_dim": 128}:
        errors.append(f"Unexpected Qwen3-8B geometry: {expected}")
    for path in (args.train_source, args.test_source, config_path):
        if not Path(path).is_file():
            errors.append(f"Missing file: {path}")
    if free_bytes < 150 * 1024**3:
        errors.append(f"Less than 150 GiB free disk: {free_bytes / 1024**3:.1f} GiB")
    report = {
        "status": "complete" if not errors else "failed",
        "errors": errors,
        "model_geometry": expected,
        "free_disk_gib": free_bytes / 1024**3,
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sender_modes": list(SENDER_MODES),
        "sources": list(SOURCES),
        "layer_sets": LAYER_SETS,
        "question_independent_contract": "evidence_only sender text contains no question field",
    }
    write_json(args.out, report)
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
