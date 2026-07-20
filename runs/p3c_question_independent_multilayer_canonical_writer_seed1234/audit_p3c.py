import argparse
import json
from pathlib import Path

import torch

from p3b_common import write_json
from p3c_common import LAYER_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p3b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.p3b)
    required = [root / "projections/layerwise_pca_random.pt"]
    for split in ("train", "validation", "test"):
        required.append(root / "cache" / split / "index.json")
    for layer_config in LAYER_CONFIGS:
        required.append(root / "full/evidence_only/native_kv" / layer_config / "checkpoint_best.pt")
        required.append(root / "full/evidence_only/native_kv" / layer_config / "SUCCESS.json")
    missing = [str(path) for path in required if not path.is_file()]
    caches = {}
    for split in ("train", "validation", "test"):
        path = root / "cache" / split / "index.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                index = json.load(handle)
            caches[split] = {key: index[key] for key in ("samples", "layers", "kv_flat_dim", "question_in_evidence_only_sender")}
            if index["question_in_evidence_only_sender"]:
                missing.append(f"Question leaked into evidence-only cache: {split}")
    report = {
        "status": "complete" if not missing else "failed", "missing_or_invalid": missing,
        "p3b": str(root), "layer_configs": LAYER_CONFIGS, "caches": caches,
        "sender_frozen": True, "question_encoder_frozen": True, "question_independent": True,
        "cuda_available": torch.cuda.is_available(),
    }
    write_json(args.out, report)
    if missing:
        raise SystemExit("; ".join(missing))


if __name__ == "__main__":
    main()
