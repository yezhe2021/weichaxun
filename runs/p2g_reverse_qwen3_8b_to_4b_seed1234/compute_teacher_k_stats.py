import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Compute per-layer/head Native 8B K RMS for P2-C2")
    parser.add_argument("--teacher-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=512)
    args = parser.parse_args()
    with open(args.teacher_index, encoding="utf-8") as handle:
        index = json.load(handle)
    root = Path(args.teacher_index).parent
    entries = index["pair_files"][: args.max_pairs]
    sum_squares = torch.zeros(index["layers"], index["kv_heads"], dtype=torch.float64)
    counts = torch.zeros(index["layers"], index["kv_heads"], dtype=torch.float64)
    examples = 0
    for entry in tqdm(entries, desc="teacher_k_rms"):
        payload = torch.load(root / entry["file"], map_location="cpu", weights_only=False)
        for example in payload["examples"]:
            for layer, key in enumerate(example["memory"]["keys"]):
                key = key.double()
                sum_squares[layer] += key.square().sum(dim=(1, 2))
                counts[layer] += key.shape[1] * key.shape[2]
            examples += 1
    rms = (sum_squares / counts.clamp_min(1)).sqrt().float()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(rms, output / "teacher_k_rms.pt")
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "teacher_index": args.teacher_index,
                "pairs": len(entries),
                "examples": examples,
                "shape": list(rms.shape),
                "minimum": float(rms.min()),
                "maximum": float(rms.max()),
                "mean": float(rms.mean()),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
