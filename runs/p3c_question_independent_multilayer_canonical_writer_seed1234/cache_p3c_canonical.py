import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from p3b_common import file_sha256, write_json
from train_eval_p3b_probe import Cache

from p3c_common import MultiLayerCanonicalWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-cache", required=True)
    parser.add_argument("--writer", required=True)
    parser.add_argument("--projections", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.writer, map_location="cpu", weights_only=False)
    projections = torch.load(args.projections, map_location="cpu", weights_only=False)
    config = checkpoint["writer_config"]
    writer = MultiLayerCanonicalWriter(projections, config["selected_layers"], config["rank"]).to(device)
    writer.load_state_dict(checkpoint["writer"])
    writer.eval()
    for parameter in writer.parameters():
        parameter.requires_grad_(False)
    cache = Cache(args.native_cache)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    entries, lengths = [], []
    with torch.inference_mode():
        for index in tqdm(range(len(cache)), desc=f"p3c_cache_{output.name}"):
            payload = cache.load(index)
            native = payload["modes"]["evidence_only"]
            keys = native["keys"].float().to(device)
            values = native["values"].float().to(device)
            canonical_keys, canonical_values = writer(keys, values)
            filename = f"sample_{index:05d}.pt"
            torch.save({
                "row": payload["row"], "evidence": payload["evidence"],
                "question_state": payload["question_state"],
                "keys": canonical_keys.half().cpu().contiguous(),
                "values": canonical_values.half().cpu().contiguous(),
                "metadata": {
                    "offsets": native["offsets"], "answer_token_spans": native["answer_token_spans"],
                    "token_ids": native["token_ids"], "valid_mask": native["valid_mask"],
                    "support_token_mask": native["support_token_mask"],
                },
            }, output / filename)
            entries.append({"id": payload["row"]["id"], "file": filename, "answer": payload["row"]["answer"]})
            lengths.append(canonical_keys.shape[1])
    result = {
        "status": "complete", "samples": len(entries), "entries": entries,
        "layers": len(config["selected_layers"]), "original_layer_indices": config["selected_layers"],
        "canonical_dim": 256, "max_tokens": max(lengths), "question_independent": True,
        "writer_checkpoint_sha256": file_sha256(args.writer),
    }
    write_json(output / "index.json", result)
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
