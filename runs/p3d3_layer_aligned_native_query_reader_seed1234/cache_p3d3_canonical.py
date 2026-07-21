import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import MemoryCache, SELECTED_LAYERS, file_sha256, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-cache", required=True); parser.add_argument("--writer", required=True); parser.add_argument("--projections", required=True)
    parser.add_argument("--p3c-code", required=True); parser.add_argument("--out", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); sys.path.insert(0, args.p3c_code)
    from p3c_common import MultiLayerCanonicalWriter
    checkpoint = torch.load(args.writer, map_location="cpu", weights_only=False); config = checkpoint["writer_config"]
    if config["selected_layers"] != SELECTED_LAYERS: raise RuntimeError("P3-C Writer is not the uniform16 protocol")
    projections = torch.load(args.projections, map_location="cpu", weights_only=False)
    writer = MultiLayerCanonicalWriter(projections, config["selected_layers"], config["rank"]).to(args.device).eval(); writer.load_state_dict(checkpoint["writer"])
    for parameter in writer.parameters(): parameter.requires_grad_(False)
    cache = MemoryCache(args.native_cache); native_index = read_json(args.native_cache)
    if native_index["original_layer_indices"] != SELECTED_LAYERS: raise RuntimeError("Native layer order differs from Writer")
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); entries = []
    with torch.inference_mode():
        for index in tqdm(range(len(cache)), desc="p3d3_canonical_cache"):
            payload = cache.load(index); keys, values = [], []
            for local, module in enumerate(writer.layers):
                key, value = module(payload["keys"][local].float().to(args.device), payload["values"][local].float().to(args.device))
                keys.append(key.half().cpu()); values.append(value.half().cpu())
            filename = f"sample_{index:05d}.pt"
            torch.save({"row": payload["row"], "evidence": payload["evidence"], "keys": torch.stack(keys), "values": torch.stack(values), "metadata": payload["metadata"]}, output / filename)
            entries.append({"id": payload["row"]["id"], "file": filename, "answer": payload["row"]["answer"]})
    result = {"status": "complete", "entries": entries, "samples": len(entries), "layers": 16, "original_layer_indices": SELECTED_LAYERS,
              "memory_dim": 256, "question_independent": True, "writer_frozen": True, "writer_checkpoint": args.writer,
              "writer_checkpoint_sha256": file_sha256(args.writer), "native_cache": args.native_cache}
    write_json(output / "index.json", result); write_json(output / "SUCCESS.json", result)


if __name__ == "__main__": main()
