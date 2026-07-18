import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from canonical_modules import CanonicalEvidenceWriter
from p2i_common import LazyPairCache, native_to, parse_dtype, resolve_device, state_sha256


def main():
    parser = argparse.ArgumentParser(description="Freeze P2-I Writer outputs into forkable Canonical Evidence-KV")
    parser.add_argument("--sender-index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    source = LazyPairCache(args.sender_index, capacity=1)
    count = min(len(source), args.max_pairs) if args.max_pairs > 0 else len(source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    interface = checkpoint["interface"]
    geometry = checkpoint["writer_geometry"]
    writer = CanonicalEvidenceWriter(
        geometry["sender_layers"],
        geometry["sender_heads"],
        geometry["sender_head_dim"],
        interface["slots"],
        interface["canonical_dim"],
        geometry["atom_dim"],
    ).to(device).eval()
    writer.load_state_dict(checkpoint["writer"])
    for parameter in writer.parameters():
        parameter.requires_grad_(False)
    writer_hash = state_sha256(writer.state_dict())
    if writer_hash != checkpoint["writer_sha256"]:
        raise RuntimeError("Checkpoint Writer hash mismatch")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    pair_files = []
    with torch.inference_mode():
        for index in tqdm(range(count), desc="cache_canonical_evidence"):
            pair = source.load(index)
            examples = []
            for variant in ("base", "counterfactual"):
                row = pair[variant]
                native = native_to(row["memory"], device, dtype)
                canonical = writer(native, output_dtype=torch.float16)
                memory = {
                    "keys": canonical["keys"].cpu(),
                    "values": canonical["values"].cpu(),
                }
                if "answer_slot_mass" in canonical:
                    memory["answer_slot_mass"] = canonical["answer_slot_mass"].half().cpu()
                examples.append({**{key: value for key, value in row.items() if key != "memory"}, "memory": memory})
            filename = f"pair_{index:05d}.pt"
            torch.save({"pair_id": examples[0]["pair_id"], "examples": examples}, output / filename)
            pair_files.append(
                {
                    "pair_id": examples[0]["pair_id"],
                    "file": filename,
                    "base_answer": examples[0]["answer"],
                    "counterfactual_answer": examples[1]["answer"],
                }
            )

    metadata = {
        "format_version": 1,
        "interface": "receiver_independent_canonical_evidence_kv",
        "coordinate_system": "none",
        "slots": interface["slots"],
        "canonical_dim": interface["canonical_dim"],
        "writer_sha256": writer_hash,
        "writer_checkpoint": args.checkpoint,
        "source_index": args.sender_index,
        "pairs": count,
        "pair_files": pair_files,
    }
    with open(output / "index.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    with open(output / "CACHE_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", **metadata}, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
