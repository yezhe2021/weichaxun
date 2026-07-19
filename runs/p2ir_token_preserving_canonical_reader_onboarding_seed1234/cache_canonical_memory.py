import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p2ir_common import PairCache, P2IW_ROOT, TokenCanonicalWriter, file_sha256, state_sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-index", required=True); parser.add_argument("--writer-checkpoint", required=True)
    parser.add_argument("--projections", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    source = PairCache(args.token_index, capacity=2)
    projection_bundle = torch.load(args.projections, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.writer_checkpoint, map_location="cpu", weights_only=False)
    writer = TokenCanonicalWriter(projection_bundle["pca"], **checkpoint["writer_config"]).to(device).eval()
    writer.load_state_dict(checkpoint["writer"])
    for parameter in writer.parameters():
        parameter.requires_grad_(False)
    writer_file_hash = file_sha256(args.writer_checkpoint); writer_state_hash = state_sha256(writer.state_dict())
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    entries = []
    with torch.inference_mode():
        for index in tqdm(range(len(source)), desc="cache_frozen_p2iw_writer"):
            pair = source.load(index); variants = []
            for variant in ("base", "counterfactual"):
                row = pair[variant]
                written = writer(row["key_flat"].to(device), row["value_flat"].to(device))
                variants.append({
                    "pair_id": row["pair_id"], "id": row["id"], "variant": variant,
                    "question": row["question"], "answer": row["answer"],
                    "memory": {
                        "keys": written["keys"].half().cpu().contiguous(),
                        "values": written["values"].half().cpu().contiguous(),
                        "mask": torch.ones(written["keys"].shape[0], dtype=torch.bool),
                        "answer_token_mask": row["answer_mask"].bool().cpu(),
                    },
                })
            filename = f"pair_{index:05d}.pt"; torch.save({"pair_id": variants[0]["pair_id"], "variants": variants}, output / filename)
            entries.append({
                "pair_id": variants[0]["pair_id"], "file": filename,
                "base_answer": variants[0]["answer"], "counterfactual_answer": variants[1]["answer"],
            })
    metadata = {
        "format_version": 1, "interface": "token_preserving_canonical_evidence_kv",
        "pairs": len(entries), "canonical_dim": 256, "variable_token_axis": True,
        "writer_checkpoint": str(Path(args.writer_checkpoint).resolve()),
        "writer_checkpoint_sha256": writer_file_hash, "writer_state_sha256": writer_state_hash,
        "source_token_index": str(Path(args.token_index).resolve()), "pair_files": entries,
    }
    write_json(output / "index.json", metadata); write_json(output / "CACHE_SUCCESS.json", {"status": "complete", **metadata})


if __name__ == "__main__":
    main()
