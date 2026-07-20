import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p2is_common import PairCache, file_sha256, state_sha256, write_json, writer_from_checkpoint


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--q4-index", required=True); parser.add_argument("--ridge", required=True)
    parser.add_argument("--writer-checkpoint", required=True); parser.add_argument("--out", required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args(); device = torch.device(args.device); source = PairCache(args.q4_index, 2)
    writer, checkpoint = writer_from_checkpoint(args.ridge, args.writer_checkpoint, device); writer.eval()
    for parameter in writer.parameters(): parameter.requires_grad_(False)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); entries = []
    with torch.inference_mode():
        count = min(len(source), args.max_pairs) if args.max_pairs > 0 else len(source)
        for index in tqdm(range(count), desc="cache_new_sender_writer"):
            pair = source.load(index); rows = []
            for variant in ("base", "counterfactual"):
                source_row = pair[variant]; memory = writer(source_row["key_flat"].to(device), source_row["value_flat"].to(device))
                rows.append({
                    "pair_id": source_row["pair_id"], "id": source_row["id"], "variant": variant,
                    "question": source_row["question"], "answer": source_row["answer"],
                    "memory": {
                        "keys": memory["keys"].half().cpu(), "values": memory["values"].half().cpu(),
                        "mask": torch.ones(memory["keys"].shape[0], dtype=torch.bool),
                        "answer_token_mask": source_row["answer_mask"].bool(),
                    },
                })
            filename = f"pair_{index:05d}.pt"; torch.save({"pair_id": rows[0]["pair_id"], "variants": rows}, output / filename)
            entries.append({"pair_id": rows[0]["pair_id"], "file": filename, "base_answer": rows[0]["answer"], "counterfactual_answer": rows[1]["answer"]})
    metadata = {
        "format_version": 1, "sender": "qwen3_4b", "pairs": len(entries), "canonical_dim": 256,
        "writer_checkpoint": str(Path(args.writer_checkpoint).resolve()), "writer_checkpoint_sha256": file_sha256(args.writer_checkpoint),
        "writer_state_sha256": state_sha256(writer.state_dict()), "stage": checkpoint.get("stage"), "pair_files": entries,
    }
    write_json(output / "index.json", metadata); write_json(output / "CACHE_SUCCESS.json", {"status": "complete", **metadata})


if __name__ == "__main__": main()
