import argparse
from pathlib import Path

from p3d_common import file_sha256, read_json, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True); parser.add_argument("--model4", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); protocol = read_json(args.protocol); errors = []
    for name in ("canonical16", "canonical36"):
        item = protocol[name]
        if file_sha256(item["writer_checkpoint"]) != item["writer_sha256"]: errors.append(f"{name} Writer hash mismatch")
        for split, path in item["canonical_cache"].items():
            index = read_json(path)
            if index["writer_checkpoint_sha256"] != item["writer_sha256"]: errors.append(f"{name}/{split} cache hash mismatch")
            if not index.get("question_independent"): errors.append(f"{name}/{split} is not question-independent")
    config = read_json(Path(args.model4) / "config.json")
    geometry = {"layers": config.get("num_hidden_layers"), "hidden_size": config.get("hidden_size"), "heads": config.get("num_attention_heads")}
    if geometry != {"layers": 36, "hidden_size": 2560, "heads": 32}: errors.append(f"Unexpected Qwen3-4B geometry: {geometry}")
    write_json(args.out, {"status": "complete" if not errors else "failed", "errors": errors, "protocol": protocol, "receiver_geometry": geometry})
    if errors: raise SystemExit("; ".join(errors))


if __name__ == "__main__": main()
