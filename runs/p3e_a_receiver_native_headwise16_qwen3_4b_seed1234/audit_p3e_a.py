import argparse
from pathlib import Path

import torch
from transformers import AutoConfig

from p3d3_common import SELECTED_LAYERS, file_sha256, load_jsonl, read_json, write_json


def audit_cache(path, expected_samples, data_path, errors):
    index = read_json(path); rows = load_jsonl(data_path)
    if index.get("samples") != expected_samples: errors.append(f"Expected {expected_samples} samples in {path}")
    if index.get("original_layer_indices") != SELECTED_LAYERS: errors.append(f"Layer order mismatch in {path}")
    if index.get("memory_shape") != "[16,T,8,128]" or index.get("kv_heads") != 8 or index.get("head_dim") != 128: errors.append(f"Native head geometry mismatch in {path}")
    if not index.get("question_independent") or index.get("sender_input") != "evidence_only": errors.append(f"Sender input invariant failed in {path}")
    if [entry["id"] for entry in index["entries"]] != [row["id"] for row in rows]: errors.append(f"Data/cache IDs differ in {path}")
    payload = torch.load(Path(path).parent / index["entries"][0]["file"], map_location="cpu", weights_only=False)
    if payload["keys"].ndim != 4 or payload["keys"].shape[0] != 16 or payload["keys"].shape[-2:] != (8, 128): errors.append(f"Payload K shape invalid in {path}")
    if payload["keys"].shape != payload["values"].shape: errors.append(f"K/V shape mismatch in {path}")
    if payload["evidence"] != f"EVIDENCE A\n{payload['row']['evidence_a']}\n\nEVIDENCE B\n{payload['row']['evidence_b']}": errors.append(f"Sender cache includes unexpected text in {path}")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--train-cache", required=True); parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--train-data", required=True); parser.add_argument("--validation-data", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); errors = []; config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    geometry = {"layers": config.num_hidden_layers, "query_heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads,
                "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads), "hidden_size": config.hidden_size}
    if geometry != {"layers": 36, "query_heads": 32, "kv_heads": 8, "head_dim": 128, "hidden_size": 2560}: errors.append(f"Unexpected Qwen3-4B geometry: {geometry}")
    train = audit_cache(args.train_cache, 512, args.train_data, errors); validation = audit_cache(args.validation_cache, 64, args.validation_data, errors)
    if {entry["id"] for entry in train["entries"]} & {entry["id"] for entry in validation["entries"]}: errors.append("Train/validation overlap")
    result = {"status": "complete" if not errors else "failed", "errors": errors, "geometry": geometry, "selected_layers": SELECTED_LAYERS,
              "model_config_sha256": file_sha256(Path(args.model) / "config.json"),
              "invariants": {"same_model_evidence_encoder_and_receiver": True, "native_128d_heads": True, "canonical_projection": False,
                             "fixed_gqa_mapping": "query_head // 4", "sender_sees_question": False}}
    write_json(args.out, result)
    if errors: raise RuntimeError("; ".join(errors))


if __name__ == "__main__": main()
