import argparse
from pathlib import Path

from transformers import AutoConfig

from p3d3_common import SELECTED_LAYERS, file_sha256, load_jsonl, read_json, write_json
from p3e_b_common import SenderNativeHeadwiseCache


def audit_split(index_path, data_path, expected, errors):
    source = read_json(index_path); data = load_jsonl(data_path); cache = SenderNativeHeadwiseCache(index_path)
    if source.get("samples") != expected or len(cache) != expected: errors.append(f"Expected {expected} samples in {index_path}")
    if source.get("original_layer_indices") != SELECTED_LAYERS: errors.append(f"Layer mismatch in {index_path}")
    if source.get("memory_dim") != 1024 or not source.get("question_independent") or source.get("sender_input") != "evidence_only": errors.append(f"Source cache invariants failed in {index_path}")
    if [entry["id"] for entry in source["entries"]] != [row["id"] for row in data]: errors.append(f"Data/cache IDs differ in {index_path}")
    payload = cache.load(0)
    if payload["keys"].shape[0] != 16 or payload["keys"].shape[-2:] != (8, 128): errors.append(f"Headwise reshape failed in {index_path}")
    return source


def geometry(config):
    return {"layers": config.num_hidden_layers, "query_heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads,
            "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads), "hidden_size": config.hidden_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", required=True); parser.add_argument("--receiver", required=True); parser.add_argument("--train-cache", required=True); parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--train-data", required=True); parser.add_argument("--validation-data", required=True); parser.add_argument("--stage-a-gate", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); errors = []; sender = AutoConfig.from_pretrained(args.sender, trust_remote_code=True, local_files_only=True); receiver = AutoConfig.from_pretrained(args.receiver, trust_remote_code=True, local_files_only=True)
    sender_geometry, receiver_geometry = geometry(sender), geometry(receiver)
    if sender_geometry["layers"] != 36 or sender_geometry["kv_heads"] != 8 or sender_geometry["head_dim"] != 128: errors.append(f"Unexpected Qwen3-8B geometry: {sender_geometry}")
    if receiver_geometry != {"layers": 36, "query_heads": 32, "kv_heads": 8, "head_dim": 128, "hidden_size": 2560}: errors.append(f"Unexpected Qwen3-4B geometry: {receiver_geometry}")
    train = audit_split(args.train_cache, args.train_data, 512, errors); validation = audit_split(args.validation_cache, args.validation_data, 64, errors)
    if {entry["id"] for entry in train["entries"]} & {entry["id"] for entry in validation["entries"]}: errors.append("Train/validation overlap")
    gate = read_json(args.stage_a_gate)
    if not gate.get("passed"): errors.append("Stage A overfit gate did not pass; Stage B must not run")
    result = {"status": "complete" if not errors else "failed", "errors": errors, "stage_a_gate": gate,
              "sender_geometry": sender_geometry, "receiver_geometry": receiver_geometry, "selected_layers": SELECTED_LAYERS,
              "sender_config_sha256": file_sha256(Path(args.sender) / "config.json"), "receiver_config_sha256": file_sha256(Path(args.receiver) / "config.json"),
              "invariants": {"writer_loaded": False, "canonical_projection": False, "native_reshape_lossless": True,
                             "memory": "Qwen3-8B evidence-only [16,T,8,128]", "receiver": "frozen Qwen3-4B"}}
    write_json(args.out, result)
    if errors: raise RuntimeError("; ".join(errors))


if __name__ == "__main__": main()
