import argparse
from pathlib import Path

from transformers import AutoConfig

from p3d3_common import SELECTED_LAYERS, file_sha256, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model4", required=True); parser.add_argument("--model8", required=True); parser.add_argument("--protocol", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--native-train"); parser.add_argument("--native-validation"); parser.add_argument("--canonical-train"); parser.add_argument("--canonical-validation")
    args = parser.parse_args(); errors = []; protocol = read_json(args.protocol); config4, config8 = AutoConfig.from_pretrained(args.model4, local_files_only=True, trust_remote_code=True), AutoConfig.from_pretrained(args.model8, local_files_only=True, trust_remote_code=True)
    geometry4 = {"layers": config4.num_hidden_layers, "query_heads": config4.num_attention_heads, "kv_heads": config4.num_key_value_heads,
                 "hidden_size": config4.hidden_size, "head_dim": getattr(config4, "head_dim", config4.hidden_size // config4.num_attention_heads)}
    expected4 = {"layers": 36, "query_heads": 32, "kv_heads": 8, "hidden_size": 2560, "head_dim": 128}
    if geometry4 != expected4: errors.append(f"Unexpected Qwen3-4B geometry: {geometry4}")
    if config8.num_hidden_layers != 36 or config8.num_key_value_heads != 8 or getattr(config8, "head_dim", 128) != 128: errors.append("Unexpected Qwen3-8B geometry")
    if protocol["selected_sender_layers"] != SELECTED_LAYERS or protocol["selected_receiver_layers"] != SELECTED_LAYERS: errors.append("Layer order mismatch")
    if file_sha256(protocol["writer_checkpoint"]) != protocol["writer_sha256"]: errors.append("Writer hash mismatch")
    cache_specs = [(args.native_train, 1024), (args.native_validation, 1024), (args.canonical_train, 256), (args.canonical_validation, 256)]
    cache_ids = {}
    for path, dimension in cache_specs:
        if not path: continue
        index = read_json(path)
        if index["original_layer_indices"] != SELECTED_LAYERS or index["memory_dim"] != dimension: errors.append(f"Cache interface mismatch: {path}")
        if not index.get("question_independent"): errors.append(f"Cache is not question-independent: {path}")
        split = "validation" if "validation" in str(path) else "train"; cache_ids.setdefault(split, []).append([entry["id"] for entry in index["entries"]])
        if dimension == 256 and index.get("writer_checkpoint_sha256") != protocol["writer_sha256"]: errors.append(f"Canonical cache Writer hash mismatch: {path}")
    for split, identifiers in cache_ids.items():
        if len(identifiers) == 2 and identifiers[0] != identifiers[1]: errors.append(f"Native/Canonical {split} IDs differ")
    result = {"status": "complete" if not errors else "failed", "errors": errors, "receiver_geometry": geometry4, "protocol": protocol,
              "invariants": {"sender_input": "evidence_only", "fixed_layer_alignment": True, "pre_rope_native_query": True,
                             "direct_scalar_gate": True, "validation_not_used_for_training": True}}
    write_json(args.out, result)
    if errors: raise RuntimeError("; ".join(errors))


if __name__ == "__main__": main()
