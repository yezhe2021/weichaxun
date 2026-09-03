import argparse

from transformers import AutoConfig

from p3d3_common import file_sha256, read_json, write_json
from p3e_c2_common import SenderNativeHeadwiseCache


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--train-cache", required=True); parser.add_argument("--validation-cache", required=True); parser.add_argument("--writer", required=True); parser.add_argument("--c2-success", required=True); parser.add_argument("--qwen35", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); errors = []; train, validation = SenderNativeHeadwiseCache(args.train_cache), SenderNativeHeadwiseCache(args.validation_cache)
    if len(train) != 512 or len(validation) != 64: errors.append("Expected train512/validation64")
    if {entry["id"] for entry in train.entries} & {entry["id"] for entry in validation.entries}: errors.append("Train/validation overlap")
    if read_json(args.c2_success).get("status") != "complete": errors.append("C2 completion marker missing")
    outer = AutoConfig.from_pretrained(args.qwen35, trust_remote_code=True, local_files_only=True); config = outer.text_config
    full_layers = [index for index, kind in enumerate(config.layer_types) if kind == "full_attention"]
    expected = {"layers": 32, "query_heads": 16, "kv_heads": 4, "head_dim": 256, "hidden_size": 2560, "full_layers": [3, 7, 11, 15, 19, 23, 27, 31]}
    observed = {"layers": config.num_hidden_layers, "query_heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads,
                "head_dim": config.head_dim, "hidden_size": config.hidden_size, "full_layers": full_layers}
    if observed != expected: errors.append(f"Unexpected Qwen3.5 geometry: {observed}")
    write_json(args.out, {"status": "complete" if not errors else "failed", "errors": errors, "writer": args.writer, "writer_sha256": file_sha256(args.writer),
        "qwen35_geometry": observed, "pipeline": ["C3-A fully_random two seeds", "C3-A weak_pair two seeds", "C3-B pair ablation two strict seeds", "C4 Qwen3.5 two seeds"],
        "unconditional_execution": True, "score_based_early_stop": False})
    if errors: raise RuntimeError("; ".join(errors))


if __name__ == "__main__": main()
