from transformers import AutoConfig

from common import geometry, run_root, save_json, text_config, tokenizer


def run(cfg):
    expected = {"qwen": (36, 8, 128), "gemma": (34, 4, 256)}
    records = {}
    for family, path in cfg["models"].items():
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        observed = geometry(config)
        if observed != expected[family]:
            raise RuntimeError(f"{family} geometry mismatch: expected={expected[family]} observed={observed}")
        tok = tokenizer(path)
        records[family] = {
            "path": path, "layers": observed[0], "kv_heads": observed[1], "head_dim": observed[2],
            "tokenizer_class": type(tok).__name__, "bos_token_id": tok.bos_token_id,
        }
        if family == "gemma":
            config = text_config(config)
            records[family]["layer_types"] = list(config.layer_types)
            records[family]["rope_parameters"] = config.rope_parameters
    records["translator"] = {
        "source_shape": [36, 32, 8, 128], "target_shape": [34, 32, 4, 256],
        "per_head_depth_width": 4608, "depth_hidden_dim": cfg["depth_hidden_dim"],
        "depth_output_dim": cfg["depth_output_dim"], "head_mapping": cfg["head_mapping"],
        "head_input_width": 1024, "head_output_width": 1024,
        "token_mixing": False, "k_v_shared": False,
    }
    save_json(run_root(cfg) / "audit" / "model_geometry.json", records)
    print(records, flush=True)
    return records
