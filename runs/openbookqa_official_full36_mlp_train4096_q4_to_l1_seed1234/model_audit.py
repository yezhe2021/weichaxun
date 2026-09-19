from transformers import AutoConfig

from common import run_root, save_json, tokenizer


def run(cfg):
    expected = {"qwen": (36, 8, 128), "llama": (16, 8, 64)}
    records = {}
    for family, path in cfg["models"].items():
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        observed = (config.num_hidden_layers, config.num_key_value_heads, config.head_dim)
        if observed != expected[family]:
            raise RuntimeError(f"{family} geometry mismatch: expected={expected[family]} observed={observed}")
        tok = tokenizer(path)
        records[family] = {"path": path, "layers": observed[0], "kv_heads": observed[1],
                           "head_dim": observed[2], "tokenizer_class": type(tok).__name__,
                           "bos_token_id": tok.bos_token_id}
    records["translator"] = {
        "source_shape": [36, 32, 8, 128], "target_shape": [16, 32, 8, 64],
        "per_head_depth_width": 4608, "hidden_dim": cfg["mlp_hidden_dim"],
        "head_mixing": False, "token_mixing": False, "k_v_shared": False,
    }
    save_json(run_root(cfg) / "audit" / "model_geometry.json", records)
    print(records, flush=True)
    return records
