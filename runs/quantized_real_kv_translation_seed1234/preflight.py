import argparse
import importlib.util
import json
import sys
from pathlib import Path

from transformers import AutoConfig

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
sys.path.insert(0, str(REAL_ROOT))

from real_kv_common import head_dim_from_config, rope_theta_from_config  # noqa: E402


def signature(config):
    return {
        "layers": config.num_hidden_layers,
        "kv_heads": config.num_key_value_heads,
        "attention_heads": config.num_attention_heads,
        "head_dim": head_dim_from_config(config),
        "rope_theta": rope_theta_from_config(config),
        "vocab_size": config.vocab_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sender = AutoConfig.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver = AutoConfig.from_pretrained(args.receiver_model, trust_remote_code=True)
    sender_sig, receiver_sig = signature(sender), signature(receiver)
    direct = all(sender_sig[key] == receiver_sig[key] for key in ("layers", "kv_heads", "head_dim"))
    payload = {
        "sender": sender_sig,
        "receiver": receiver_sig,
        "alignment_mode": "direct_layer_head" if direct else "fixed_index_relation_only",
        "packages": {
            name: bool(importlib.util.find_spec(name))
            for name in ("torch", "transformers", "accelerate", "bitsandbytes")
        },
        "int4_backend_ready": bool(importlib.util.find_spec("bitsandbytes")),
        "note": "No model weights were loaded. INT4 runs fail rather than fall back when backend is missing.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
