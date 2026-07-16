import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def describe(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    return {
        "model": model_path,
        "architecture": (config.architectures or [config.model_type])[0],
        "model_type": config.model_type,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "hidden_size": int(config.hidden_size),
        "layers": int(config.num_hidden_layers),
        "query_heads": int(config.num_attention_heads),
        "kv_heads": int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "rope_theta": float(getattr(config, "rope_theta", 10000.0)),
        "rope_scaling": getattr(config, "rope_scaling", None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sender = describe(args.sender_model)
    receiver = describe(args.receiver_model)
    audit = {
        "status": "complete",
        "sender": sender,
        "receiver": receiver,
        "differences": {
            key: {"sender": sender[key], "receiver": receiver[key]}
            for key in (
                "architecture",
                "tokenizer_class",
                "vocab_size",
                "layers",
                "query_heads",
                "kv_heads",
                "head_dim",
                "rope_theta",
                "rope_scaling",
            )
            if sender[key] != receiver[key]
        },
        "writer_contract": {
            "input": "all evidence-token pre-RoPE K and native V from every sender layer",
            "output": "one complete receiver-compatible Evidence-KV sequence per receiver layer",
            "compression": False,
            "receiver_specific": True,
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
