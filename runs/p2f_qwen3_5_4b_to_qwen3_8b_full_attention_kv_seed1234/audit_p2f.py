import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def text_config(config):
    return getattr(config, "text_config", config)


def describe(model_path):
    outer = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    config = text_config(outer)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    layer_types = list(
        getattr(config, "layer_types", ["full_attention"] * config.num_hidden_layers)
    )
    full_layers = [
        index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention"
    ]
    return {
        "model": model_path,
        "outer_architecture": (outer.architectures or [outer.model_type])[0],
        "model_type": config.model_type,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "hidden_size": int(config.hidden_size),
        "layers": int(config.num_hidden_layers),
        "layer_types": layer_types,
        "full_attention_layers": full_layers,
        "query_heads": int(config.num_attention_heads),
        "kv_heads": int(config.num_key_value_heads),
        "head_dim": int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        ),
        "rope_parameters": getattr(config, "rope_parameters", None),
        "rope_theta": float(getattr(config, "rope_theta", 10000.0)),
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
        "writer_contract": {
            "input": "evidence-token pre-RoPE K/native V from genuine Qwen3.5 full-attention layers only",
            "source_layers": sender["full_attention_layers"],
            "excluded_source_layers": [
                index
                for index, kind in enumerate(sender["layer_types"])
                if kind != "full_attention"
            ],
            "output": "complete Qwen3-8B Reader-compatible per-layer KV",
            "compression": "no token compression; structural layer/head/dimension mapping only",
            "claim_boundary": "Linear-attention recurrent state is not represented as token-level KV.",
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
