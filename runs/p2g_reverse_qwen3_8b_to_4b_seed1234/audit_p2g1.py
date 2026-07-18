import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Audit the P2-G1 Qwen3-4B receiver and dataset split")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-4B")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    geometry = {
        "layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "query_heads": int(config.num_attention_heads),
        "kv_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "query_width": int(config.num_attention_heads) * int(config.head_dim),
    }
    expected = {
        "layers": 36,
        "hidden_size": 2560,
        "query_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "query_width": 4096,
    }
    if geometry != expected:
        raise RuntimeError(f"Unexpected Qwen3-4B geometry: {geometry}, expected {expected}")

    def inspect_split(path):
        rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        grouped = {}
        for row in rows:
            grouped.setdefault(row["pair_id"], set()).add(row["variant"])
        incomplete = [pair_id for pair_id, variants in grouped.items() if variants != {"base", "counterfactual"}]
        if incomplete:
            raise RuntimeError(f"Incomplete base/CF pairs in {path}: {incomplete[:5]}")
        return {"path": path, "rows": len(rows), "pairs": len(grouped)}

    train = inspect_split(args.train_data)
    test = inspect_split(args.test_data)
    if train["pairs"] < 512 or test["pairs"] < 64:
        raise RuntimeError(f"Insufficient data: train={train['pairs']} pairs, test={test['pairs']} pairs")

    report = {
        "status": "passed",
        "model": args.model,
        "model_type": config.model_type,
        "tokenizer_class": tokenizer.__class__.__name__,
        "geometry": geometry,
        "train": train,
        "test": test,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
