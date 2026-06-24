import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from quant_kv_common import dtype_from_name, load_quantized_model, write_json

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
sys.path.insert(0, str(REAL_ROOT))

from real_kv_common import (  # noqa: E402
    answer_ce,
    assert_tokenizer_compatible,
    build_example,
    extract_cache,
    head_dim_from_config,
    load_rows,
    normalized_kv_mse,
    rope_roundtrip_error,
    rope_theta_from_config,
)
from real_kv_translator import (  # noqa: E402
    RealCrossModelKVTranslator,
    load_real_translator,
    save_real_translator,
)


def build_translator(sender_config, receiver_config, args, device):
    return RealCrossModelKVTranslator(
        sender_layers=sender_config.num_hidden_layers,
        sender_kv_heads=sender_config.num_key_value_heads,
        sender_head_dim=head_dim_from_config(sender_config),
        sender_rope_theta=rope_theta_from_config(sender_config),
        receiver_layers=receiver_config.num_hidden_layers,
        receiver_kv_heads=receiver_config.num_key_value_heads,
        receiver_head_dim=head_dim_from_config(receiver_config),
        receiver_rope_theta=rope_theta_from_config(receiver_config),
        hidden=args.hidden,
        gate_init=args.gate_init,
    ).to(device)


@torch.no_grad()
def validate(sender, receiver, tokenizer, translator, rows, args):
    translator.eval()
    totals = {"mse": [], "receiver_native_ce": [], "translated_ce": []}
    for row in rows:
        example = build_example(tokenizer, row, args.max_context_tokens)
        context_ids = example["context_ids"].to(args.device_obj)
        tail_ids = example["tail_ids"].to(args.device_obj)
        answer_ids = example["answer_ids"].to(args.device_obj)
        sender_context = extract_cache(sender(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values)
        receiver_context = extract_cache(
            receiver(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values
        )
        translated = translator(sender_context)
        totals["mse"].append(normalized_kv_mse(receiver_context, translated).item())
        totals["receiver_native_ce"].append(
            answer_ce(receiver, receiver_context, tail_ids, example["query_ids"].shape[1], answer_ids).item()
        )
        totals["translated_ce"].append(
            answer_ce(receiver, translated, tail_ids, example["query_ids"].shape[1], answer_ids).item()
        )
    translator.train()
    result = {f"val_{key}": float(np.mean(values)) for key, values in totals.items()} if rows else {}
    if rows:
        result["val_ce_delta"] = result["val_translated_ce"] - result["val_receiver_native_ce"]
    return result


def train_epoch(sender, receiver, tokenizer, translator, rows, optimizer, args, epoch, history):
    random.Random(args.seed + epoch).shuffle(rows)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(rows, desc=f"epoch {epoch + 1}")
    global_step = history[-1]["step"] if history else 0
    for index, row in enumerate(progress):
        example = build_example(tokenizer, row, args.max_context_tokens)
        context_ids = example["context_ids"].to(args.device_obj)
        tail_ids = example["tail_ids"].to(args.device_obj)
        answer_ids = example["answer_ids"].to(args.device_obj)
        with torch.no_grad():
            sender_context = extract_cache(sender(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values)
            receiver_context = extract_cache(
                receiver(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values
            )
        translated = translator(sender_context)
        mse = normalized_kv_mse(receiver_context, translated)
        ce = (
            answer_ce(receiver, translated, tail_ids, example["query_ids"].shape[1], answer_ids)
            if args.objective == "ce"
            else None
        )
        loss = mse if args.objective == "mse" else ce
        (loss / args.grad_accum).backward()
        if (index + 1) % args.grad_accum == 0 or index + 1 == len(rows):
            torch.nn.utils.clip_grad_norm_(translator.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        item = {
            "epoch": epoch,
            "sample": index,
            "step": global_step,
            "objective": args.objective,
            "loss": loss.detach().item(),
            "mse": mse.detach().item(),
            "ce": ce.detach().item() if ce is not None else None,
        }
        history.append(item)
        progress.set_postfix(loss=f"{item['loss']:.3f}", mse=f"{item['mse']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train quantization-controlled context-only KV translator")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--sender-precision", choices=["fp16", "int4"], required=True)
    parser.add_argument("--receiver-precision", choices=["fp16", "int4"], required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--objective", choices=["mse", "ce"], required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--gate-init", type=float, default=2.0)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device_obj = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows(args.train_data, args.max_train_samples)
    val_rows = load_rows(args.val_data, args.max_val_samples)
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(
        sender_tokenizer,
        receiver_tokenizer,
        train_rows[:8] + val_rows[:8],
        args.max_context_tokens,
        8,
    )
    sender, sender_audit = load_quantized_model(
        args.sender_model, args.sender_precision, dtype, args.device_obj
    )
    receiver, receiver_audit = load_quantized_model(
        args.receiver_model, args.receiver_precision, dtype, args.device_obj
    )
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    rope_checks = {
        "sender": rope_roundtrip_error(
            (1, sender.config.num_key_value_heads, 8, head_dim_from_config(sender.config)),
            rope_theta_from_config(sender.config),
            args.device_obj,
            dtype,
        ),
        "receiver": rope_roundtrip_error(
            (1, receiver.config.num_key_value_heads, 8, head_dim_from_config(receiver.config)),
            rope_theta_from_config(receiver.config),
            args.device_obj,
            dtype,
        ),
    }
    if args.init_checkpoint:
        translator, init_metadata = load_real_translator(args.init_checkpoint, map_location=args.device_obj)
        translator = translator.to(args.device_obj).train()
    else:
        translator = build_translator(sender.config, receiver.config, args, args.device_obj).train()
        init_metadata = None
    optimizer = torch.optim.AdamW(
        translator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    for epoch in range(args.epochs):
        train_epoch(sender, receiver, receiver_tokenizer, translator, train_rows, optimizer, args, epoch, history)
        validation = validate(sender, receiver, receiver_tokenizer, translator, val_rows, args)
        metadata = {
            "args": {key: value for key, value in vars(args).items() if key != "device_obj"},
            "epoch": epoch,
            "sender_quantization": sender_audit,
            "receiver_quantization": receiver_audit,
            "rope_checks": rope_checks,
            "init_checkpoint_metadata": init_metadata,
            "translator_config": translator.config_dict(),
            **validation,
        }
        save_real_translator(output / f"checkpoint_epoch{epoch + 1}.pt", translator, metadata)
        with open(output / "train_history.jsonl", "w", encoding="utf-8") as handle:
            for item in history:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        write_json(output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
