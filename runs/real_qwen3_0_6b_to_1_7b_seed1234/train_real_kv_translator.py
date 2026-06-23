import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_kv_common import (
    answer_ce,
    assert_tokenizer_compatible,
    build_example,
    extract_cache,
    head_dim_from_config,
    load_rows,
    native_cache_equivalence,
    normalized_kv_mse,
    rope_roundtrip_error,
    rope_theta_from_config,
)
from real_kv_translator import RealCrossModelKVTranslator, load_real_translator, save_real_translator


def load_model(path, dtype, device, eager=False):
    kwargs = {"dtype": dtype, "trust_remote_code": True}
    if eager:
        kwargs["attn_implementation"] = "eager"
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()


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
    totals = {"mse": [], "ce": []}
    for row in rows:
        example = build_example(tokenizer, row, args.max_context_tokens)
        context_ids = example["context_ids"].to(args.device_obj)
        tail_ids = example["tail_ids"].to(args.device_obj)
        answer_ids = example["answer_ids"].to(args.device_obj)
        sender_context = extract_cache(sender(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values)
        receiver_context = extract_cache(receiver(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values)
        translated = translator(sender_context)
        totals["mse"].append(normalized_kv_mse(receiver_context, translated).item())
        totals["ce"].append(answer_ce(receiver, translated, tail_ids, example["query_ids"].shape[1], answer_ids).item())
    translator.train()
    return {f"val_{key}": float(np.mean(values)) for key, values in totals.items()} if rows else {}


def train_epoch(sender, receiver, tokenizer, translator, rows, optimizer, args, epoch, history):
    random.Random(args.seed + epoch).shuffle(rows)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(rows, desc=f"epoch {epoch + 1}")
    global_step = history[-1]["step"] if history else 0
    resolved_mse_weight = args.resolved_mse_weight
    for index, row in enumerate(progress):
        example = build_example(tokenizer, row, args.max_context_tokens)
        context_ids = example["context_ids"].to(args.device_obj)
        tail_ids = example["tail_ids"].to(args.device_obj)
        answer_ids = example["answer_ids"].to(args.device_obj)
        with torch.no_grad():
            sender_context = extract_cache(sender(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values)
            receiver_context = extract_cache(receiver(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values)
        translated = translator(sender_context)
        mse = normalized_kv_mse(receiver_context, translated)
        ce = None
        if args.objective in {"ce", "mse_ce"}:
            ce = answer_ce(receiver, translated, tail_ids, example["query_ids"].shape[1], answer_ids)
        if args.objective == "mse":
            loss = mse
        elif args.objective == "ce":
            loss = ce
        else:
            if resolved_mse_weight is None:
                resolved_mse_weight = max(1.0, ce.detach().item() / max(mse.detach().item(), 1e-8))
                args.resolved_mse_weight = resolved_mse_weight
            loss = ce + resolved_mse_weight * mse
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
            "mse_weight": resolved_mse_weight,
        }
        history.append(item)
        progress.set_postfix(loss=f"{item['loss']:.3f}", mse=f"{item['mse']:.3f}")
    return global_step


def main():
    parser = argparse.ArgumentParser(description="Train real Qwen3-0.6B -> Qwen3-1.7B context KV translator")
    parser.add_argument("--sender-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--train-data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl")
    parser.add_argument("--val-data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--objective", choices=["mse", "ce", "mse_ce"], required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--gate-init", type=float, default=2.0)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mse-weight", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--tokenizer-check-samples", type=int, default=8)
    parser.add_argument("--equivalence-atol", type=float, default=None)
    args = parser.parse_args()
    if args.device == "cpu" and args.dtype == "float16":
        raise ValueError("float16 on CPU is unsupported")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device_obj = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows(args.train_data, args.max_train_samples)
    val_rows = load_rows(args.val_data, args.max_val_samples)
    check_rows = train_rows[: args.tokenizer_check_samples] + val_rows[: args.tokenizer_check_samples]
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    assert_tokenizer_compatible(sender_tokenizer, receiver_tokenizer, check_rows, args.max_context_tokens, args.tokenizer_check_samples)
    tokenizer = receiver_tokenizer
    sender = load_model(args.sender_model, dtype, args.device_obj)
    receiver = load_model(args.receiver_model, dtype, args.device_obj)
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    sender_theta = rope_theta_from_config(sender.config)
    receiver_theta = rope_theta_from_config(receiver.config)
    rope_checks = {
        "sender_rope_roundtrip_max_abs": rope_roundtrip_error(
            (1, sender.config.num_key_value_heads, 8, head_dim_from_config(sender.config)), sender_theta, args.device_obj, dtype
        ),
        "receiver_rope_roundtrip_max_abs": rope_roundtrip_error(
            (1, receiver.config.num_key_value_heads, 8, head_dim_from_config(receiver.config)), receiver_theta, args.device_obj, dtype
        ),
    }
    atol = args.equivalence_atol if args.equivalence_atol is not None else (0.25 if args.dtype == "float16" else 1e-3)
    equivalence = None
    if val_rows:
        ex = build_example(tokenizer, val_rows[0], args.max_context_tokens)
        equivalence = native_cache_equivalence(
            receiver,
            ex["context_ids"].to(args.device_obj),
            ex["tail_ids"].to(args.device_obj),
            ex["query_ids"].shape[1],
            ex["answer_ids"].shape[1],
            atol,
        )
    if args.init_checkpoint:
        translator, init_metadata = load_real_translator(args.init_checkpoint, map_location=args.device_obj)
        translator = translator.to(args.device_obj).train()
    else:
        translator = build_translator(sender.config, receiver.config, args, args.device_obj).train()
        init_metadata = None
    optimizer = torch.optim.AdamW(translator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    args.resolved_mse_weight = args.mse_weight
    global_step = 0
    for epoch in range(args.epochs):
        global_step = train_epoch(sender, receiver, tokenizer, translator, train_rows, optimizer, args, epoch, history)
        validation = validate(sender, receiver, tokenizer, translator, val_rows, args)
        metadata = {
            "args": {k: v for k, v in vars(args).items() if k not in {"device_obj", "resolved_mse_weight"}},
            "epoch": epoch,
            "global_step": global_step,
            "resolved_mse_weight": args.resolved_mse_weight,
            "init_checkpoint_metadata": init_metadata,
            "rope_checks": rope_checks,
            "native_context_cache_equivalence": equivalence,
            "translator_config": translator.config_dict(),
            **validation,
        }
        save_real_translator(output / f"checkpoint_epoch{epoch + 1}.pt", translator, metadata)
        with open(output / "train_history.jsonl", "w", encoding="utf-8") as f:
            for item in history:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(output / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
