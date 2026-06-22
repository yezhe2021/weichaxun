import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_translators import PseudoSenderTranslator, save_translator
from translated_kv_diagnostics import (
    answer_logits,
    build_example,
    extract_cache,
    load_rows,
    make_cache,
)


def normalized_mse(native, translated):
    losses = []
    for (nk, nv), (tk, tv) in zip(native, translated):
        losses.append((tk.float() - nk.float()).pow(2).mean() / (nk.float().pow(2).mean() + 1e-8))
        losses.append((tv.float() - nv.float()).pow(2).mean() / (nv.float().pow(2).mean() + 1e-8))
    return torch.stack(losses).mean()


def answer_ce(model, translated, tail_ids, query_len, answer_ids):
    cache = make_cache(translated, model.config)
    out = model(input_ids=tail_ids, past_key_values=cache, use_cache=True)
    logits = answer_logits(out.logits, query_len, answer_ids.shape[1]).float()
    n = min(logits.shape[1], answer_ids.shape[1])
    return F.cross_entropy(logits[:, :n].reshape(-1, logits.shape[-1]), answer_ids[:, :n].reshape(-1))


def rope_theta_from_config(config):
    direct = getattr(config, "rope_theta", None)
    if direct is not None:
        return float(direct)
    parameters = getattr(config, "rope_parameters", None) or {}
    if "rope_theta" in parameters:
        return float(parameters["rope_theta"])
    for value in parameters.values():
        if isinstance(value, dict) and "rope_theta" in value:
            return float(value["rope_theta"])
    return 10_000.0


def evaluate(model, tokenizer, translator, rows, args, device):
    translator.eval()
    totals = {"mse": [], "ce": []}
    with torch.no_grad():
        for row in rows:
            ex = build_example(tokenizer, row, args.max_context_tokens)
            context_ids = ex["context_ids"].to(device)
            tail_ids = ex["tail_ids"].to(device)
            answer_ids = ex["answer_ids"].to(device)
            native = extract_cache(model(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values, detach=True)
            translated = translator(native)
            totals["mse"].append(normalized_mse(native, translated).item())
            totals["ce"].append(answer_ce(model, translated, tail_ids, ex["query_ids"].shape[1], answer_ids).item())
    translator.train()
    return {f"val_{key}": float(np.mean(values)) for key, values in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train pseudo-sender KV translators with controlled objectives")
    parser.add_argument("--model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B")
    parser.add_argument("--train-data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl")
    parser.add_argument("--val-data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl")
    parser.add_argument("--out", default="runs/kv_translator")
    parser.add_argument("--objective", choices=["mse", "ce", "mse_ce"], required=True)
    parser.add_argument("--translator-kind", choices=["pseudo_sender", "autoencoder"], default="pseudo_sender")
    parser.add_argument("--rope-disentangled", action="store_true")
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-val-samples", type=int, default=16)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mse-weight", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    args = parser.parse_args()
    if args.translator_kind == "autoencoder" and args.objective != "mse":
        raise ValueError("autoencoder is the MSE reconstruction control; use --objective mse")
    if args.device == "cpu" and args.dtype == "float16":
        raise ValueError("float16 training on CPU is unsupported")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = model.config
    translator = PseudoSenderTranslator(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        bottleneck=args.bottleneck,
        hidden=args.hidden,
        seed=args.seed,
        trainable_encoder=args.translator_kind == "autoencoder",
        rope_disentangled=args.rope_disentangled,
        rope_theta=rope_theta_from_config(config),
    ).to(device).train()
    optimizer = torch.optim.AdamW(translator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_rows = load_rows(args.train_data, args.max_train_samples)
    val_rows = load_rows(args.val_data, args.max_val_samples)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    resolved_mse_weight = args.mse_weight
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(train_rows)
        progress = tqdm(train_rows, desc=f"epoch {epoch + 1}")
        for index, row in enumerate(progress):
            ex = build_example(tokenizer, row, args.max_context_tokens)
            context_ids = ex["context_ids"].to(device)
            tail_ids = ex["tail_ids"].to(device)
            answer_ids = ex["answer_ids"].to(device)
            with torch.no_grad():
                native = extract_cache(model(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values, detach=True)
            translated = translator(native)
            mse = normalized_mse(native, translated)
            ce = None
            if args.objective in {"ce", "mse_ce"}:
                ce = answer_ce(model, translated, tail_ids, ex["query_ids"].shape[1], answer_ids)
            if args.objective == "mse":
                loss = mse
            elif args.objective == "ce":
                loss = ce
            else:
                if resolved_mse_weight is None:
                    resolved_mse_weight = max(1.0, ce.detach().item() / max(mse.detach().item(), 1e-8))
                loss = ce + resolved_mse_weight * mse
            (loss / args.grad_accum).backward()
            if (index + 1) % args.grad_accum == 0 or index + 1 == len(train_rows):
                torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            row_log = {
                "epoch": epoch,
                "sample": index,
                "step": global_step,
                "loss": loss.detach().item(),
                "mse": mse.detach().item(),
                "ce": ce.detach().item() if ce is not None else None,
            }
            history.append(row_log)
            progress.set_postfix(loss=f"{row_log['loss']:.3f}", mse=f"{row_log['mse']:.3f}")
        validation = evaluate(model, tokenizer, translator, val_rows, args, device) if val_rows else {}
        metadata = {
            "args": vars(args),
            "epoch": epoch,
            "global_step": global_step,
            "resolved_mse_weight": resolved_mse_weight,
            **validation,
        }
        save_translator(output / f"checkpoint_epoch{epoch + 1}.pt", translator, metadata)
        with open(output / "train_history.jsonl", "w", encoding="utf-8") as f:
            for item in history:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(output / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
