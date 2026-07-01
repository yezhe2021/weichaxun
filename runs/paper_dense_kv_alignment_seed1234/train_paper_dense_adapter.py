import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from paper_dense_common import (
    assert_tokenizer_compatible,
    build_paper_example,
    generation_ce,
    load_rows,
    receiver_cache_reconstruction_loss,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
for path in (PROJECT_ROOT, REAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_kv_common import extract_cache, head_dim_from_config, rope_roundtrip_error, rope_theta_from_config  # noqa: E402
from real_kv_translator import RealCrossModelKVTranslator, load_real_translator, save_real_translator  # noqa: E402


def load_model(path, dtype, device):
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, trust_remote_code=True).to(device).eval()


def build_adapter(sender_config, receiver_config, args, device):
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


def method_plan(method, args):
    if method == "paper_rec_then_mixed_generation":
        return [
            ("phase1_receiver_cache_reconstruction", args.phase1_lr, args.phase1_epochs),
            ("phase2_mixed_generation", args.phase2_lr, args.phase2_epochs),
        ]
    if method == "mse_only":
        return [("mse", args.phase1_lr, args.phase1_epochs + args.phase2_epochs)]
    if method == "mse_then_ce":
        return [
            ("mse", args.phase1_lr, args.phase1_epochs),
            ("ce", args.phase2_lr, args.phase2_epochs),
        ]
    raise ValueError(f"Unknown method: {method}")


def load_or_build_adapter(args, sender_config, receiver_config, device):
    if args.init_checkpoint:
        adapter, init_metadata = load_real_translator(args.init_checkpoint, map_location=device)
        return adapter.to(device), init_metadata
    return build_adapter(sender_config, receiver_config, args, device), None


def get_source_caches(sender, receiver, source_ids):
    with torch.no_grad():
        sender_pairs = extract_cache(sender(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
        receiver_pairs = extract_cache(receiver(input_ids=source_ids, use_cache=True, logits_to_keep=1).past_key_values)
    return sender_pairs, receiver_pairs


def generation_losses(receiver, translated_pairs, example, device, aware_weight):
    answer_ids = example["answer_ids"].to(device)
    aware_ce = generation_ce(
        receiver,
        translated_pairs,
        example["aware_tail_ids"].to(device),
        example["aware_prefix_len"],
        answer_ids,
    )
    unaware_ce = generation_ce(
        receiver,
        translated_pairs,
        example["unaware_tail_ids"].to(device),
        example["unaware_prefix_len"],
        answer_ids,
    )
    mixed = aware_weight * aware_ce + (1.0 - aware_weight) * unaware_ce
    return mixed, aware_ce, unaware_ce


def train_stage(sender, receiver, tokenizer, adapter, rows, optimizer, args, stage_name, stage_index, history, global_step):
    stage_rows = list(rows)
    random.Random(args.seed + stage_index).shuffle(stage_rows)
    progress = tqdm(stage_rows, desc=f"{args.method}:{stage_name}")
    optimizer.zero_grad(set_to_none=True)
    for index, row in enumerate(progress):
        example = build_paper_example(tokenizer, row, args.max_source_tokens)
        source_ids = example["source_ids"].to(args.device_obj)
        sender_pairs, receiver_pairs = get_source_caches(sender, receiver, source_ids)
        translated_pairs = adapter(sender_pairs)

        rec_loss = receiver_cache_reconstruction_loss(receiver_pairs, translated_pairs)
        aware_ce = torch.zeros((), device=args.device_obj)
        unaware_ce = torch.zeros((), device=args.device_obj)
        gen_loss = torch.zeros((), device=args.device_obj)

        if stage_name in {"phase1_receiver_cache_reconstruction", "mse"}:
            loss = rec_loss
        elif stage_name == "phase2_mixed_generation":
            gen_loss, aware_ce, unaware_ce = generation_losses(
                receiver, translated_pairs, example, args.device_obj, args.context_aware_weight
            )
            loss = gen_loss
        elif stage_name == "ce":
            answer_ids = example["answer_ids"].to(args.device_obj)
            unaware_ce = generation_ce(
                receiver,
                translated_pairs,
                example["unaware_tail_ids"].to(args.device_obj),
                example["unaware_prefix_len"],
                answer_ids,
            )
            gen_loss = unaware_ce
            loss = unaware_ce
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

        (loss / args.grad_accum).backward()
        if (index + 1) % args.grad_accum == 0 or index + 1 == len(stage_rows):
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        item = {
            "method": args.method,
            "stage": stage_name,
            "stage_index": stage_index,
            "sample": index,
            "step": global_step,
            "loss": float(loss.detach().item()),
            "receiver_cache_reconstruction_loss": float(rec_loss.detach().item()),
            "generation_loss": float(gen_loss.detach().item()),
            "context_aware_ce": float(aware_ce.detach().item()),
            "context_unaware_ce": float(unaware_ce.detach().item()),
        }
        history.append(item)
        progress.set_postfix(loss=f"{item['loss']:.3f}", rec=f"{item['receiver_cache_reconstruction_loss']:.3f}", gen=f"{item['generation_loss']:.3f}")
    return global_step


@torch.no_grad()
def validate(sender, receiver, tokenizer, adapter, rows, args):
    adapter.eval()
    totals = {"rec": [], "aware_ce": [], "unaware_ce": [], "mixed_gen": []}
    for row in rows:
        example = build_paper_example(tokenizer, row, args.max_source_tokens)
        source_ids = example["source_ids"].to(args.device_obj)
        sender_pairs, receiver_pairs = get_source_caches(sender, receiver, source_ids)
        translated_pairs = adapter(sender_pairs)
        rec_loss = receiver_cache_reconstruction_loss(receiver_pairs, translated_pairs)
        mixed, aware_ce, unaware_ce = generation_losses(receiver, translated_pairs, example, args.device_obj, args.context_aware_weight)
        totals["rec"].append(rec_loss.item())
        totals["aware_ce"].append(aware_ce.item())
        totals["unaware_ce"].append(unaware_ce.item())
        totals["mixed_gen"].append(mixed.item())
    adapter.train()
    return {f"val_{key}": float(np.mean(values)) for key, values in totals.items()} if rows else {}


def main():
    parser = argparse.ArgumentParser(description="Train paper-style dense KV cache alignment adapter")
    parser.add_argument("--sender-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--train-data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl")
    parser.add_argument("--val-data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["paper_rec_then_mixed_generation", "mse_only", "mse_then_ce"], required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--gate-init", type=float, default=2.0)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--phase1-epochs", type=int, default=1)
    parser.add_argument("--phase2-epochs", type=int, default=1)
    parser.add_argument("--phase1-lr", type=float, default=2e-4)
    parser.add_argument("--phase2-lr", type=float, default=5e-5)
    parser.add_argument("--context-aware-weight", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--tokenizer-check-samples", type=int, default=8)
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
    assert_tokenizer_compatible(sender_tokenizer, receiver_tokenizer, check_rows, args.max_source_tokens, args.tokenizer_check_samples)

    sender = load_model(args.sender_model, dtype, args.device_obj)
    receiver = load_model(args.receiver_model, dtype, args.device_obj)
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    adapter, init_metadata = load_or_build_adapter(args, sender.config, receiver.config, args.device_obj)
    adapter.train()

    rope_checks = {
        "sender_rope_roundtrip_max_abs": rope_roundtrip_error(
            (1, sender.config.num_key_value_heads, 8, head_dim_from_config(sender.config)),
            rope_theta_from_config(sender.config),
            args.device_obj,
            dtype,
        ),
        "receiver_rope_roundtrip_max_abs": rope_roundtrip_error(
            (1, receiver.config.num_key_value_heads, 8, head_dim_from_config(receiver.config)),
            rope_theta_from_config(receiver.config),
            args.device_obj,
            dtype,
        ),
    }

    history = []
    global_step = 0
    plan = method_plan(args.method, args)
    metadata = {}
    for stage_index, (stage_name, lr, epochs) in enumerate(plan):
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=args.weight_decay)
        for _ in range(epochs):
            global_step = train_stage(sender, receiver, receiver_tokenizer, adapter, train_rows, optimizer, args, stage_name, stage_index, history, global_step)
        validation = validate(sender, receiver, receiver_tokenizer, adapter, val_rows, args)
        metadata = {
            "args": {k: v for k, v in vars(args).items() if k != "device_obj"},
            "method": args.method,
            "stage_index": stage_index,
            "stage_name": stage_name,
            "global_step": global_step,
            "method_plan": [{"stage": s, "learning_rate": lr, "epochs": ep} for s, lr, ep in plan],
            "init_checkpoint_metadata": init_metadata,
            "loss_definitions": {
                "phase1": "receiver_cache_reconstruction_loss = mean_l,g MSE(K_trans,K_receiver)+MSE(V_trans,V_receiver)",
                "phase2": "generation_loss = -sum_t log p_R(y_t | y_<t, translated_cache, X_R), mixed over context-aware X_R=X and context-unaware X_R=empty",
                "mse_then_ce_baseline": "receiver_cache_reconstruction followed by context-unaware gold-answer CE",
            },
            "source_x": "context + question, no gold answer",
            "context_aware_receiver_input": "X + Answer: + answer_prefix",
            "context_unaware_receiver_input": "Answer: + answer_prefix",
            "rope_checks": rope_checks,
            "translator_config": adapter.config_dict(),
            **validation,
        }
        save_real_translator(output / f"checkpoint_stage{stage_index + 1}.pt", adapter, metadata)
        with open(output / "train_history.jsonl", "w", encoding="utf-8") as handle:
            for item in history:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(output / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
    save_real_translator(output / "checkpoint_final.pt", adapter, metadata)


if __name__ == "__main__":
    main()
