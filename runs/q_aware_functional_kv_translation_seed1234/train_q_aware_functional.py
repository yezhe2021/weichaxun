import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from q_aware_common import logit_kl_loss, q_aware_losses, tail_logits

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
REAL_ROOT = PROJECT_ROOT / "runs" / "real_qwen3_0_6b_to_1_7b_seed1234"
for path in (PROJECT_ROOT, REAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_kv_common import (  # noqa: E402
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
from real_kv_translator import RealCrossModelKVTranslator, save_real_translator  # noqa: E402


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


def regime_stages(regime, args):
    if regime == "mse_only":
        return [("mse", args.mse_lr, args.stage_epochs)] * 3
    if regime == "mse_then_ce":
        return [
            ("mse", args.mse_lr, args.stage_epochs),
            ("ce", args.ce_lr, args.stage_epochs),
            ("ce", args.ce_lr, args.stage_epochs),
        ]
    if regime == "q_aware_functional":
        return [
            ("mse", args.mse_lr, args.stage_epochs),
            ("q_aware", args.qaware_lr, args.stage_epochs),
            ("functional", args.functional_lr, args.stage_epochs),
        ]
    raise ValueError(f"Unknown regime: {regime}")


def sample_batch_tensors(sender, receiver, tokenizer, row, args):
    example = build_example(tokenizer, row, args.max_context_tokens)
    context_ids = example["context_ids"].to(args.device_obj)
    tail_ids = example["tail_ids"].to(args.device_obj)
    answer_ids = example["answer_ids"].to(args.device_obj)
    query_len = example["query_ids"].shape[1]
    answer_len = answer_ids.shape[1]
    with torch.no_grad():
        sender_pairs = extract_cache(
            sender(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values
        )
        native_pairs = extract_cache(
            receiver(input_ids=context_ids, use_cache=True, logits_to_keep=1).past_key_values
        )
        native_logits, query_states, _ = tail_logits(
            receiver,
            native_pairs,
            tail_ids,
            query_len,
            answer_len,
            capture_q=True,
        )
    return example, tail_ids, answer_ids, query_len, answer_len, sender_pairs, native_pairs, native_logits.detach(), query_states


def train_stage(sender, receiver, tokenizer, translator, rows, optimizer, args, stage_name, stage_index, history, global_step):
    stage_rows = list(rows)
    random.Random(args.seed + stage_index).shuffle(stage_rows)
    translator.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(stage_rows, desc=f"{args.regime}:{stage_name}")
    for index, row in enumerate(progress):
        (
            _example,
            tail_ids,
            answer_ids,
            query_len,
            answer_len,
            sender_pairs,
            native_pairs,
            native_logits,
            query_states,
        ) = sample_batch_tensors(sender, receiver, tokenizer, row, args)

        translated_pairs = translator(sender_pairs)
        mse = normalized_kv_mse(native_pairs, translated_pairs)
        ce = torch.zeros((), device=args.device_obj)
        logit_kl = torch.zeros((), device=args.device_obj)
        route_loss = torch.zeros((), device=args.device_obj)
        readout_loss = torch.zeros((), device=args.device_obj)
        readout_mse = torch.zeros((), device=args.device_obj)
        readout_cos_loss = torch.zeros((), device=args.device_obj)

        if stage_name == "mse":
            loss = mse
        elif stage_name == "ce":
            ce = answer_ce(receiver, translated_pairs, tail_ids, query_len, answer_ids)
            loss = ce + args.ce_weak_mse_weight * mse
        elif stage_name == "q_aware":
            losses = q_aware_losses(
                receiver,
                query_states,
                native_pairs,
                translated_pairs,
                query_len,
                answer_len,
                args.route_loss,
            )
            route_loss = losses["route_loss"]
            readout_loss = losses["readout_loss"]
            readout_mse = losses["readout_mse"]
            readout_cos_loss = losses["readout_cos_loss"]
            loss = (
                args.qaware_route_weight * route_loss
                + args.qaware_readout_weight * readout_loss
                + args.qaware_weak_mse_weight * mse
            )
        elif stage_name == "functional":
            translated_logits, _, _ = tail_logits(
                receiver,
                translated_pairs,
                tail_ids,
                query_len,
                answer_len,
                capture_q=False,
            )
            ce = F.cross_entropy(
                translated_logits.float().reshape(-1, translated_logits.shape[-1]),
                answer_ids.reshape(-1),
            )
            logit_kl = logit_kl_loss(native_logits, translated_logits)
            losses = q_aware_losses(
                receiver,
                query_states,
                native_pairs,
                translated_pairs,
                query_len,
                answer_len,
                args.route_loss,
            )
            route_loss = losses["route_loss"]
            readout_loss = losses["readout_loss"]
            readout_mse = losses["readout_mse"]
            readout_cos_loss = losses["readout_cos_loss"]
            loss = (
                args.functional_ce_weight * ce
                + args.functional_logit_kl_weight * logit_kl
                + args.functional_readout_weight * readout_loss
                + args.functional_weak_mse_weight * mse
            )
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

        (loss / args.grad_accum).backward()
        if (index + 1) % args.grad_accum == 0 or index + 1 == len(stage_rows):
            torch.nn.utils.clip_grad_norm_(translator.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        item = {
            "regime": args.regime,
            "stage": stage_name,
            "stage_index": stage_index,
            "sample": index,
            "step": global_step,
            "loss": float(loss.detach().item()),
            "mse": float(mse.detach().item()),
            "ce": float(ce.detach().item()),
            "logit_kl": float(logit_kl.detach().item()),
            "route_loss": float(route_loss.detach().item()),
            "readout_loss": float(readout_loss.detach().item()),
            "readout_mse": float(readout_mse.detach().item()),
            "readout_cos_loss": float(readout_cos_loss.detach().item()),
        }
        history.append(item)
        progress.set_postfix(loss=f"{item['loss']:.3f}", mse=f"{item['mse']:.3f}", ce=f"{item['ce']:.3f}")
    return global_step


@torch.no_grad()
def validate(sender, receiver, tokenizer, translator, rows, args):
    translator.eval()
    totals = {"mse": [], "ce": [], "logit_kl": [], "route_loss": [], "readout_loss": []}
    for row in rows:
        (
            _example,
            tail_ids,
            answer_ids,
            query_len,
            answer_len,
            sender_pairs,
            native_pairs,
            native_logits,
            query_states,
        ) = sample_batch_tensors(sender, receiver, tokenizer, row, args)
        translated_pairs = translator(sender_pairs)
        totals["mse"].append(normalized_kv_mse(native_pairs, translated_pairs).item())
        translated_logits, _, _ = tail_logits(receiver, translated_pairs, tail_ids, query_len, answer_len)
        totals["ce"].append(
            F.cross_entropy(
                translated_logits.float().reshape(-1, translated_logits.shape[-1]),
                answer_ids.reshape(-1),
            ).item()
        )
        totals["logit_kl"].append(logit_kl_loss(native_logits, translated_logits).item())
        losses = q_aware_losses(
            receiver,
            query_states,
            native_pairs,
            translated_pairs,
            query_len,
            answer_len,
            args.route_loss,
        )
        totals["route_loss"].append(losses["route_loss"].item())
        totals["readout_loss"].append(losses["readout_loss"].item())
    translator.train()
    return {f"val_{key}": float(np.mean(values)) for key, values in totals.items()} if rows else {}


def main():
    parser = argparse.ArgumentParser(description="Train q-aware functional real cross-model KV translator")
    parser.add_argument("--sender-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B")
    parser.add_argument("--receiver-model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--train-data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl")
    parser.add_argument("--val-data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--regime", choices=["mse_only", "mse_then_ce", "q_aware_functional"], required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--gate-init", type=float, default=2.0)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--stage-epochs", type=int, default=1)
    parser.add_argument("--mse-lr", type=float, default=2e-4)
    parser.add_argument("--ce-lr", type=float, default=5e-5)
    parser.add_argument("--qaware-lr", type=float, default=1e-4)
    parser.add_argument("--functional-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--route-loss", choices=["js", "kl"], default="js")
    parser.add_argument("--qaware-route-weight", type=float, default=1.0)
    parser.add_argument("--qaware-readout-weight", type=float, default=1.0)
    parser.add_argument("--qaware-weak-mse-weight", type=float, default=0.05)
    parser.add_argument("--functional-ce-weight", type=float, default=1.0)
    parser.add_argument("--functional-logit-kl-weight", type=float, default=0.5)
    parser.add_argument("--functional-readout-weight", type=float, default=0.25)
    parser.add_argument("--functional-weak-mse-weight", type=float, default=0.02)
    parser.add_argument("--ce-weak-mse-weight", type=float, default=0.0)
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
    assert_tokenizer_compatible(
        sender_tokenizer,
        receiver_tokenizer,
        check_rows,
        args.max_context_tokens,
        args.tokenizer_check_samples,
    )
    tokenizer = receiver_tokenizer

    sender = load_model(args.sender_model, dtype, args.device_obj)
    receiver = load_model(args.receiver_model, dtype, args.device_obj, eager=True)
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

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
    atol = args.equivalence_atol if args.equivalence_atol is not None else (0.5 if args.dtype == "float16" else 1e-3)
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

    translator = build_translator(sender.config, receiver.config, args, args.device_obj).train()
    history = []
    global_step = 0
    stage_plan = regime_stages(args.regime, args)
    for stage_index, (stage_name, lr, epochs) in enumerate(stage_plan):
        optimizer = torch.optim.AdamW(translator.parameters(), lr=lr, weight_decay=args.weight_decay)
        for _epoch in range(epochs):
            global_step = train_stage(
                sender,
                receiver,
                tokenizer,
                translator,
                train_rows,
                optimizer,
                args,
                stage_name,
                stage_index,
                history,
                global_step,
            )
        validation = validate(sender, receiver, tokenizer, translator, val_rows, args)
        metadata = {
            "args": {k: v for k, v in vars(args).items() if k != "device_obj"},
            "stage_index": stage_index,
            "stage_name": stage_name,
            "global_step": global_step,
            "stage_plan": [
                {"stage": item[0], "learning_rate": item[1], "epochs": item[2]}
                for item in stage_plan
            ],
            "budget_policy": "same stage count and sample-pass budget for all regimes by default",
            "rope_checks": rope_checks,
            "native_context_cache_equivalence": equivalence,
            "translator_config": translator.config_dict(),
            **validation,
        }
        save_real_translator(output / f"checkpoint_stage{stage_index + 1}.pt", translator, metadata)
        with open(output / "train_history.jsonl", "w", encoding="utf-8") as handle:
            for item in history:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(output / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

    save_real_translator(output / "checkpoint_final.pt", translator, metadata)


if __name__ == "__main__":
    main()
