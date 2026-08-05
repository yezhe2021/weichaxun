"""阶段0：Tokenizer 与协议审计（方案 §四）。

逐样本显式检查 Sender / Receiver 对同一文本产出完全一致的：
  full_input_ids / context_input_ids / question_suffix_ids / question_only_ids / answer_token_ids
并比较 special tokens 与 chat template 结构。

输出 artifacts/token_protocol_audit.json，要求 all_samples_identical = true。
若不成立，说明 token 序列不一致，无法做逐 token KV 目标对齐，应停止。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from protocol import SYSTEM, load_json, progress, render, save_json, tokenizer


def audit_pair(sender_path, receiver_path, data_path, max_samples, max_answer_tokens):
    sender_tok = tokenizer(sender_path)
    receiver_tok = tokenizer(receiver_path)
    raws = load_json(data_path)[:max_samples]

    checks = {name: [] for name in ("full_input_ids", "context_input_ids", "question_suffix_ids", "question_only_ids")}
    mismatches = []
    answer_ids_match = True
    for index, raw in enumerate(raws):
        sender = render(sender_tok, raw, True)
        receiver = render(receiver_tok, raw, True)
        qonly_s = render(sender_tok, raw, False)
        qonly_r = render(receiver_tok, raw, False)
        for name in checks:
            s = sender.get(name) or qonly_s.get(name)
            r = receiver.get(name) or qonly_r.get(name)
            ok = (s == r)
            checks[name].append(ok)
            if not ok:
                mismatches.append({"sample": raw["_id"], "check": name})
        answer_s = sender_tok(raw["answer"], add_special_tokens=False).input_ids[:max_answer_tokens]
        answer_r = receiver_tok(raw["answer"], add_special_tokens=False).input_ids[:max_answer_tokens]
        if answer_s != answer_r:
            answer_ids_match = False
            mismatches.append({"sample": raw["_id"], "check": "answer_token_ids"})
        if (index + 1) % 64 == 0:
            progress(f"audit {index + 1}/{len(raws)}")

    # special tokens 与 chat template 结构（BOS/EOS/PAD + im_start/im_end/thinking）
    def token_summary(tok):
        return {
            "bos_token_id": tok.bos_token_id,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "chat_template_has_thinking": "enable_thinking" in (tok.chat_template or ""),
        }

    return {
        "sender": sender_path,
        "receiver": receiver_path,
        "n_samples": len(raws),
        "checks": {name: all(values) for name, values in checks.items()},
        "answer_token_ids_match": answer_ids_match,
        "special_tokens": {"sender": token_summary(sender_tok), "receiver": token_summary(receiver_tok)},
        "special_tokens_identical": token_summary(sender_tok) == token_summary(receiver_tok),
        "mismatches": mismatches[:50],
        "all_samples_identical": (
            all(all(values) for values in checks.values())
            and answer_ids_match
            and token_summary(sender_tok) == token_summary(receiver_tok)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="阶段0：Tokenizer 与协议审计")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--data", default="/home/yezhe/数据集/HotpotQA/raw/hotpot_dev_distractor_v1.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    args = parser.parse_args()

    result = audit_pair(args.sender_model, args.receiver_model, args.data, args.max_samples, args.max_answer_tokens)
    save_json(args.out, result)
    progress(f"all_samples_identical = {result['all_samples_identical']}")
    print({"all_samples_identical": result["all_samples_identical"], "checks": result["checks"]})


if __name__ == "__main__":
    main()
