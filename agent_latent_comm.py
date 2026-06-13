import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared_latent_readout import SharedMemoryReader, get_layers, masked_logits, teacher_topk_match


def load_hotpot(path: str, limit: int):
    rows = json.load(open(path, encoding="utf-8"))
    out = []
    for row in rows:
        if row.get("context") and row.get("question") and row.get("answer"):
            out.append(row)
        if limit and len(out) >= limit:
            break
    return out


def flatten_context(row: Dict, max_chars: int):
    chunks = []
    for title, sentences in row["context"]:
        text = " ".join(s.strip() for s in sentences if s.strip())
        if text:
            chunks.append(f"[{title}] {text}")
    return "\n".join(chunks)[:max_chars]


def evidence_message(row: Dict):
    by_title = {title: sentences for title, sentences in row["context"]}
    lines = []
    seen = set()
    for title, idx in row.get("supporting_facts", []):
        key = (title, idx)
        if key in seen:
            continue
        seen.add(key)
        sentences = by_title.get(title, [])
        if 0 <= idx < len(sentences):
            lines.append(f"[{title}] {sentences[idx].strip()}")
    return "\n".join(lines) if lines else ""


def prompt_cq(row: Dict, max_context_chars: int):
    return f"Context:\n{flatten_context(row, max_context_chars)}\n\nQuestion:\n{row['question']}\n\nAnswer:"


def prompt_cq_message(row: Dict, max_context_chars: int):
    msg = evidence_message(row)
    return (
        f"Context:\n{flatten_context(row, max_context_chars)}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Sender evidence message:\n{msg}\n\nAnswer:"
    )


def tokenize_prompt_answer(tokenizer, prompt: str, answer: str, device, max_length: int):
    text = prompt + " " + answer
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids
    prompt_len = min(prompt_ids.shape[1], enc.input_ids.shape[1] - 1)
    labels = enc.input_ids.clone()
    labels[:, :prompt_len] = -100
    return enc.input_ids, enc.attention_mask, labels, prompt_len


class AttnInputCapture:
    def __init__(self, module):
        self.input = None
        self._pre = module.register_forward_pre_hook(self._save, with_kwargs=True)

    def _save(self, module, args, kwargs):
        self.input = kwargs.get("hidden_states", args[0] if args else None).detach()

    def close(self):
        self._pre.remove()


def self_attn(model, layer: int):
    return get_layers(model)[layer].self_attn


def projected_qk_saliency(model, layer: int, hidden, hist_len: int):
    attn = self_attn(model, layer)
    q = attn.q_proj(hidden[:, hist_len:, :])
    k = attn.k_proj(hidden[:, :hist_len, :])
    cfg = model.config
    n_heads = getattr(cfg, "num_attention_heads")
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = q.shape[-1] // n_heads
    q = q.view(q.shape[0], q.shape[1], n_heads, head_dim)
    k = k.view(k.shape[0], k.shape[1], n_kv_heads, head_dim)
    if n_heads != n_kv_heads:
        repeat = n_heads // n_kv_heads
        k = k.repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bqhd,bkhd->bqhk", q, k) / math.sqrt(head_dim)
    return scores.mean(dim=(1, 2)).softmax(dim=-1)


@torch.no_grad()
def build_latent_message(sender, tokenizer, row: Dict, layer: int, topk: int, device, max_length: int, fake_tokens: int, max_context_chars: int):
    base_prompt = prompt_cq(row, max_context_chars)
    fake_suffix = " " + " ".join(["evidence"] * fake_tokens)
    base_ids = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    full_ids = tokenizer(base_prompt + fake_suffix, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    hist_len = min(base_ids.input_ids.shape[1], full_ids.input_ids.shape[1] - 1)
    cap = AttnInputCapture(self_attn(sender, layer))
    sender(**full_ids, use_cache=False)
    hidden = cap.input
    cap.close()
    sal = projected_qk_saliency(sender, layer, hidden, hist_len)
    attn = self_attn(sender, layer)
    hist_hidden = hidden[:, :hist_len, :]
    k = attn.k_proj(hist_hidden)
    v = attn.v_proj(hist_hidden)
    keep = min(topk, hist_len)
    idx = torch.topk(sal[0], keep).indices.sort().values
    return {
        "k": k[:, idx, :].float(),
        "v": v[:, idx, :].float(),
        "hidden": hist_hidden[:, idx, :].float(),
        "position": idx.view(1, -1).to(device),
        "saliency": sal[:, idx].unsqueeze(-1).float(),
    }


def run_receiver_plain(receiver, tokenizer, row, prompt_fn, device, max_length, max_context_chars):
    ids, mask, labels, _ = tokenize_prompt_answer(tokenizer, prompt_fn(row, max_context_chars), row["answer"], device, max_length)
    return receiver(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False), labels


def run_receiver_latent(receiver, tokenizer, row, reader, memory, layer, alpha, device, max_length, max_context_chars):
    ids, mask, labels, prompt_len = tokenize_prompt_answer(tokenizer, prompt_cq(row, max_context_chars), row["answer"], device, max_length)
    target_layer = get_layers(receiver)[layer]
    cache = {}

    def hook(module, args, kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        target = hidden[:, prompt_len - 1 : -1, :]
        patch = reader(target.float(), memory).to(hidden.dtype)
        n = min(target.shape[1], patch.shape[1])
        mixed = hidden.clone()
        mixed[:, prompt_len - 1 : prompt_len - 1 + n, :] = target[:, :n, :] + alpha * patch[:, :n, :]
        cache["patch"] = patch.float()
        if isinstance(output, tuple):
            return (mixed,) + output[1:]
        return mixed

    handle = target_layer.register_forward_hook(hook, with_kwargs=True)
    out = receiver(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    handle.remove()
    return out, labels, cache["patch"]


@torch.no_grad()
def evaluate(args, sender, receiver, tok_s, tok_r, reader, rows, device):
    eval_rows = []
    memories = [build_latent_message(sender, tok_s, row, args.layer, args.topk, device, args.max_length, args.fake_tokens, args.max_context_chars) for row in rows]
    for i, row in enumerate(tqdm(rows, desc="eval")):
        no_comm, labels = run_receiver_plain(receiver, tok_r, row, prompt_cq, device, args.max_length, args.max_context_chars)
        text_comm, text_labels = run_receiver_plain(receiver, tok_r, row, prompt_cq_message, device, args.max_length, args.max_context_chars)
        latent, latent_labels, _ = run_receiver_latent(receiver, tok_r, row, reader, memories[i], args.layer, args.alpha, device, args.max_length, args.max_context_chars)
        shuffled, shuffled_labels, _ = run_receiver_latent(receiver, tok_r, row, reader, memories[(i + 1) % len(memories)], args.layer, args.alpha, device, args.max_length, args.max_context_chars)

        n_log, n_lab = masked_logits(no_comm.logits.float(), labels)
        t_log, t_lab = masked_logits(text_comm.logits.float(), text_labels)
        l_log, l_lab = masked_logits(latent.logits.float(), latent_labels)
        s_log, s_lab = masked_logits(shuffled.logits.float(), shuffled_labels)
        n = min(n_log.shape[0], t_log.shape[0], l_log.shape[0], s_log.shape[0])
        eval_rows.append({
            "no_comm_ce": float(F.cross_entropy(n_log[:n], n_lab[:n]).cpu()),
            "text_comm_ce": float(F.cross_entropy(t_log[:n], t_lab[:n]).cpu()),
            "latent_comm_ce": float(F.cross_entropy(l_log[:n], l_lab[:n]).cpu()),
            "shuffled_latent_ce": float(F.cross_entropy(s_log[:n], s_lab[:n]).cpu()),
            "latent_vs_text_kl": float(F.kl_div(F.log_softmax(l_log[:n], dim=-1), F.softmax(t_log[:n], dim=-1), reduction="batchmean").cpu()),
            "no_comm_vs_text_kl": float(F.kl_div(F.log_softmax(n_log[:n], dim=-1), F.softmax(t_log[:n], dim=-1), reduction="batchmean").cpu()),
            "latent_top1_text_match": float((l_log[:n].argmax(dim=-1) == t_log[:n].argmax(dim=-1)).float().mean().cpu()),
            "latent_top5_text_match": float(teacher_topk_match(l_log[:n], t_log[:n], 5).cpu()),
            "latent_top10_text_match": float(teacher_topk_match(l_log[:n], t_log[:n], 10).cpu()),
            "no_comm_top1_text_match": float((n_log[:n].argmax(dim=-1) == t_log[:n].argmax(dim=-1)).float().mean().cpu()),
            "shuffled_top1_text_match": float((s_log[:n].argmax(dim=-1) == t_log[:n].argmax(dim=-1)).float().mean().cpu()),
        })
    return eval_rows


def mean_numeric(rows: List[Dict]):
    return {k: sum(float(r[k]) for r in rows) / len(rows) for k, v in rows[0].items() if isinstance(v, (int, float))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sender-model", default="Qwen3-0.6B")
    p.add_argument("--receiver-model", default="Qwen3-1.7B")
    p.add_argument("--data", default="/home/yezhe/数据集/HotpotQA/raw/hotpot_dev_distractor_v1.json")
    p.add_argument("--out", default="runs/agent_latent_comm")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--fake-tokens", type=int, default=16)
    p.add_argument("--max-context-chars", type=int, default=2500)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--eval-samples", type=int, default=16)
    p.add_argument("--reader-hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--w-kl", type=float, default=0.05)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device.type == "cuda" else torch.float32
    tok_s = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    tok_r = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    if tok_s.pad_token is None:
        tok_s.pad_token = tok_s.eos_token
    if tok_r.pad_token is None:
        tok_r.pad_token = tok_r.eos_token
    sender = AutoModelForCausalLM.from_pretrained(args.sender_model, dtype=dtype, trust_remote_code=True).to(device).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver_model, dtype=dtype, trust_remote_code=True).to(device).eval()
    for p_ in sender.parameters():
        p_.requires_grad_(False)
    for p_ in receiver.parameters():
        p_.requires_grad_(False)

    attn = self_attn(sender, args.layer)
    reader = SharedMemoryReader(receiver.config.hidden_size, attn.k_proj.out_features, attn.v_proj.out_features, sender.config.hidden_size, args.reader_hidden).to(device)
    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr)
    rows = load_hotpot(args.data, args.max_samples + args.eval_samples)
    train_rows = rows[: args.max_samples]
    eval_rows_src = rows[args.max_samples : args.max_samples + args.eval_samples]

    train_metrics = []
    for epoch in range(args.epochs):
        for row in tqdm(train_rows, desc=f"epoch {epoch}"):
            memory = build_latent_message(sender, tok_s, row, args.layer, args.topk, device, args.max_length, args.fake_tokens, args.max_context_chars)
            with torch.no_grad():
                text_comm, text_labels = run_receiver_plain(receiver, tok_r, row, prompt_cq_message, device, args.max_length, args.max_context_chars)
            latent, latent_labels, _ = run_receiver_latent(receiver, tok_r, row, reader, memory, args.layer, args.alpha, device, args.max_length, args.max_context_chars)
            l_log, l_lab = masked_logits(latent.logits.float(), latent_labels)
            t_log, _ = masked_logits(text_comm.logits.float(), text_labels)
            n = min(l_log.shape[0], t_log.shape[0])
            ce = F.cross_entropy(l_log[:n], l_lab[:n])
            kl = F.kl_div(F.log_softmax(l_log[:n], dim=-1), F.softmax(t_log[:n], dim=-1), reduction="batchmean")
            loss = ce + args.w_kl * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            opt.step()
            train_metrics.append({"loss": float(loss.detach().cpu()), "ce": float(ce.detach().cpu()), "text_kl": float(kl.detach().cpu())})

    torch.save({"reader": reader.state_dict(), "args": vars(args)}, Path(args.out) / "reader.pt")
    eval_rows = evaluate(args, sender, receiver, tok_s, tok_r, reader, eval_rows_src, device)
    summary = mean_numeric(eval_rows)
    summary.update({"layer": args.layer, "topk": args.topk, "fake_tokens": args.fake_tokens, "alpha": args.alpha})
    with open(Path(args.out) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "train": train_metrics, "eval_rows": eval_rows, "summary": summary}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
