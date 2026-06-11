import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from transformers.utils import import_utils as _transformers_import_utils

_transformers_import_utils._torchvision_available = False

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Example:
    context: str
    question: str
    answer: str


class JsonInstructionDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        self.examples = list(self._load(path))
        if max_samples is not None:
            self.examples = self.examples[:max_samples]

    def _load(self, path: str) -> Iterable[Example]:
        with open(path, "r", encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            rows = json.load(f) if first == "[" else [json.loads(line) for line in f if line.strip()]
        for row in rows:
            if "conversations" in row:
                turns = row.get("conversations") or []
                human = [t.get("value", "") for t in turns if t.get("from") in {"human", "user"}]
                assistant = [t.get("value", "") for t in turns if t.get("from") in {"gpt", "assistant"}]
                if human and assistant:
                    yield Example(context="", question=str(human[-1]), answer=str(assistant[-1]))
                continue
            instruction = row.get("instruction") or row.get("query") or row.get("question") or ""
            inp = row.get("input") or row.get("context") or ""
            answer = row.get("output") or row.get("answer") or row.get("response") or ""
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            if not instruction or not answer:
                continue
            yield Example(context=str(inp), question=str(instruction), answer=str(answer))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_one(batch: List[Example]) -> Example:
    return batch[0]


def format_full(ex: Example) -> str:
    if ex.context.strip():
        return f"Context:\n{ex.context}\n\nQuestion:\n{ex.question}\n\nAnswer:"
    return f"Question:\n{ex.question}\n\nAnswer:"


def format_query(ex: Example) -> str:
    return f"Question:\n{ex.question}\n\nAnswer:"


def tokenize_prompt_answer(tokenizer, prompt: str, answer: str, device: torch.device, max_length: int):
    text = prompt + " " + answer
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids
    prompt_len = min(prompt_ids.shape[1], enc.input_ids.shape[1] - 1)
    labels = enc.input_ids.clone()
    labels[:, :prompt_len] = -100
    return enc.input_ids, enc.attention_mask, labels, prompt_len


def get_layers(model):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    raise ValueError("Unsupported model: expected a Llama/Qwen-style .model.layers stack")


def self_attn_of(model, layer_idx: int):
    return get_layers(model)[layer_idx].self_attn


class ActivationCapture:
    def __init__(self, module: nn.Module):
        self.input = None
        self.output = None
        self._pre = module.register_forward_pre_hook(self._save_input, with_kwargs=True)
        self._post = module.register_forward_hook(self._save_output, with_kwargs=True)

    def _save_input(self, module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        self.input = hidden_states.detach()

    def _save_output(self, module, args, kwargs, output):
        self.output = output[0].detach() if isinstance(output, tuple) else output.detach()

    def close(self):
        self._pre.remove()
        self._post.remove()


class SharedMemoryReader(nn.Module):
    def __init__(self, receiver_dim: int, sender_k_dim: int, sender_v_dim: int, sender_hidden_dim: int, hidden: int, pos_dim: int = 32):
        super().__init__()
        self.pos = nn.Embedding(8192, pos_dim)
        self.mem_in = nn.Linear(sender_k_dim + sender_v_dim + sender_hidden_dim + pos_dim + 1, hidden)
        self.q = nn.Linear(receiver_dim, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, receiver_dim)
        self.out = nn.Sequential(nn.LayerNorm(receiver_dim), nn.Linear(receiver_dim, receiver_dim))

    def forward(self, receiver_q_hidden, memory):
        pos = self.pos(memory["position"].clamp_max(self.pos.num_embeddings - 1))
        mem = torch.cat([memory["k"], memory["v"], memory["hidden"], pos, memory["saliency"]], dim=-1)
        mem = torch.tanh(self.mem_in(mem))
        q = self.q(receiver_q_hidden)
        k = self.k(mem)
        v = self.v(mem)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
        return self.out(torch.matmul(attn, v))


def infer_sender_kv(sender, layer_idx: int, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    attn = self_attn_of(sender, layer_idx)
    k = attn.k_proj(hidden)
    v = attn.v_proj(hidden)
    return k, v


def token_saliency(hidden: torch.Tensor, query_start: int) -> torch.Tensor:
    query = hidden[:, query_start:, :].mean(dim=1, keepdim=True)
    return F.cosine_similarity(hidden, query, dim=-1).clamp_min(0)


@torch.no_grad()
def build_shared_memory(sender, tokenizer, ex: Example, layer_idx: int, topk: int, device, max_length: int):
    prompt = format_full(ex)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    q_len = tokenizer(format_query(ex), return_tensors="pt", truncation=True, max_length=max_length).input_ids.shape[1]
    query_start = max(ids.input_ids.shape[1] - q_len, 0)
    cap = ActivationCapture(self_attn_of(sender, layer_idx))
    sender(**ids, use_cache=False)
    hidden = cap.input
    cap.close()
    k, v = infer_sender_kv(sender, layer_idx, hidden)
    sal = token_saliency(hidden, query_start)
    keep = min(topk, hidden.shape[1])
    idx = torch.topk(sal[0], keep).indices.sort().values
    pooled = hidden.mean(dim=1, keepdim=True).expand(-1, keep, -1)
    return {
        "k": k[:, idx, :].float(),
        "v": v[:, idx, :].float(),
        "hidden": pooled.float(),
        "position": idx.view(1, -1).to(device),
        "saliency": sal[:, idx].unsqueeze(-1).float(),
    }


@torch.no_grad()
def capture_receiver_teacher(receiver, tokenizer, ex: Example, layer_idx: int, device, max_length: int):
    ids, mask, labels, prompt_len = tokenize_prompt_answer(tokenizer, format_full(ex), ex.answer, device, max_length)
    cap = ActivationCapture(self_attn_of(receiver, layer_idx))
    out = receiver(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    attn_out = cap.output[:, prompt_len - 1 : -1, :].float()
    q_hidden = cap.input[:, prompt_len - 1 : -1, :].float()
    cap.close()
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": labels,
        "teacher_attn": attn_out,
        "teacher_q_hidden": q_hidden,
        "teacher_logits": out.logits.float(),
        "teacher_ce": out.loss.detach().float(),
    }


def run_receiver_patched(receiver, tokenizer, ex: Example, layer_idx: int, reader, memory, alpha, device, max_length: int):
    ids, mask, labels, prompt_len = tokenize_prompt_answer(tokenizer, format_query(ex), ex.answer, device, max_length)
    patch_cache = {}
    module = self_attn_of(receiver, layer_idx)

    def hook(mod, args, kwargs, output):
        attn_out = output[0] if isinstance(output, tuple) else output
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        q_hidden = hidden_states[:, prompt_len - 1 : -1, :]
        patch = reader(q_hidden.float(), memory).to(attn_out.dtype)
        mixed = attn_out.clone()
        mixed[:, prompt_len - 1 : -1, :] = alpha * patch + (1.0 - alpha) * mixed[:, prompt_len - 1 : -1, :]
        patch_cache["patch"] = patch.float()
        if isinstance(output, tuple):
            return (mixed,) + output[1:]
        return mixed

    handle = module.register_forward_hook(hook, with_kwargs=True)
    out = receiver(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    handle.remove()
    return out, patch_cache["patch"], labels


@torch.no_grad()
def run_receiver_no_context(receiver, tokenizer, ex: Example, device, max_length: int):
    ids, mask, labels, _ = tokenize_prompt_answer(tokenizer, format_query(ex), ex.answer, device, max_length)
    return receiver(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)


def masked_logits(logits, labels):
    pred = logits[:, :-1, :]
    lab = labels[:, 1:]
    mask = lab.ne(-100)
    return pred[mask], lab[mask]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tok_r = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    if tok_r.pad_token is None:
        tok_r.pad_token = tok_r.eos_token
    if tok_s.pad_token is None:
        tok_s.pad_token = tok_s.eos_token

    sender = AutoModelForCausalLM.from_pretrained(args.sender_model, dtype=dtype, device_map=None, trust_remote_code=True).to(device).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver_model, dtype=dtype, device_map=None, trust_remote_code=True).to(device).eval()
    for p in sender.parameters():
        p.requires_grad_(False)
    for p in receiver.parameters():
        p.requires_grad_(False)

    receiver_dim = receiver.config.hidden_size
    sender_hidden_dim = sender.config.hidden_size
    sender_attn = self_attn_of(sender, args.layer)
    sender_k_dim = sender_attn.k_proj.out_features
    sender_v_dim = sender_attn.v_proj.out_features
    reader = SharedMemoryReader(receiver_dim, sender_k_dim, sender_v_dim, sender_hidden_dim, args.reader_hidden).to(device)
    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = JsonInstructionDataset(args.data, args.max_samples)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_one)

    os.makedirs(args.out, exist_ok=True)
    metrics = []
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for step, ex in enumerate(pbar):
            memory = build_shared_memory(sender, tok_s, ex, args.layer, args.topk, device, args.max_length)
            teacher = capture_receiver_teacher(receiver, tok_r, ex, args.layer, device, args.max_length)
            patched, patch, labels = run_receiver_patched(receiver, tok_r, ex, args.layer, reader, memory, args.alpha, device, args.max_length)

            attn_n = min(teacher["teacher_attn"].shape[1], patch.shape[1])
            patch_for_loss = patch[:, :attn_n, :]
            t_attn = teacher["teacher_attn"][:, :attn_n, :]
            mse = F.mse_loss(patch_for_loss, t_attn)
            cosine = 1.0 - F.cosine_similarity(patch_for_loss.flatten(0, 1), t_attn.flatten(0, 1), dim=-1).mean()
            p_log, p_lab = masked_logits(patched.logits.float(), labels)
            t_log, _ = masked_logits(teacher["teacher_logits"].float(), teacher["labels"])
            n = min(p_log.shape[0], t_log.shape[0])
            kl = F.kl_div(F.log_softmax(p_log[:n], dim=-1), F.softmax(t_log[:n], dim=-1), reduction="batchmean")
            ce = F.cross_entropy(p_log, p_lab)
            loss = args.w_mse * mse + args.w_cos * cosine + args.w_kl * kl + args.w_ce * ce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), args.grad_clip)
            opt.step()

            row = {
                "epoch": epoch,
                "step": step,
                "loss": float(loss.detach().cpu()),
                "mse": float(mse.detach().cpu()),
                "cosine_loss": float(cosine.detach().cpu()),
                "kl": float(kl.detach().cpu()),
                "patched_ce": float(ce.detach().cpu()),
                "teacher_ce": float(teacher["teacher_ce"].cpu()),
            }
            metrics.append(row)
            pbar.set_postfix({k: round(row[k], 4) for k in ["loss", "mse", "kl", "patched_ce"]})

    torch.save({"reader": reader.state_dict(), "args": vars(args)}, Path(args.out) / "reader.pt")
    evaluate(args, sender, receiver, tok_s, tok_r, reader, ds, device, metrics)


@torch.no_grad()
def evaluate(args, sender, receiver, tok_s, tok_r, reader, ds, device, train_metrics):
    rows = []
    peak_start = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    for ex in tqdm(list(ds)[: args.eval_samples], desc="eval"):
        memory = build_shared_memory(sender, tok_s, ex, args.layer, args.topk, device, args.max_length)
        teacher = capture_receiver_teacher(receiver, tok_r, ex, args.layer, device, args.max_length)
        noctx = run_receiver_no_context(receiver, tok_r, ex, device, args.max_length)
        patched, patch, labels = run_receiver_patched(receiver, tok_r, ex, args.layer, reader, memory, args.alpha, device, args.max_length)
        p_log, p_lab = masked_logits(patched.logits.float(), labels)
        n_log, _ = masked_logits(noctx.logits.float(), labels)
        t_log, _ = masked_logits(teacher["teacher_logits"].float(), teacher["labels"])
        n = min(p_log.shape[0], t_log.shape[0], n_log.shape[0])
        rows.append({
            "teacher_ce": float(teacher["teacher_ce"].cpu()),
            "no_context_ce": float(F.cross_entropy(n_log[:n], p_lab[:n]).cpu()),
            "patched_ce": float(F.cross_entropy(p_log[:n], p_lab[:n]).cpu()),
            "patched_kl": float(F.kl_div(F.log_softmax(p_log[:n], dim=-1), F.softmax(t_log[:n], dim=-1), reduction="batchmean").cpu()),
            "no_context_kl": float(F.kl_div(F.log_softmax(n_log[:n], dim=-1), F.softmax(t_log[:n], dim=-1), reduction="batchmean").cpu()),
            "patched_top1_match": float((p_log[:n].argmax(dim=-1) == t_log[:n].argmax(dim=-1)).float().mean().cpu()),
            "no_context_top1_match": float((n_log[:n].argmax(dim=-1) == t_log[:n].argmax(dim=-1)).float().mean().cpu()),
            "attn_mse": float(F.mse_loss(patch[:, : min(patch.shape[1], teacher["teacher_attn"].shape[1]), :], teacher["teacher_attn"][:, : min(patch.shape[1], teacher["teacher_attn"].shape[1]), :]).cpu()),
        })
    summary = {k: sum(r[k] for r in rows) / max(len(rows), 1) for k in rows[0]} if rows else {}
    summary["cuda_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated() - peak_start) if torch.cuda.is_available() else 0
    with open(Path(args.out) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_metrics, "eval_rows": rows, "summary": summary}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sender-model", default="/home/yezhe/伪查询/Qwen3-0.6B")
    p.add_argument("--receiver-model", default="/home/yezhe/伪查询/Qwen3-1.7B")
    p.add_argument("--data", default="/home/yezhe/数据集/swift/OpenHermes-2___5/openhermes2_5.json")
    p.add_argument("--out", default="runs/shared_latent_readout")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--topk", type=int, default=128)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--eval-samples", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--reader-hidden", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--w-mse", type=float, default=1.0)
    p.add_argument("--w-cos", type=float, default=0.2)
    p.add_argument("--w-kl", type=float, default=0.05)
    p.add_argument("--w-ce", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
