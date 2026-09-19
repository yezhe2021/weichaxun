from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DynamicCache


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda")


def source_mode(mode):
    # The audit smoke must be an 8-sample prefix of the exact formal protocol;
    # R1's historical smoke manifest contains only four test examples.
    return "formal"


def r1_module(cfg):
    path = str(Path(cfg["r1_dir"]))
    if path not in sys.path:
        sys.path.insert(0, path)
    import r1_common
    return r1_common


def load_reader(cfg, mode):
    r1 = r1_module(cfg)
    model = r1.inject_lora(r1.load_model(cfg["model_4b"]), cfg)
    checkpoint = Path(cfg["r1_dir"]) / "artifacts" / source_mode(mode) / "sparse_reader" / "best.pt"
    r1.load_lora(model, checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return r1, model, r1.tokenizer(cfg["model_4b"]), checkpoint


def rows_for(cfg, mode):
    manifest = load_json(
        Path(cfg["r1_dir"]) / "artifacts" / source_mode(mode) / "manifest.json"
    )
    count = cfg["smoke_samples"] if mode == "smoke" else cfg["formal_samples"]
    return manifest, manifest["test"][:count]


class FullNativeCapture:
    def __init__(self, model, cfg):
        self.cfg, self.values = cfg, {}
        self.handles = [
            layer.self_attn.register_forward_pre_hook(
                self._hook(index), with_kwargs=True
            )
            for index, layer in enumerate(model.model.layers)
        ]

    def _hook(self, index):
        def capture(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            batch, length, _ = hidden.shape
            key = module.k_norm(module.k_proj(hidden).view(batch, length, 8, 128))
            value = module.v_proj(hidden).view(batch, length, 8, 128)
            self.values[index] = (
                key[0].detach().clone(), value[0].detach().clone()
            )
        return capture

    def stacked(self):
        if len(self.values) != self.cfg["num_layers"]:
            raise RuntimeError(f"captured {len(self.values)} layers")
        return (
            torch.stack([self.values[i][0] for i in range(36)]),
            torch.stack([self.values[i][1] for i in range(36)]),
        )

    def close(self):
        for handle in self.handles:
            handle.remove()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), -1)


def manual_post_rope(model, pre_key, positions):
    position_ids = torch.tensor([positions], dtype=torch.long, device=pre_key.device)
    dummy = torch.empty(
        1, len(positions), model.config.hidden_size,
        dtype=pre_key.dtype, device=pre_key.device,
    )
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    cos, sin = cos[0][None, :, None], sin[0][None, :, None]
    return pre_key * cos + rotate_half(pre_key) * sin


def cache_tensors(cache):
    keys = torch.stack([
        layer.keys[0].permute(1, 0, 2) for layer in cache.layers
    ])
    values = torch.stack([
        layer.values[0].permute(1, 0, 2) for layer in cache.layers
    ])
    return keys, values


def dynamic_cache(model, key, value):
    items = [
        (
            key[layer].permute(1, 0, 2).unsqueeze(0),
            value[layer].permute(1, 0, 2).unsqueeze(0),
        )
        for layer in range(key.shape[0])
    ]
    return DynamicCache(ddp_cache_data=items, config=model.config)


def context_prefill(model, r1, sample, cfg):
    r1.set_lora(model, False)
    capture = FullNativeCapture(model, cfg)
    ids = torch.tensor(
        [sample["full_context_token_ids"]], dtype=torch.long, device=cuda()
    )
    positions = torch.arange(ids.shape[1], device=cuda()).unsqueeze(0)
    with torch.no_grad():
        output = model(
            ids, attention_mask=torch.ones_like(ids), position_ids=positions,
            use_cache=True,
        )
    pre_key, manual_value = capture.stacked()
    capture.close()
    official_key, official_value = cache_tensors(output.past_key_values)
    manual_key = manual_post_rope(
        model, pre_key, list(range(ids.shape[1]))
    )
    return official_key, official_value, manual_key, manual_value


def answer_target(r1, tok, cfg, answer):
    return r1.answer_target(tok, answer, cfg["max_answer_tokens"])


def teacher_logits_direct(cfg, r1, model, tok, sample, lora):
    r1.set_lora(model, lora)
    target = answer_target(r1, tok, cfg, sample["answer"])
    prompt = sample["full_sequence_token_ids"]
    current = torch.tensor([prompt + target[:-1]], device=cuda())
    positions = torch.arange(current.shape[1], device=cuda()).unsqueeze(0)
    with torch.no_grad():
        logits = model(
            current, attention_mask=torch.ones_like(current),
            position_ids=positions, use_cache=False,
        ).logits
    selected = logits[:, len(prompt) - 1:len(prompt) - 1 + len(target)].float()[0]
    return selected, torch.tensor(target, device=cuda())


def teacher_logits_cache(cfg, r1, model, tok, sample, key, value, lora):
    r1.set_lora(model, lora)
    target = answer_target(r1, tok, cfg, sample["answer"])
    question = sample["question_token_ids"]
    current = torch.tensor([question + target[:-1]], device=cuda())
    positions = sample["question_position_ids"] + list(
        range(sample["question_position_ids"][-1] + 1,
              sample["question_position_ids"][-1] + len(target))
    )
    prefix = key.shape[1]
    mask = torch.ones(1, prefix + current.shape[1], dtype=torch.long, device=cuda())
    with torch.no_grad():
        logits = model(
            current, attention_mask=mask,
            position_ids=torch.tensor([positions], device=cuda()),
            past_key_values=dynamic_cache(model, key, value),
            cache_position=torch.arange(prefix, prefix + current.shape[1], device=cuda()),
            use_cache=False,
        ).logits
    selected = logits[:, len(question) - 1:len(question) - 1 + len(target)].float()[0]
    return selected, torch.tensor(target, device=cuda())


def nll(logits, gold):
    return F.cross_entropy(logits, gold).item()


def compare_distributions(reference, other, gold):
    ref_logp = reference.log_softmax(-1)
    other_logp = other.log_softmax(-1)
    ref_p = ref_logp.exp()
    kl = (ref_p * (ref_logp - other_logp)).sum(-1).mean()
    ref_gold = reference.gather(1, gold[:, None])
    other_gold = other.gather(1, gold[:, None])
    return {
        "token_kl": kl.item(),
        "top1_match": (reference.argmax(-1) == other.argmax(-1)).float().mean().item(),
        "reference_gold_rank": ((reference > ref_gold).sum(-1) + 1).float().mean().item(),
        "other_gold_rank": ((other > other_gold).sum(-1) + 1).float().mean().item(),
        "max_logit_absolute_error": (reference - other).abs().max().item(),
        "logits_cosine": F.cosine_similarity(
            reference.flatten(), other.flatten(), 0
        ).item(),
    }


def generate(cfg, r1, model, tok, sample, lora, key=None, value=None, direct=False):
    r1.set_lora(model, lora)
    if direct:
        prompt = sample["full_sequence_token_ids"]
        positions = list(range(len(prompt)))
    else:
        prompt = sample["question_token_ids"]
        positions = sample["question_position_ids"]
    ids = torch.tensor([prompt], dtype=torch.long, device=cuda())
    mask = torch.ones_like(ids)
    kwargs = {}
    if key is not None:
        prefix = key.shape[1]
        mask = torch.ones(1, prefix + ids.shape[1], dtype=torch.long, device=cuda())
        kwargs = {
            "past_key_values": dynamic_cache(model, key, value),
            "cache_position": torch.arange(prefix, prefix + ids.shape[1], device=cuda()),
        }
    with torch.no_grad():
        output = model(
            ids, attention_mask=mask,
            position_ids=torch.tensor([positions], device=cuda()),
            use_cache=True, **kwargs,
        )
    past = output.past_key_values
    next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    generated, next_position = [], positions[-1] + 1
    for _ in range(cfg["max_new_tokens"]):
        token = int(next_token.item())
        if token == tok.eos_token_id:
            break
        generated.append(token)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=cuda())], 1)
        with torch.no_grad():
            output = model(
                next_token, attention_mask=mask,
                position_ids=torch.tensor([[next_position]], device=cuda()),
                cache_position=torch.tensor([past.get_seq_length()], device=cuda()),
                past_key_values=past, use_cache=True,
            )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    return tok.decode(generated, skip_special_tokens=True).strip()


def cache_layer_rows(sample_id, official_k, official_v, manual_k, manual_v):
    rows = []
    for layer in range(36):
        record = {"sample_id": sample_id, "layer": layer}
        for name, official, manual in (
            ("k", official_k[layer].float(), manual_k[layer].float()),
            ("v", official_v[layer].float(), manual_v[layer].float()),
        ):
            difference = manual - official
            record.update({
                f"{name}_cosine": F.cosine_similarity(
                    manual.flatten(), official.flatten(), 0
                ).item(),
                f"{name}_nmse": (
                    difference.square().mean() / official.square().mean().clamp_min(1e-8)
                ).item(),
                f"{name}_max_absolute_error": difference.abs().max().item(),
                f"{name}_mean_absolute_error": difference.abs().mean().item(),
                f"{name}_nan_count": int(torch.isnan(manual).sum().item()),
                f"{name}_inf_count": int(torch.isinf(manual).sum().item()),
            })
        record.update({
            "official_shape": list(official_k[layer].shape),
            "manual_shape": list(manual_k[layer].shape),
            "official_dtype": str(official_k.dtype),
            "manual_dtype": str(manual_k.dtype),
            "cache_length": official_k.shape[1],
        })
        rows.append(record)
    return rows


class TraceCapture:
    def __init__(self, model):
        self.attention, self.hidden, self.handles = {}, {}, []
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.register_forward_hook(self._attn(index)))
            self.handles.append(layer.register_forward_hook(self._layer(index)))

    def _attn(self, index):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            self.attention[index] = value[0, 0].detach().float()
        return hook

    def _layer(self, index):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            self.hidden[index] = value[0, 0].detach().float()
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def first_question_trace(model, r1, sample, key, value):
    r1.set_lora(model, False)
    capture = TraceCapture(model)
    token = torch.tensor([[sample["question_token_ids"][0]]], device=cuda())
    prefix = key.shape[1]
    with torch.no_grad():
        output = model(
            token, attention_mask=torch.ones(1, prefix + 1, dtype=torch.long, device=cuda()),
            position_ids=torch.tensor([[sample["question_position_ids"][0]]], device=cuda()),
            past_key_values=dynamic_cache(model, key, value),
            cache_position=torch.tensor([prefix], device=cuda()), use_cache=False,
        )
    capture.close()
    layer_logits = {}
    with torch.no_grad():
        for layer, hidden in capture.hidden.items():
            layer_logits[layer] = model.lm_head(
                model.model.norm(hidden[None, None].to(model.dtype))
            )[0, 0].float()
    return capture.attention, capture.hidden, layer_logits, output.logits[0, 0].float()


def compare_traces(sample_id, official, manual):
    oa, oh, ol, final_o = official
    ma, mh, ml, final_m = manual
    rows = []
    for layer in range(36):
        rows.append({
            "sample_id": sample_id, "layer": layer,
            "attention_output_cosine": F.cosine_similarity(oa[layer], ma[layer], 0).item(),
            "hidden_state_cosine": F.cosine_similarity(oh[layer], mh[layer], 0).item(),
            "layer_logits_cosine": F.cosine_similarity(ol[layer], ml[layer], 0).item(),
            "layer_top1_match": bool(ol[layer].argmax() == ml[layer].argmax()),
            "final_logits_cosine": F.cosine_similarity(final_o, final_m, 0).item(),
            "final_top1_match": bool(final_o.argmax() == final_m.argmax()),
        })
    return rows


def sparse_memory(cfg, mode, manifest, sample, r1):
    r1_cfg = dict(cfg)
    r1_cfg["work_dir"] = cfg["r1_dir"]
    store = r1.ShardStore(r1_cfg, source_mode(mode), manifest)
    return r1.memory_content(store, "test", "4b", sample, "native", "correct")


def aggregate_generation(rows):
    output = {}
    for condition in sorted({x["condition"] for x in rows}):
        selected = [x for x in rows if x["condition"] == condition]
        output[condition] = {
            "em": sum(x["em"] for x in selected) / len(selected),
            "token_f1": sum(x["token_f1"] for x in selected) / len(selected),
            "answer_nll": sum(x["answer_nll"] for x in selected) / len(selected),
            "count": len(selected),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    manifest, samples = rows_for(cfg, args.mode)
    r1, model, tok, checkpoint = load_reader(cfg, args.mode)
    root = Path(cfg["work_dir"]) / "artifacts" / args.mode
    partial = root / "partial"
    partial.mkdir(parents=True, exist_ok=True)
    names = (
        "prompt_rows", "cache_rows", "trace_rows",
        "distribution_rows", "generation_rows",
    )
    loaded = {}
    for name in names:
        path = partial / f"{name}.json"
        loaded[name] = load_json(path) if path.exists() else []
    prompt_rows = loaded["prompt_rows"]
    cache_rows = loaded["cache_rows"]
    trace_rows = loaded["trace_rows"]
    distribution_rows = loaded["distribution_rows"]
    generation_rows = loaded["generation_rows"]
    completed_ids = {x["sample_id"] for x in prompt_rows}
    conditions = [
        ("full_text_lora_off", "direct", False),
        ("official_fullcache_lora_off", "official", False),
        ("manual_fullcache_lora_off", "manual", False),
        ("official_fullcache_lora_on", "official", True),
        ("manual_fullcache_lora_on", "manual", True),
        ("native4_sparse_gold_lora_off", "sparse", False),
        ("native4_sparse_gold_lora_on", "sparse", True),
        ("full_text_lora_on_all", "direct", True),
    ]
    for index, sample in enumerate(samples, 1):
        if sample["id"] in completed_ids:
            progress(f"{args.mode}: resume skip {index}/{len(samples)}")
            continue
        context, question, full = (
            sample["full_context_token_ids"], sample["question_token_ids"],
            sample["full_sequence_token_ids"],
        )
        equal = context + question == full
        if not equal:
            raise RuntimeError(f"prompt split mismatch: {sample['id']}")
        prompt_rows.append({
            "sample_id": sample["id"], "split_equal": equal,
            "context_tokens": len(context), "question_prompt_tokens": len(question),
            "full_prompt_tokens": len(full),
            "question_start_position": sample["question_position_ids"][0],
        })
        official_k, official_v, manual_k, manual_v = context_prefill(
            model, r1, sample, cfg
        )
        cache_rows.extend(cache_layer_rows(
            sample["id"], official_k, official_v, manual_k, manual_v
        ))
        trace_rows.extend(compare_traces(
            sample["id"],
            first_question_trace(model, r1, sample, official_k, official_v),
            first_question_trace(model, r1, sample, manual_k, manual_v),
        ))
        logits_a, gold = teacher_logits_direct(cfg, r1, model, tok, sample, False)
        logits_b, _ = teacher_logits_cache(
            cfg, r1, model, tok, sample, official_k, official_v, False
        )
        logits_c, _ = teacher_logits_cache(
            cfg, r1, model, tok, sample, manual_k, manual_v, False
        )
        distribution_rows.append({
            "sample_id": sample["id"], "comparison": "A_full_text_vs_B_official_cache",
            "reference_nll": nll(logits_a, gold), "other_nll": nll(logits_b, gold),
            **compare_distributions(logits_a, logits_b, gold),
        })
        distribution_rows.append({
            "sample_id": sample["id"], "comparison": "B_official_vs_C_manual_cache",
            "reference_nll": nll(logits_b, gold), "other_nll": nll(logits_c, gold),
            **compare_distributions(logits_b, logits_c, gold),
        })
        sparse_k, sparse_v, sparse_mask = sparse_memory(
            cfg, args.mode, manifest, sample, r1
        )
        sparse_k, sparse_v = sparse_k.to(cuda()), sparse_v.to(cuda())
        for condition, family, lora in conditions:
            if family == "direct":
                prediction = generate(cfg, r1, model, tok, sample, lora, direct=True)
                logits, current_gold = teacher_logits_direct(cfg, r1, model, tok, sample, lora)
            elif family in {"official", "manual"}:
                key, value = (official_k, official_v) if family == "official" else (manual_k, manual_v)
                prediction = generate(cfg, r1, model, tok, sample, lora, key, value)
                logits, current_gold = teacher_logits_cache(
                    cfg, r1, model, tok, sample, key, value, lora
                )
            else:
                r1.set_lora(model, lora)
                prediction = r1.greedy_generate(
                    cfg, model, tok, sample, sparse_k, sparse_v, sparse_mask
                )
                answer_nll = r1.answer_loss(
                    cfg, model, tok, sample, sparse_k, sparse_v, sparse_mask
                ).item()
                logits = current_gold = None
            if family != "sparse":
                answer_nll = nll(logits, current_gold)
            generation_rows.append({
                "sample_id": sample["id"], "type": sample.get("type", "unknown"),
                "condition": condition, "answer": sample["answer"],
                "prediction": prediction,
                "em": float(r1.normalize_answer(prediction) == r1.normalize_answer(sample["answer"])),
                "token_f1": r1.token_f1(prediction, sample["answer"]),
                "answer_nll": answer_nll,
            })
        save_json(root / "progress.json", {"completed_samples": index, "total_samples": len(samples)})
        for name, value in (
            ("prompt_rows", prompt_rows), ("cache_rows", cache_rows),
            ("trace_rows", trace_rows),
            ("distribution_rows", distribution_rows),
            ("generation_rows", generation_rows),
        ):
            save_json(partial / f"{name}.json", value)
        progress(f"{args.mode}: full-cache audit {index}/{len(samples)}")
    save_json(root / "prompt_split_audit.json", prompt_rows)
    save_json(root / "cache_equivalence_per_layer.json", cache_rows)
    save_json(root / "execution_trajectory_per_layer.json", trace_rows)
    save_json(root / "answer_distribution_comparisons.json", distribution_rows)
    save_json(root / "generation_per_sample.json", generation_rows)
    save_json(root / "summary.json", {
        "generation": aggregate_generation(generation_rows),
        "samples": len(samples), "reader_checkpoint": str(checkpoint),
        "hard_gate": None,
    })
    save_json(root / "completion.json", {
        "completed": True, "training_performed": False,
        "sender_present": False, "hard_gate": None,
    })


if __name__ == "__main__":
    main()
