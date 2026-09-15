from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from common import (
    answer_f1, cache_tensors, device, dynamic_cache, load_json, load_model,
    load_tokenizer, nmse, normalize_answer, progress, save_json, selected_samples,
    target_ids, vector_metrics,
)


class NativeCapture:
    def __init__(self, model):
        self.pre, self.value, self.rotated, self.handles = {}, {}, {}, []
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(
                self._hook(index), with_kwargs=True,
            ))

    def _hook(self, index):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            batch, length, _ = hidden.shape
            shape = (batch, length, -1, module.head_dim)
            key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
            value = module.v_proj(hidden).view(shape).transpose(1, 2)
            embeddings = kwargs["position_embeddings"]
            _, rotated = apply_rotary_pos_emb(key, key, embeddings[0], embeddings[1])
            self.pre[index] = key[0].detach().clone()
            self.value[index] = value[0].detach().clone()
            self.rotated[index] = rotated[0].detach().clone()
        return hook

    def stacked(self, layers):
        if len(self.pre) != layers:
            raise RuntimeError(f"captured {len(self.pre)} layers, expected {layers}")
        return (
            [self.pre[i] for i in range(layers)],
            [self.value[i] for i in range(layers)],
            [self.rotated[i] for i in range(layers)],
        )

    def close(self):
        for handle in self.handles:
            handle.remove()


class TraceCapture:
    def __init__(self, model, token_index):
        self.token_index = token_index
        self.attention, self.hidden, self.handles = {}, {}, []
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.register_forward_hook(self._attn(index)))
            self.handles.append(layer.register_forward_hook(self._hidden(index)))

    def _select(self, output):
        value = output[0] if isinstance(output, tuple) else output
        return value[0, self.token_index].detach().float().cpu()

    def _attn(self, index):
        def hook(module, inputs, output): self.attention[index] = self._select(output)
        return hook

    def _hidden(self, index):
        def hook(module, inputs, output): self.hidden[index] = self._select(output)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def prefill(model, ids, cfg, capture=False):
    token_ids = torch.tensor([ids], dtype=torch.long, device=device())
    positions = torch.arange(len(ids), device=device()).unsqueeze(0)
    native = NativeCapture(model) if capture else None
    with torch.no_grad():
        output = model(
            input_ids=token_ids, attention_mask=torch.ones_like(token_ids),
            position_ids=positions, use_cache=True,
        )
    captured = native.stacked(cfg["num_layers"]) if native else None
    if native:
        native.close()
    return output.past_key_values, captured


def forward_logits(model, prompt, target, prefix=0, cache=None, trace_index=None):
    current = prompt + target[:-1]
    ids = torch.tensor([current], dtype=torch.long, device=device())
    positions = torch.arange(prefix, prefix + len(current), device=device()).unsqueeze(0)
    mask = torch.ones(1, prefix + len(current), dtype=torch.long, device=device())
    trace = TraceCapture(model, trace_index) if trace_index is not None else None
    with torch.no_grad():
        output = model(
            input_ids=ids, attention_mask=mask, position_ids=positions,
            past_key_values=cache, use_cache=False,
        )
    if trace:
        trace.close()
    start = len(prompt) - 1
    logits = output.logits[0, start:start + len(target)].float().cpu()
    traces = None if trace is None else {"attention": trace.attention, "hidden": trace.hidden}
    return logits, traces


def generate(model, tok, prompt, cfg, prefix=0, cache=None):
    ids = torch.tensor([prompt], dtype=torch.long, device=device())
    positions = torch.arange(prefix, prefix + len(prompt), device=device()).unsqueeze(0)
    mask = torch.ones(1, prefix + len(prompt), dtype=torch.long, device=device())
    with torch.no_grad():
        output = model(
            input_ids=ids, attention_mask=mask, position_ids=positions,
            past_key_values=cache, use_cache=True,
        )
    past = output.past_key_values
    next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    generated = []
    next_position = prefix + len(prompt)
    for _ in range(cfg["max_new_tokens"]):
        token = int(next_token.item())
        if token == tok.eos_token_id:
            break
        generated.append(token)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=device())], 1)
        with torch.no_grad():
            output = model(
                input_ids=next_token, attention_mask=mask,
                position_ids=torch.tensor([[next_position]], device=device()),
                past_key_values=past, use_cache=True,
            )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
        next_position += 1
    return tok.decode(generated, skip_special_tokens=True).strip(), generated


def nll(logits, gold):
    return F.cross_entropy(logits, torch.tensor(gold, dtype=torch.long)).item()


def distribution(reference, other, gold, tok):
    ref_logp, other_logp = reference.log_softmax(-1), other.log_softmax(-1)
    kl = (ref_logp.exp() * (ref_logp - other_logp)).sum(-1)
    ref_top5 = reference[0].topk(5).indices.tolist()
    other_top5 = other[0].topk(5).indices.tolist()
    return {
        "mean_kl": kl.mean().item(), "max_kl": kl.max().item(),
        "top1_match_rate": (reference.argmax(-1) == other.argmax(-1)).float().mean().item(),
        "first_top1_match": bool(reference[0].argmax() == other[0].argmax()),
        "first_reference_top5_ids": ref_top5, "first_other_top5_ids": other_top5,
        "first_reference_top5_tokens": tok.convert_ids_to_tokens(ref_top5),
        "first_other_top5_tokens": tok.convert_ids_to_tokens(other_top5),
        "top5_set_equal": set(ref_top5) == set(other_top5),
        "logits_max_absolute_error": (reference - other).abs().max().item(),
        "logits_cosine": F.cosine_similarity(reference.flatten(), other.flatten(), 0).item(),
        "reference_nll": nll(reference, gold), "other_nll": nll(other, gold),
        "nll_absolute_difference": abs(nll(reference, gold) - nll(other, gold)),
    }


def cache_rows(sample_id, official_k, official_v, post_k, manual_v, pre_post_k):
    rows = []
    for layer in range(len(official_k)):
        row = {"sample_id": sample_id, "layer": layer, "cache_tokens": official_k[layer].shape[-2]}
        for prefix, reference, other in (
            ("post_k", official_k[layer], post_k[layer]),
            ("post_v", official_v[layer], manual_v[layer]),
            ("pre_rope_k", official_k[layer], pre_post_k[layer]),
            ("pre_rope_v", official_v[layer], manual_v[layer]),
        ):
            for name, value in vector_metrics(reference, other).items():
                row[f"{prefix}_{name}"] = value
        rows.append(row)
    return rows


def trace_rows(sample_id, reference, conditions, layers):
    rows = []
    for name, trace in conditions.items():
        for layer in range(layers):
            row = {"sample_id": sample_id, "comparison": name, "layer": layer}
            for family in ("attention", "hidden"):
                metrics = vector_metrics(reference[family][layer], trace[family][layer])
                row[f"{family}_cosine"] = metrics["cosine"]
                row[f"{family}_relative_error"] = math.sqrt(metrics["nmse"])
                row[f"{family}_max_absolute_error"] = metrics["max_absolute_error"]
            rows.append(row)
    return rows


def closest_shuffle(samples):
    mapping = {}
    for sample in samples:
        choices = [x for x in samples if x["id"] != sample["id"] and normalize_answer(x["answer"]) != normalize_answer(sample["answer"])]
        mapping[sample["id"]] = min(choices, key=lambda x: abs(len(x["context_input_ids"]) - len(sample["context_input_ids"])))
    return mapping


def generation_row(sample, condition, prediction, ids, answer_nll):
    return {
        "sample_id": sample["id"], "type": sample["type"], "condition": condition,
        "answer": sample["answer"], "prediction": prediction,
        "em": float(normalize_answer(prediction) == normalize_answer(sample["answer"])),
        "f1": answer_f1(prediction, sample["answer"]), "nll": answer_nll,
        "output_tokens": len(ids),
        "has_extra_explanation": len(normalize_answer(prediction).split()) > len(normalize_answer(sample["answer"]).split()) + 5,
    }


def aggregate(records):
    generations = [x for record in records for x in record["generation"]]
    distributions = [x for record in records for x in record["distribution"]]
    caches = [x for record in records for x in record["cache"]]
    traces = [x for record in records for x in record["trace"]]
    conditions = {}
    for name in sorted({x["condition"] for x in generations}):
        rows = [x for x in generations if x["condition"] == name]
        conditions[name] = {
            "count": len(rows), "em": sum(x["em"] for x in rows) / len(rows),
            "f1": sum(x["f1"] for x in rows) / len(rows),
            "nll": sum(x["nll"] for x in rows) / len(rows),
            "bridge_f1": sum(x["f1"] for x in rows if x["type"] == "bridge") / max(sum(x["type"] == "bridge" for x in rows), 1),
            "comparison_f1": sum(x["f1"] for x in rows if x["type"] == "comparison") / max(sum(x["type"] == "comparison" for x in rows), 1),
        }
    text = {x["sample_id"]: x["prediction"] for x in generations if x["condition"] == "full_context_text"}
    for name in conditions:
        rows = [x for x in generations if x["condition"] == name]
        conditions[name]["generation_match_vs_text"] = sum(normalize_answer(x["prediction"]) == normalize_answer(text[x["sample_id"]]) for x in rows) / len(rows)
    comparisons = {}
    for name in sorted({x["comparison"] for x in distributions}):
        rows = [x for x in distributions if x["comparison"] == name]
        comparisons[name] = {key: sum(x[key] for x in rows) / len(rows) for key in ("mean_kl", "top1_match_rate", "first_top1_match", "logits_max_absolute_error", "nll_absolute_difference")}
    layerwise = []
    for layer in range(36):
        c = [x for x in caches if x["layer"] == layer]
        t = [x for x in traces if x["layer"] == layer]
        layerwise.append({
            "layer": layer,
            **{key: sum(x[key] for x in c) / len(c) for key in (
                "post_k_cosine", "post_k_nmse", "post_v_cosine", "post_v_nmse",
                "pre_rope_k_cosine", "pre_rope_k_nmse", "pre_rope_v_cosine", "pre_rope_v_nmse",
            )},
            "trace_comparisons": {name: {
                "hidden_cosine": sum(x["hidden_cosine"] for x in t if x["comparison"] == name) / max(sum(x["comparison"] == name for x in t), 1),
                "attention_output_cosine": sum(x["attention_cosine"] for x in t if x["comparison"] == name) / max(sum(x["comparison"] == name for x in t), 1),
            } for name in sorted({x["comparison"] for x in t})},
        })
    return conditions, comparisons, layerwise, generations, distributions, caches, traces


def gates(cfg, conditions, comparisons, layerwise):
    g = cfg["gates"]
    checks = {
        "A1_A2_top1": comparisons["full_text_vs_official"]["top1_match_rate"] >= g["official_top1_match"],
        "A1_A2_KL": comparisons["full_text_vs_official"]["mean_kl"] <= g["official_mean_kl_max"],
        "A1_A2_generation": conditions["official_native_cache"]["generation_match_vs_text"] >= g["generation_match_min"],
        "A1_A3post_top1": comparisons["full_text_vs_manual_post"]["top1_match_rate"] >= g["manual_post_top1_match"],
        "A1_A3post_KL": comparisons["full_text_vs_manual_post"]["mean_kl"] <= g["manual_post_mean_kl_max"],
        "A1_A3pre_top1": comparisons["full_text_vs_manual_pre"]["top1_match_rate"] >= g["manual_pre_top1_match"],
        "A1_A3pre_KL": comparisons["full_text_vs_manual_pre"]["mean_kl"] <= g["manual_pre_mean_kl_max"],
        "manual_post_cache": max(max(x["post_k_nmse"], x["post_v_nmse"]) for x in layerwise) <= g["cache_nmse_max"],
        "manual_pre_cache": max(max(x["pre_rope_k_nmse"], x["pre_rope_v_nmse"]) for x in layerwise) <= g["cache_nmse_max"],
        "cache_cosine": min(min(x["post_k_cosine"], x["post_v_cosine"], x["pre_rope_k_cosine"], x["pre_rope_v_cosine"]) for x in layerwise) >= g["cache_cosine_min"],
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": g}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    samples = selected_samples(cfg, args.mode)
    shuffled = closest_shuffle(samples)
    model, tok = load_model(cfg), load_tokenizer(cfg)
    root = Path(cfg["work_dir"]) / "artifacts" / args.mode
    records_dir = root / "records"; records_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples, 1):
        destination = records_dir / f"{sample['id']}.json"
        if destination.exists():
            progress(f"{args.mode}: resume skip {index}/{len(samples)}")
            continue
        target = target_ids(tok, sample["answer"], cfg["max_answer_tokens"])
        official_cache, captured = prefill(model, sample["context_input_ids"], cfg, True)
        pre_k, manual_v, pre_post_k = captured
        official_k, official_v = cache_tensors(official_cache)
        manual_post_k = [value.detach().clone() for value in official_k]
        cache_metrics = cache_rows(sample["id"], official_k, official_v, manual_post_k, manual_v, pre_post_k)

        logits_text, trace_text = forward_logits(
            model, sample["input_ids"], target, trace_index=sample["context_end_index"],
        )
        logits_official, trace_official = forward_logits(
            model, sample["question_input_ids"], target,
            prefix=len(sample["context_input_ids"]), cache=copy.deepcopy(official_cache), trace_index=0,
        )
        post_cache = dynamic_cache(model, manual_post_k, manual_v)
        logits_post, trace_post = forward_logits(
            model, sample["question_input_ids"], target,
            prefix=len(sample["context_input_ids"]), cache=post_cache, trace_index=0,
        )
        pre_cache = dynamic_cache(model, pre_post_k, manual_v)
        logits_pre, trace_pre = forward_logits(
            model, sample["question_input_ids"], target,
            prefix=len(sample["context_input_ids"]), cache=pre_cache, trace_index=0,
        )
        logits_q, _ = forward_logits(model, sample["question_only_input_ids"], target)

        donor = shuffled[sample["id"]]
        donor_cache, _ = prefill(model, donor["context_input_ids"], cfg, False)
        donor_prefix = len(donor["context_input_ids"])
        logits_shuffled, _ = forward_logits(
            model, sample["question_input_ids"], target,
            prefix=donor_prefix, cache=copy.deepcopy(donor_cache),
        )

        distributions = []
        for name, other in (
            ("full_text_vs_official", logits_official),
            ("full_text_vs_manual_post", logits_post),
            ("full_text_vs_manual_pre", logits_pre),
            ("full_text_vs_shuffled", logits_shuffled),
        ):
            distributions.append({"sample_id": sample["id"], "comparison": name, **distribution(logits_text, other, target, tok)})

        generated = []
        prediction, ids = generate(model, tok, sample["question_only_input_ids"], cfg)
        generated.append(generation_row(sample, "question_only", prediction, ids, nll(logits_q, target)))
        prediction, ids = generate(model, tok, sample["input_ids"], cfg)
        generated.append(generation_row(sample, "full_context_text", prediction, ids, nll(logits_text, target)))
        prediction, ids = generate(model, tok, sample["question_input_ids"], cfg, len(sample["context_input_ids"]), copy.deepcopy(official_cache))
        generated.append(generation_row(sample, "official_native_cache", prediction, ids, nll(logits_official, target)))
        prediction, ids = generate(model, tok, sample["question_input_ids"], cfg, len(sample["context_input_ids"]), dynamic_cache(model, manual_post_k, manual_v))
        generated.append(generation_row(sample, "manual_post_rope_cache", prediction, ids, nll(logits_post, target)))
        prediction, ids = generate(model, tok, sample["question_input_ids"], cfg, len(sample["context_input_ids"]), dynamic_cache(model, pre_post_k, manual_v))
        generated.append(generation_row(sample, "manual_pre_rope_cache", prediction, ids, nll(logits_pre, target)))
        prediction, ids = generate(model, tok, sample["question_input_ids"], cfg, donor_prefix, copy.deepcopy(donor_cache))
        generated.append(generation_row(sample, "shuffled_native_cache", prediction, ids, nll(logits_shuffled, target)))

        record = {
            "sample_id": sample["id"], "type": sample["type"],
            "context_tokens": len(sample["context_input_ids"]),
            "question_suffix_tokens": len(sample["question_input_ids"]),
            "shuffle_id": donor["id"], "shuffle_context_tokens": donor_prefix,
            "cache": cache_metrics,
            "trace": trace_rows(sample["id"], trace_text, {
                "text_vs_official": trace_official,
                "text_vs_manual_post": trace_post,
                "text_vs_manual_pre": trace_pre,
            }, cfg["num_layers"]),
            "distribution": distributions, "generation": generated,
        }
        save_json(destination, record)
        save_json(root / "progress.json", {"completed": index, "total": len(samples)})
        progress(f"{args.mode}: full-context cache audit {index}/{len(samples)}")
        del official_cache, donor_cache
        torch.cuda.empty_cache()

    records = [load_json(records_dir / f"{sample['id']}.json") for sample in samples]
    conditions, comparisons, layerwise, generation_rows, distribution_rows, cache_all, trace_all = aggregate(records)
    save_json(root / "official_vs_text.json", comparisons["full_text_vs_official"])
    save_json(root / "manual_post_vs_official.json", {
        "distribution": comparisons["full_text_vs_manual_post"],
        "layerwise": [{"layer": x["layer"], **{k: x[k] for k in ("post_k_cosine", "post_k_nmse", "post_v_cosine", "post_v_nmse")}} for x in layerwise],
    })
    save_json(root / "manual_pre_vs_post.json", {
        "distribution": comparisons["full_text_vs_manual_pre"],
        "layerwise": [{"layer": x["layer"], **{k: x[k] for k in ("pre_rope_k_cosine", "pre_rope_k_nmse", "pre_rope_v_cosine", "pre_rope_v_nmse")}} for x in layerwise],
    })
    save_json(root / "layerwise_hidden_metrics.json", layerwise)
    save_json(root / "answer_logit_metrics.json", distribution_rows)
    save_json(root / "cache_tensor_metrics.json", cache_all)
    save_json(root / "execution_trace_metrics.json", trace_all)
    with (root / "generation_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in generation_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    gate = gates(cfg, conditions, comparisons, layerwise)
    summary = {
        "experiment": cfg["experiment_name"], "samples": len(samples),
        "training_performed": False, "writer_present": False, "reader_present": False,
        "lora_enabled": False, "conditions": conditions,
        "logit_comparisons": comparisons, "gate": gate,
    }
    save_json(root / "summary.json", summary)
    save_json(root / "completion.json", {"completed": True, "passed": gate["passed"], "samples": len(samples)})
    progress(f"{args.mode}: audit completed; gate_passed={gate['passed']}")
    if args.mode == "development" and cfg["enforce_hard_gates"] and not gate["passed"]:
        raise RuntimeError(f"full-cache equivalence hard gate failed: {[k for k,v in gate['checks'].items() if not v]}")


if __name__ == "__main__":
    main()
