from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not start P0-B")
    return torch.device("cuda")


def empty_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def p0a_module(cfg):
    return import_file("source_p0a", Path(cfg["source_p0a_dir"]) / "p0a.py")


def p0a2_module(cfg):
    return import_file("source_p0a2", Path(cfg["source_p0a2_dir"]) / "p0a2.py")


def source_config(cfg):
    return load_json(Path(cfg["source_p0a_dir"]) / "config.json")


def validation_samples(cfg, limit=None):
    rows = load_json(Path(cfg["source_p0a_dir"]) / "manifests" / "dataset.json")["validation"]
    return rows[:limit] if limit else rows


def load_native(cfg, role, sample_id):
    return torch.load(
        Path(cfg["source_p0a_dir"]) / "cache" / role / "validation" / f"{sample_id}.pt",
        map_location="cpu",
        weights_only=True,
    )


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


class NativeSelectorHook:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.selected = set(cfg["selected_layers"])
        self.sample = None
        self.layer_scores = {}
        self.handles = []
        for index, block in enumerate(model.model.layers):
            if index in self.selected:
                self.handles.append(
                    block.self_attn.register_forward_pre_hook(self._make(index), with_kwargs=True)
                )

    def set_sample(self, sample):
        self.sample = sample
        self.layer_scores.clear()

    def _make(self, layer):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            hidden = hidden.detach()
            length = hidden.shape[1]
            qh, kh, d = self.cfg["num_query_heads"], self.cfg["num_kv_heads"], self.cfg["head_dim"]
            q = module.q_norm(module.q_proj(hidden).view(1, length, qh, d))[0]
            k = module.k_norm(module.k_proj(hidden).view(1, length, kh, d))[0]
            q = apply_rope(q, cos[0], sin[0])
            k = apply_rope(k, cos[0], sin[0])
            sample = self.sample
            q = q[sample["question_start"] : sample["question_end"]]
            k = k[: sample["context_length"]].repeat_interleave(qh // kh, dim=1)
            logits = torch.einsum("qhd,thd->qht", q, k) / math.sqrt(d)
            attention = logits.softmax(dim=-1)
            scores = []
            for span in sample["sentence_spans"]:
                if span["end"] <= span["start"]:
                    scores.append(attention.new_tensor(0.0))
                else:
                    scores.append(attention[:, :, span["start"] : span["end"]].sum(-1).mean())
            self.layer_scores[layer] = torch.stack(scores).detach().float().cpu()
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def extract_selector_scores(cfg, role, limit=None):
    device = require_cuda()
    model_path = cfg["model_a"] if role == "a" else cfg["model_b"]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hook = NativeSelectorHook(model, cfg)
    samples = validation_samples(cfg, limit)
    root = Path(cfg["work_dir"]) / "cache" / "selector_scores" / role
    progress(f"Native selector extraction {role}: {len(samples)} samples")
    for index, sample in enumerate(samples, 1):
        out = root / f"{sample['id']}.pt"
        if out.exists():
            continue
        hook.set_sample(sample)
        ids = torch.tensor([sample["full_ids"]], dtype=torch.long, device=device)
        positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=positions,
            use_cache=False,
        )
        if len(hook.layer_scores) != len(cfg["selected_layers"]):
            raise RuntimeError(f"Missing selector layers for {sample['id']}")
        per_layer = torch.stack([hook.layer_scores[layer] for layer in cfg["selected_layers"]])
        scores = per_layer.mean(0)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"id": sample["id"], "scores": scores, "per_layer_scores": per_layer}, out)
        if index % 4 == 0 or index == len(samples):
            progress(f"Native selector extraction {role}: {index}/{len(samples)}")
    hook.close()
    del model
    empty_cuda()


def sentence_gold(sample):
    return [bool(span["gold"]) for span in sample["sentence_spans"]]


def top_indices(scores, budget):
    count = min(int(budget), scores.numel())
    return scores.topk(count).indices.tolist()


def deterministic_random_indices(sample, count, salt, allowed=None):
    candidates = list(range(len(sample["sentence_spans"]))) if allowed is None else list(allowed)
    rng = random.Random(f"{sample['id']}:{salt}")
    rng.shuffle(candidates)
    return candidates[: min(count, len(candidates))]


def condition_indices(cfg, sample, scores, condition, role):
    gold = [i for i, value in enumerate(sentence_gold(sample)) if value]
    non_gold = [i for i in range(len(sample["sentence_spans"])) if i not in set(gold)]
    if condition == "gold_only":
        return gold
    if condition == "gold_plus_distractors":
        extra = deterministic_random_indices(
            sample,
            max(0, cfg["main_budget"] - len(gold)),
            f"{role}:gold_distractors",
            non_gold,
        )
        return gold + extra
    if condition == "random_4":
        return deterministic_random_indices(sample, cfg["main_budget"], f"{role}:random")
    if condition.startswith("auto_top_"):
        return top_indices(scores, int(condition.rsplit("_", 1)[1]))
    raise ValueError(condition)


def selected_token_indices(sample, sentence_indices):
    result = []
    for sentence_index in sentence_indices:
        span = sample["sentence_spans"][sentence_index]
        result.extend(range(span["start"], span["end"]))
    if not result:
        raise RuntimeError(f"Selection is empty for {sample['id']}")
    return result


def selector_metrics(cfg, limit=None):
    samples = validation_samples(cfg, limit)
    table = {}
    for budget in cfg["budgets"]:
        recalls_a, recalls_b, overlaps = [], [], []
        for sample in samples:
            a = torch.load(
                Path(cfg["work_dir"]) / "cache" / "selector_scores" / "a" / f"{sample['id']}.pt",
                map_location="cpu",
                weights_only=True,
            )["scores"]
            b = torch.load(
                Path(cfg["work_dir"]) / "cache" / "selector_scores" / "b" / f"{sample['id']}.pt",
                map_location="cpu",
                weights_only=True,
            )["scores"]
            ia, ib = set(top_indices(a, budget)), set(top_indices(b, budget))
            gold = {i for i, value in enumerate(sentence_gold(sample)) if value}
            recalls_a.append(len(ia & gold) / max(1, len(gold)))
            recalls_b.append(len(ib & gold) / max(1, len(gold)))
            overlaps.append(len(ia & ib) / max(1, len(ia | ib)))
        table[f"top_{budget}"] = {
            "sender_a_support_recall": sum(recalls_a) / len(recalls_a),
            "sender_b_support_recall": sum(recalls_b) / len(recalls_b),
            "sender_jaccard_overlap": sum(overlaps) / len(overlaps),
        }
    random_a, random_b, overlaps = [], [], []
    for sample in samples:
        ia = set(condition_indices(cfg, sample, None, "random_4", "a"))
        ib = set(condition_indices(cfg, sample, None, "random_4", "b"))
        gold = {i for i, value in enumerate(sentence_gold(sample)) if value}
        random_a.append(len(ia & gold) / max(1, len(gold)))
        random_b.append(len(ib & gold) / max(1, len(gold)))
        overlaps.append(len(ia & ib) / max(1, len(ia | ib)))
    table["random_4"] = {
        "sender_a_support_recall": sum(random_a) / len(random_a),
        "sender_b_support_recall": sum(random_b) / len(random_b),
        "sender_jaccard_overlap": sum(overlaps) / len(overlaps),
    }
    return table


class FrozenArtifacts:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = require_cuda()
        self.p0a = p0a_module(cfg)
        source_cfg = source_config(cfg)
        self.system = self.p0a.CanonicalSystem(source_cfg).to(self.device)
        state = torch.load(
            Path(cfg["source_p0a_dir"]) / "checkpoints" / "shared_formal.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.system.load_state_dict(state["model"])
        self.system.eval()
        for parameter in self.system.parameters():
            parameter.requires_grad_(False)
        p0a2 = p0a2_module(cfg)
        self.probe_a, self.probe_b = self._load_probe(p0a2, "a"), self._load_probe(p0a2, "b")

    def _load_probe(self, p0a2, sender):
        state = torch.load(
            Path(self.cfg["source_p0a2_dir"])
            / "checkpoints"
            / f"shared_canonical_{sender}_formal.pt",
            map_location="cpu",
            weights_only=True,
        )
        probe = p0a2.LinearContentProbe(state["text_dim"]).to(self.device)
        probe.load_state_dict(state["model"])
        probe.eval()
        for parameter in probe.parameters():
            parameter.requires_grad_(False)
        return probe


def to_device(item, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in item.items()}


def canonical_parts(artifacts, cfg, role, sample, sentence_indices):
    native = to_device(load_native(cfg, role, sample["id"]), artifacts.device)
    indices = selected_token_indices(sample, sentence_indices)
    selected = {
        "k": native["k"][:, indices],
        "v": native["v"][:, indices],
        "q": native["q"],
    }
    k, v, q = artifacts.system.transform(role, selected)
    return {"k": k, "v": v, "q": q, "token_indices": indices}


def scores_for_memory(q, memory):
    return torch.einsum("luhd,lthd->t", q, memory["k"]) / (
        math.sqrt(q.shape[-1]) * q.shape[0] * q.shape[1] * q.shape[2]
    )


def canonical_readout(q, memory):
    logits = torch.einsum("luhd,lthd->luht", q, memory["k"]) / math.sqrt(q.shape[-1])
    attention = logits.softmax(-1)
    value = torch.einsum("luht,lthd->luhd", attention, memory["v"])
    return F.normalize(value.float().mean((0, 1, 2)), dim=-1)


def average_precision(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    hits, total = 0, 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            hits += 1
            total += hits / rank
    return total / positives


def metrics_from_sentence_scores(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    gold = max(1, sum(labels))
    result = {}
    for k in (1, 5):
        hits = sum(labels[i] for i in order[:k])
        result[f"support_recall_at_{k}"] = hits / gold
        result[f"support_hit_at_{k}"] = float(hits > 0)
    first = next((rank for rank, index in enumerate(order, 1) if labels[index]), None)
    result["mrr"] = 1.0 / first if first else 0.0
    result["support_auprc"] = average_precision(scores, labels)
    return result


def k_metrics(token_scores, token_indices, sample):
    mapping = {token: token_scores[i].detach().float().item() for i, token in enumerate(token_indices)}
    sentence_scores = []
    for span in sample["sentence_spans"]:
        values = [mapping[token] for token in range(span["start"], span["end"]) if token in mapping]
        sentence_scores.append(sum(values) / len(values) if values else -1e9)
    return metrics_from_sentence_scores(sentence_scores, sentence_gold(sample))


def v_metrics(probe, readout, sentence_embeddings):
    candidates = F.normalize(sentence_embeddings["embeddings"].float().to(readout.device), dim=-1)
    scores = probe(readout) @ candidates.T
    return metrics_from_sentence_scores(
        scores.detach().float().cpu().tolist(),
        sentence_embeddings["labels"].bool().tolist(),
    )


def aggregate(rows):
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def average_directions(left, right):
    return {key: 0.5 * (left[key] + right[key]) for key in left}


@torch.no_grad()
def evaluate_conditions(cfg, limit=None):
    artifacts = FrozenArtifacts(cfg)
    samples = validation_samples(cfg, limit)
    score_cache = {}
    for role in ("a", "b"):
        for sample in samples:
            score_cache[(role, sample["id"])] = torch.load(
                Path(cfg["work_dir"]) / "cache" / "selector_scores" / role / f"{sample['id']}.pt",
                map_location="cpu",
                weights_only=True,
            )["scores"]
    conditions = ["gold_only", "gold_plus_distractors", "auto_top_8", "auto_top_4", "auto_top_2", "random_4"]
    report = {}
    for condition in conditions:
        rows = {name: [] for name in ("k_aa", "k_ab", "k_ba", "k_bb", "v_aa", "v_ab", "v_ba", "v_bb", "v_ab_shuffled", "v_ba_shuffled")}
        memories = {}
        for sample in samples:
            ia = condition_indices(cfg, sample, score_cache[("a", sample["id"])], condition, "a")
            ib = condition_indices(cfg, sample, score_cache[("b", sample["id"])], condition, "b")
            memories[("a", sample["id"])] = canonical_parts(artifacts, cfg, "a", sample, ia)
            memories[("b", sample["id"])] = canonical_parts(artifacts, cfg, "b", sample, ib)
        for index, sample in enumerate(samples):
            ma = memories[("a", sample["id"])]
            mb = memories[("b", sample["id"])]
            qa, qb = ma["q"], mb["q"]
            rows["k_aa"].append(k_metrics(scores_for_memory(qa, ma), ma["token_indices"], sample))
            rows["k_ab"].append(k_metrics(scores_for_memory(qa, mb), mb["token_indices"], sample))
            rows["k_ba"].append(k_metrics(scores_for_memory(qb, ma), ma["token_indices"], sample))
            rows["k_bb"].append(k_metrics(scores_for_memory(qb, mb), mb["token_indices"], sample))
            sentence_embeddings = torch.load(
                Path(cfg["source_p0a2_dir"]) / "cache" / "sentence_embeddings" / "validation" / f"{sample['id']}.pt",
                map_location="cpu",
                weights_only=True,
            )
            rows["v_aa"].append(v_metrics(artifacts.probe_a, canonical_readout(qa, ma), sentence_embeddings))
            rows["v_ab"].append(v_metrics(artifacts.probe_a, canonical_readout(qa, mb), sentence_embeddings))
            rows["v_ba"].append(v_metrics(artifacts.probe_b, canonical_readout(qb, ma), sentence_embeddings))
            rows["v_bb"].append(v_metrics(artifacts.probe_b, canonical_readout(qb, mb), sentence_embeddings))
            wrong = samples[(index + 1) % len(samples)]
            wrong_a = memories[("a", wrong["id"])]
            wrong_b = memories[("b", wrong["id"])]
            rows["v_ab_shuffled"].append(
                v_metrics(artifacts.probe_a, canonical_readout(qa, wrong_b), sentence_embeddings)
            )
            rows["v_ba_shuffled"].append(
                v_metrics(artifacts.probe_b, canonical_readout(qb, wrong_a), sentence_embeddings)
            )
        directions = {name: aggregate(values) for name, values in rows.items()}
        report[condition] = {
            "directions": directions,
            "summary": {
                "k_self": average_directions(directions["k_aa"], directions["k_bb"]),
                "k_cross": average_directions(directions["k_ab"], directions["k_ba"]),
                "v_self": average_directions(directions["v_aa"], directions["v_bb"]),
                "v_cross": average_directions(directions["v_ab"], directions["v_ba"]),
                "v_shuffled": average_directions(directions["v_ab_shuffled"], directions["v_ba_shuffled"]),
            },
        }
        progress(f"Canonical condition completed: {condition}")
    del artifacts
    empty_cuda()
    return report


def cpu_self_test(cfg):
    q = torch.randn(2, 3, 2, 8)
    k = torch.randn(2, 7, 2, 8)
    v = torch.randn(2, 7, 2, 8)
    memory = {"k": k, "v": v}
    assert scores_for_memory(q, memory).shape == (7,)
    assert canonical_readout(q, memory).shape == (8,)
    sample = {"id": "x", "sentence_spans": [{"start": 0, "end": 2}, {"start": 2, "end": 7}]}
    assert len(selected_token_indices(sample, [0])) == 2
    progress("P0-B CPU structural self-test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["cpu_self_test", "extract_a", "extract_b", "evaluate"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.command == "cpu_self_test":
        cpu_self_test(cfg)
    elif args.command == "extract_a":
        extract_selector_scores(cfg, "a", args.limit)
    elif args.command == "extract_b":
        extract_selector_scores(cfg, "b", args.limit)
    elif args.command == "evaluate":
        result = {
            "selector": selector_metrics(cfg, args.limit),
            "canonical": evaluate_conditions(cfg, args.limit),
        }
        suffix = "smoke" if args.limit else "formal"
        save_json(Path(cfg["work_dir"]) / "metrics" / f"{suffix}_evaluation.json", result)


if __name__ == "__main__":
    main()
