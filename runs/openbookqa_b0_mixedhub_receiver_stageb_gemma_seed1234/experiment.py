import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from translators import LlamaToQwen, QwenToGemma, ResidualAdapter

ROOT = Path(__file__).resolve().parent


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_tensor(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp); tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tensor_hash(module):
    state = hashlib.sha256()
    for name, value in module.state_dict().items():
        state.update(name.encode()); state.update(value.detach().cpu().contiguous().numpy().tobytes())
    return state.hexdigest()


def text_config(model_or_config):
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "text_config", config)


def text_backbone(model):
    return getattr(model.model, "language_model", model.model)


def load_gemma(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["gemma"], local_files_only=True, dtype=torch.float32,
        attn_implementation=cfg["attention_implementation"])
    del model.model.vision_tower
    del model.model.multi_modal_projector
    model = model.cuda().eval().requires_grad_(False)
    geometry = (text_config(model).num_hidden_layers, text_config(model).num_key_value_heads,
                text_config(model).head_dim)
    if geometry != (34, 4, 256):
        raise RuntimeError(f"Unexpected Gemma geometry: {geometry}")
    return model


@dataclass(frozen=True)
class Span:
    index: int
    start: int
    end: int

    @property
    def special(self):
        return self.start == self.end


def token_spans(tokenizer, text, expected_ids):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    ids = bos + list(encoded["input_ids"])
    if ids != list(expected_ids):
        raise RuntimeError("Gemma offset-tokenization mismatch")
    offsets = [(0, 0)] * len(bos) + [tuple(value) for value in encoded["offset_mapping"]]
    return [Span(index, int(start), int(end)) for index, (start, end) in enumerate(offsets)]


def distance(point, span):
    if span.start <= point < span.end:
        return 0
    return span.start - point if point < span.start else point - span.end + 1


def overlap(first, second):
    return max(0, min(first.end, second.end) - max(first.start, second.start))


def target_token(anchor, source, spans):
    candidates = [span for span in spans if span.index and not span.special]
    containing = [span for span in candidates if span.start <= anchor < span.end]
    pool = containing or candidates
    return min(pool, key=lambda span: (distance(anchor, span), -overlap(source, span),
                                       span.end - span.start, span.index))


def rotary_embeddings(model, tensor, positions, layer_index):
    backbone = text_backbone(model)
    positions = positions.to(tensor.device).unsqueeze(0)
    config = text_config(model)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None or not isinstance(getattr(backbone.rotary_emb, "rope_type", None), dict):
        return backbone.rotary_emb(tensor.unsqueeze(0), positions)
    return backbone.rotary_emb(tensor.unsqueeze(0), positions, layer_types[layer_index])


def rotate_half(tensor):
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(model, tensor, positions, layer_index):
    cos, sin = rotary_embeddings(model, tensor, positions, layer_index)
    cos, sin = cos[0, :, None], sin[0, :, None]
    return tensor * cos + rotate_half(tensor) * sin


def make_cache(model, key, value):
    positions = torch.arange(key.shape[1], device="cuda")
    dtype = next(model.parameters()).dtype
    key, value = key.to(dtype), value.to(dtype)
    rotated = torch.stack([apply_rope(model, layer, positions, index)
                           for index, layer in enumerate(key)])
    data = [(rotated[layer].permute(1, 0, 2).unsqueeze(0),
             value[layer].permute(1, 0, 2).unsqueeze(0)) for layer in range(key.shape[0])]
    return DynamicCache(ddp_cache_data=data, config=text_config(model))


def final_logits(model, ids, key, value):
    cache = make_cache(model, key, value)
    prefix = key.shape[1]
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    output = text_backbone(model)(
        input_ids=input_ids,
        attention_mask=torch.ones((1, prefix + len(ids)), device="cuda", dtype=torch.long),
        position_ids=torch.arange(prefix, prefix + len(ids), device="cuda")[None],
        past_key_values=cache, use_cache=False)
    return model.lm_head(output.last_hidden_state[:, -1])[0].float()


@torch.no_grad()
def capture_body(model, full_ids, body_length):
    backbone = text_backbone(model)
    captured_k, captured_v, handles = {}, {}, []

    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            if hasattr(module, "k_norm"):
                try:
                    key = module.k_norm(key.transpose(1, 2)).transpose(1, 2)
                except (RuntimeError, ValueError):
                    key = module.k_norm(key)
            captured_k[index] = key[0, :body_length].cpu()
            captured_v[index] = module.v_proj(hidden).view(shape)[0, :body_length].cpu()
        return apply

    for index, layer in enumerate(backbone.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)
        backbone(input_ids=ids, attention_mask=torch.ones_like(ids),
                 position_ids=torch.arange(len(full_ids), device="cuda")[None], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack([captured_k[i] for i in range(34)]), torch.stack([captured_v[i] for i in range(34)])


def manifests(cfg, split):
    upstream = read_json(Path(cfg["upstream_root"]) / "runs/study/manifests" / f"{split}.json")["rows"]
    downstream = read_json(Path(cfg["downstream_root"]) / "runs/study/manifests" / f"{split}.json")["rows"]
    count = cfg[f"{split}_samples"]
    upstream, downstream = upstream[:count], downstream[:count]
    if [row["id"] for row in upstream] != [row["id"] for row in downstream]:
        raise RuntimeError(f"Manifest ID/order mismatch: {split}")
    return list(zip(upstream, downstream))


def upstream_source(cfg, split, row):
    return torch.load(Path(cfg["upstream_root"]) / "runs/study/source_selected_cache" / split /
                      f"{row['id']}.pt", map_location="cpu", weights_only=True)


def upstream_pair(cfg, split, row):
    return torch.load(Path(cfg["upstream_root"]) / "runs/study/pair_cache" / split /
                      f"{row['id']}.pt", map_location="cpu", weights_only=True)


def downstream_pair(cfg, split, row):
    return torch.load(Path(cfg["downstream_root"]) / "runs/study/pair_cache" / split /
                      f"{row['id']}.pt", map_location="cpu", weights_only=True)


def teacher_path(split, sample_id):
    return ROOT / "runs/study/llama_anchor_teacher" / split / f"{sample_id}.pt"


def choice_kl(logits, teacher_choice, ids, temperature):
    index = torch.tensor(ids, device=logits.device)
    student = F.log_softmax(logits[index].float() / temperature, dim=-1)
    target = F.log_softmax(teacher_choice.to(logits.device).float() / temperature, dim=-1)
    return F.kl_div(student, target.detach(), reduction="sum", log_target=True) * temperature ** 2


def receiver_logits(model, row, question_pair, key, value):
    key = torch.cat((question_pair["question_k"].cuda(), key), dim=1)
    value = torch.cat((question_pair["question_v"].cuda(), value), dim=1)
    return final_logits(model, row["encoded"]["gemma"]["receiver_answer"], key, value)


def load_state(module, checkpoint, expected_kind=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if expected_kind is not None and payload.get("kind") != expected_kind:
        raise RuntimeError(f"Checkpoint kind mismatch: {payload.get('kind')}")
    module.load_state_dict(payload["state"], strict=True)
    return module.cuda().eval().requires_grad_(False)


@torch.no_grad()
def prepare_teachers(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["gemma"], local_files_only=True)
    model = load_gemma(cfg)
    signature = digest(cfg)
    try:
        for split in ("train", "validation", "test"):
            rows = manifests(cfg, split)
            for number, (up_row, down_row) in enumerate(rows, 1):
                path = teacher_path(split, up_row["id"])
                if path.exists():
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    if payload.get("signature") == signature:
                        continue
                pair = upstream_pair(cfg, split, up_row)
                fields = down_row["encoded"]["gemma"]
                key, value = capture_body(model, fields["full"], len(fields["body"]))
                spans = token_spans(tokenizer, down_row["body"], fields["body"])
                option_start, option_end = down_row["options_char_span"]
                options = [span for span in spans if min(span.end, option_end) > max(span.start, option_start)]
                indices = []
                for anchor in pair["metadata"]["anchors"]:
                    source = Span(anchor["target_index"], anchor["target_start"], anchor["target_end"])
                    indices.append(target_token(anchor["anchor_char"], source, options).index)
                question_length = fields["question_prefix_length"]
                logits = final_logits(
                    model, fields["receiver_answer"],
                    torch.cat((key[:, :question_length].cuda(), key[:, indices].cuda()), dim=1),
                    torch.cat((value[:, :question_length].cuda(), value[:, indices].cuda()), dim=1))
                choice = logits[fields["choice_ids"]].cpu()
                save_tensor(path, {"signature": signature, "id": up_row["id"],
                                  "choice_logits": choice, "prediction": int(choice.argmax()),
                                  "target_indices": indices})
                if number % 32 == 0 or number == len(rows):
                    log(f"Llama-anchor Gemma teachers {split}: {number}/{len(rows)}")
    finally:
        del model
        torch.cuda.empty_cache()


@torch.no_grad()
def hub_states(cfg, upstream_base, downstream_base, split, up_row):
    pair = upstream_pair(cfg, split, up_row)
    qwen_native = (pair["target_k"][:, 1:].cuda()[None], pair["target_v"][:, 1:].cuda()[None])
    source = upstream_source(cfg, split, up_row)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        llama_hub = upstream_base(source["source_k"][:, 1:].cuda()[None],
                                  source["source_v"][:, 1:].cuda()[None])
        qwen_native_gemma = downstream_base(*qwen_native)
        llama_hub_gemma = downstream_base(*llama_hub)
    return {"qwen_native": qwen_native_gemma, "llama_stage_a": llama_hub_gemma}


@torch.no_grad()
def functional_metrics(cfg, model, upstream_base, downstream_base, adapter, split):
    adapter.eval()
    totals = {source: {"accuracy": 0, "oracle_accuracy": 0, "agreement": 0, "kl": 0.0}
              for source in ("qwen_native", "llama_stage_a")}
    rows = manifests(cfg, split)
    for up_row, down_row in rows:
        teacher = torch.load(teacher_path(split, up_row["id"]), map_location="cpu", weights_only=True)
        question_pair = downstream_pair(cfg, split, down_row)
        states = hub_states(cfg, upstream_base, downstream_base, split, up_row)
        ids, gold, oracle = down_row["encoded"]["gemma"]["choice_ids"], down_row["gold_index"], teacher["prediction"]
        for source, (base_k, base_v) in states.items():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                key, value = adapter(base_k, base_v)
            logits = receiver_logits(model, down_row, question_pair, key[0], value[0])
            prediction = int(logits[ids].argmax())
            totals[source]["accuracy"] += prediction == gold
            totals[source]["oracle_accuracy"] += oracle == gold
            totals[source]["agreement"] += prediction == oracle
            totals[source]["kl"] += choice_kl(logits, teacher["choice_logits"], ids, cfg["temperature"]).item()
    metrics = {source: {"accuracy": values["accuracy"] / len(rows),
                        "oracle_accuracy": values["oracle_accuracy"] / len(rows),
                        "oracle_agreement": values["agreement"] / len(rows),
                        "choice_kl": values["kl"] / len(rows)} for source, values in totals.items()}
    metrics["balanced"] = {"mean_accuracy": sum(x["accuracy"] for x in metrics.values()) / 2,
                           "mean_choice_kl": sum(x["choice_kl"] for x in metrics.values()) / 2}
    return metrics


def save_checkpoint(cfg, adapter, step, validation):
    path = ROOT / "runs/study/stage_b_mixed/checkpoints" / f"step_{step}.pt"
    save_tensor(path, {"signature": digest(cfg), "kind": "mixed_receiver_residual",
                       "step": step, "validation": validation,
                       "state": {name: value.detach().cpu() for name, value in adapter.state_dict().items()}})
    return path


def train(cfg):
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    upstream_base = load_state(LlamaToQwen(cfg["mlp_hidden_dim"]), cfg["upstream_stage_a_checkpoint"])
    downstream_base = load_state(QwenToGemma(cfg["mlp_hidden_dim"], cfg["depth_output_dim"]),
                                 cfg["downstream_stage_a_checkpoint"])
    upstream_hash, downstream_hash = tensor_hash(upstream_base), tensor_hash(downstream_base)
    adapter = ResidualAdapter(34, 4, 256, cfg["adapter_rank"]).cuda().train()
    model = load_gemma(cfg)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=1000000)
    rows = manifests(cfg, "train")
    order = list(range(len(rows))); random.Random(cfg["seed"] + 200).shuffle(order)
    root = ROOT / "runs/study/stage_b_mixed"; root.mkdir(parents=True, exist_ok=True)
    stream = (root / "training_steps.jsonl").open("w", encoding="utf-8")
    candidates, clipped, started = [], 0, time.monotonic()
    try:
        for step, begin in enumerate(range(0, len(order), cfg["source_pairs_per_step"]), 1):
            batch = [rows[index] for index in order[begin:begin + cfg["source_pairs_per_step"]]]
            if len(batch) != cfg["source_pairs_per_step"]:
                raise RuntimeError("Incomplete paired source batch")
            optimizer.zero_grad(set_to_none=True)
            losses = {"qwen_native": [], "llama_stage_a": []}
            for up_row, down_row in batch:
                teacher = torch.load(teacher_path("train", up_row["id"]), map_location="cpu", weights_only=True)
                question_pair = downstream_pair(cfg, "train", down_row)
                states = hub_states(cfg, upstream_base, downstream_base, "train", up_row)
                for source, (base_k, base_v) in states.items():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        key, value = adapter(base_k, base_v)
                    logits = receiver_logits(model, down_row, question_pair, key[0], value[0])
                    loss = choice_kl(logits, teacher["choice_logits"],
                                     down_row["encoded"]["gemma"]["choice_ids"], cfg["temperature"])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite mixed Stage-B loss")
                    scaler.scale(loss / cfg["effective_batch_size"]).backward()
                    losses[source].append(loss.item())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg["clip"])
            if not torch.isfinite(norm):
                raise RuntimeError("Invalid mixed Stage-B gradient")
            clipped += int(norm.item() > cfg["clip"])
            scaler.step(optimizer); scaler.update()
            record = {"step": step,
                      "qwen_native_choice_kl": sum(losses["qwen_native"]) / len(losses["qwen_native"]),
                      "llama_stage_a_choice_kl": sum(losses["llama_stage_a"]) / len(losses["llama_stage_a"]),
                      "mean_choice_kl": sum(map(sum, losses.values())) / cfg["effective_batch_size"],
                      "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > cfg["clip"]}
            stream.write(json.dumps(record) + "\n"); stream.flush()
            if step % 16 == 0:
                log(f"Mixed Receiver Stage-B {step}/{cfg['stage_b_steps']} KL={record['mean_choice_kl']:.6f}")
            if step % cfg["validation_interval_steps"] == 0 or step == cfg["stage_b_steps"]:
                validation = functional_metrics(cfg, model, upstream_base, downstream_base, adapter, "validation")
                adapter.train()
                path = save_checkpoint(cfg, adapter, step, validation)
                candidates.append({"step": step, "path": str(path), "validation": validation})
                save_json(root / "candidates.json", candidates)
                log(f"Mixed validation step={step} native={validation['qwen_native']['accuracy']:.4f} "
                    f"llama={validation['llama_stage_a']['accuracy']:.4f} mean={validation['balanced']['mean_accuracy']:.4f}")
            if step >= cfg["stage_b_steps"]:
                break
    finally:
        stream.close()
    if tensor_hash(upstream_base) != upstream_hash or tensor_hash(downstream_base) != downstream_hash:
        raise RuntimeError("Frozen Stage-A writer changed")
    if any(parameter.grad is not None for parameter in upstream_base.parameters()) or any(
            parameter.grad is not None for parameter in downstream_base.parameters()):
        raise RuntimeError("Frozen Stage-A writer received gradients")
    best = max(candidates, key=lambda item: (item["validation"]["balanced"]["mean_accuracy"],
                                             -item["validation"]["balanced"]["mean_choice_kl"], -item["step"]))
    summary = {"stage": "B0 mixed-source Gemma receiver Residual64",
               "objective": "50/50 Qwen-native and Llama3B-StageA Hub; Gemma Oracle choice-KL",
               "train_samples": len(rows), "effective_exposures": len(rows) * 2,
               "source_pairs_per_step": cfg["source_pairs_per_step"],
               "effective_batch_size": cfg["effective_batch_size"], "steps": cfg["stage_b_steps"],
               "parameters_trained": sum(p.numel() for p in adapter.parameters()),
               "clip_rate": clipped / cfg["stage_b_steps"], "seconds": time.monotonic() - started,
               "upstream_base_hash": upstream_hash, "downstream_base_hash": downstream_hash,
               "best_balanced_accuracy": best, "candidates": candidates}
    save_json(root / "training_summary.json", summary)
    del model, adapter, upstream_base, downstream_base, optimizer, scaler
    torch.cuda.empty_cache()


def load_new_adapter(cfg):
    summary = read_json(ROOT / "runs/study/stage_b_mixed/training_summary.json")
    payload = torch.load(summary["best_balanced_accuracy"]["path"], map_location="cpu", weights_only=True)
    if payload.get("signature") != digest(cfg) or payload.get("kind") != "mixed_receiver_residual":
        raise RuntimeError("New mixed adapter checkpoint mismatch")
    adapter = ResidualAdapter(34, 4, 256, cfg["adapter_rank"]).cuda().eval()
    adapter.load_state_dict(payload["state"], strict=True)
    return adapter, summary


def correctness_pair(first, second):
    both = sum(a and b for a, b in zip(first, second))
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    total = len(first); discordant = only_first + only_second
    p = 1.0 if not discordant else min(
        1.0, 2 * sum(math.comb(discordant, k) for k in range(min(only_first, only_second) + 1)) /
        2 ** discordant)
    return {"both_correct": both, "only_first_correct": only_first,
            "only_second_correct": only_second, "both_wrong": total - both - only_first - only_second,
            "second_minus_first": (only_second - only_first) / total, "mcnemar_p": p}


@torch.no_grad()
def evaluate(cfg):
    upstream_base = load_state(LlamaToQwen(cfg["mlp_hidden_dim"]), cfg["upstream_stage_a_checkpoint"])
    downstream_base = load_state(QwenToGemma(cfg["mlp_hidden_dim"], cfg["depth_output_dim"]),
                                 cfg["downstream_stage_a_checkpoint"])
    old_adapter = load_state(ResidualAdapter(34, 4, 256, cfg["adapter_rank"]),
                             cfg["old_downstream_residual_checkpoint"], "residual")
    new_adapter, training = load_new_adapter(cfg)
    model = load_gemma(cfg)
    names = ("stage_a", "old_stage_b", "new_mixed_stage_b")
    sources = ("qwen_native", "llama_stage_a")
    totals = {source: {name: {"correct": 0, "agreement": 0, "kl": 0.0, "delta_k": [], "delta_v": []}
                       for name in names} for source in sources}
    correct = {source: {name: [] for name in names} for source in sources}
    records = []
    rows = manifests(cfg, "test")
    for number, (up_row, down_row) in enumerate(rows, 1):
        teacher = torch.load(teacher_path("test", up_row["id"]), map_location="cpu", weights_only=True)
        question_pair = downstream_pair(cfg, "test", down_row)
        states = hub_states(cfg, upstream_base, downstream_base, "test", up_row)
        ids, gold, oracle = down_row["encoded"]["gemma"]["choice_ids"], down_row["gold_index"], teacher["prediction"]
        record = {"id": up_row["id"], "gold_index": gold, "oracle_prediction": oracle, "sources": {}}
        for source, (base_k, base_v) in states.items():
            # Keep evaluation numerics identical to training/validation.  Hub
            # states are FP16 while adapter checkpoints are loaded as FP32;
            # autocast performs the Linear operations in the common FP16 dtype.
            with torch.amp.autocast("cuda", dtype=torch.float16):
                old_k, old_v = old_adapter(base_k, base_v)
                new_k, new_v = new_adapter(base_k, base_v)
            outputs = {"stage_a": (base_k[0], base_v[0]),
                       "old_stage_b": (old_k[0], old_v[0]),
                       "new_mixed_stage_b": (new_k[0], new_v[0])}
            record["sources"][source] = {}
            for name, (key, value) in outputs.items():
                logits = receiver_logits(model, down_row, question_pair, key, value)
                prediction = int(logits[ids].argmax()); is_correct = prediction == gold
                correct[source][name].append(is_correct)
                totals[source][name]["correct"] += is_correct
                totals[source][name]["agreement"] += prediction == oracle
                totals[source][name]["kl"] += choice_kl(
                    logits, teacher["choice_logits"], ids, cfg["temperature"]).item()
                totals[source][name]["delta_k"].append(
                    0.0 if name == "stage_a" else (key - base_k[0]).float().norm().item() /
                    base_k[0].float().norm().clamp_min(1e-12).item())
                totals[source][name]["delta_v"].append(
                    0.0 if name == "stage_a" else (value - base_v[0]).float().norm().item() /
                    base_v[0].float().norm().clamp_min(1e-12).item())
                record["sources"][source][name] = {"prediction": prediction, "correct": bool(is_correct),
                    "oracle_agreement": prediction == oracle, "choice_logits": logits[ids].cpu().tolist()}
        records.append(record)
        if number % 16 == 0 or number == len(rows):
            log(f"B0 final evaluation {number}/{len(rows)}")
    metrics = {source: {name: {"correct": values["correct"],
                                      "accuracy": values["correct"] / len(rows),
                                      "oracle_agreement": values["agreement"] / len(rows),
                                      "choice_kl": values["kl"] / len(rows),
                                      "mean_relative_delta_k": sum(values["delta_k"]) / len(rows),
                                      "mean_relative_delta_v": sum(values["delta_v"]) / len(rows)}
                               for name, values in conditions.items()}
               for source, conditions in totals.items()}
    pairwise = {source: {
        "stage_a_vs_old": correctness_pair(correct[source]["stage_a"], correct[source]["old_stage_b"]),
        "stage_a_vs_new": correctness_pair(correct[source]["stage_a"], correct[source]["new_mixed_stage_b"]),
        "old_vs_new": correctness_pair(correct[source]["old_stage_b"], correct[source]["new_mixed_stage_b"])}
        for source in sources}
    result = {"status": "completed", "protocol": cfg["protocol"], "sample_count": len(rows),
              "training": training, "metrics": metrics, "pairwise_correctness": pairwise,
              "success_diagnostics": {
                  "llama_new_minus_stage_a": (metrics["llama_stage_a"]["new_mixed_stage_b"]["accuracy"] -
                                                metrics["llama_stage_a"]["stage_a"]["accuracy"]),
                  "llama_new_minus_old": (metrics["llama_stage_a"]["new_mixed_stage_b"]["accuracy"] -
                                           metrics["llama_stage_a"]["old_stage_b"]["accuracy"]),
                  "qwen_new_minus_stage_a": (metrics["qwen_native"]["new_mixed_stage_b"]["accuracy"] -
                                               metrics["qwen_native"]["stage_a"]["accuracy"]),
                  "qwen_new_minus_old": (metrics["qwen_native"]["new_mixed_stage_b"]["accuracy"] -
                                          metrics["qwen_native"]["old_stage_b"]["accuracy"])}}
    output = ROOT / "runs/study/results"; output.mkdir(parents=True, exist_ok=True)
    save_json(output / "comparison.json", result)
    with (output / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result["success_diagnostics"], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("teachers", "train", "evaluate", "all"), default="all", nargs="?")
    args = parser.parse_args()
    cfg = read_json(ROOT / "config.json")
    if args.action in ("teachers", "all"):
        prepare_teachers(cfg)
    if args.action in ("train", "all"):
        train(cfg)
    if args.action in ("evaluate", "all"):
        evaluate(cfg)


if __name__ == "__main__":
    main()
