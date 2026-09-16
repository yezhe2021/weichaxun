import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from translators import GemmaToQwen, QwenToLlama, ResidualAdapter

ROOT = Path(__file__).resolve().parent


def log(message):
    print(message, flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise RuntimeError("Offset tokenization mismatch")
    offsets = [(0, 0)] * len(bos) + [tuple(value) for value in encoded["offset_mapping"]]
    return [Span(index, int(start), int(end)) for index, (start, end) in enumerate(offsets)]


def distance(point, span):
    if span.start <= point < span.end:
        return 0
    return span.start - point if point < span.start else point - span.end + 1


def overlap(first, second):
    return max(0, min(first.end, second.end) - max(first.start, second.start))


def target_token(anchor_char, source_span, target_spans):
    candidates = [span for span in target_spans if span.index and not span.special]
    containing = [span for span in candidates if span.start <= anchor_char < span.end]
    pool = containing or candidates
    return min(pool, key=lambda span: (distance(anchor_char, span), -overlap(source_span, span),
                                       span.end - span.start, span.index))


def encode_llama(tokenizer, row):
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    body_encoding = tokenizer(row["body"], add_special_tokens=False, return_offsets_mapping=True)
    body_plain = list(body_encoding["input_ids"])
    full_plain = enc(row["full_prompt"])
    question_plain = enc(row["question_prefix"])
    if body_plain + enc(row["answer_prefix"]) != full_plain:
        raise RuntimeError("Llama body/Answer boundary is not compositional")
    if full_plain[:len(question_plain)] != question_plain:
        raise RuntimeError("Llama Question boundary is not compositional")
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    choice_ids = []
    for label in row["labels"]:
        continuation = enc(row["full_prompt"] + " " + label)
        if continuation[:len(full_plain)] != full_plain or len(continuation) != len(full_plain) + 1:
            raise RuntimeError(f"Unstable Llama choice token: {label}")
        choice_ids.append(continuation[-1])
    return {
        "body": bos + body_plain,
        "full": bos + full_plain,
        "question_prefix_length": len(bos) + len(question_plain),
        "receiver_answer": enc(row["receiver_answer"]),
        "choice_ids": choice_ids,
    }


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(model, tensor, positions):
    positions = positions.to(tensor.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(tensor.unsqueeze(0), positions)
    cos, sin = cos[0, :, None], sin[0, :, None]
    return tensor * cos + rotate_half(tensor) * sin


def make_cache(model, key, value, positions):
    dtype = next(model.parameters()).dtype
    key, value = key.to(dtype), value.to(dtype)
    rotated = torch.stack([apply_rope(model, layer, positions) for layer in key])
    data = [(rotated[layer].permute(1, 0, 2).unsqueeze(0),
             value[layer].permute(1, 0, 2).unsqueeze(0)) for layer in range(key.shape[0])]
    return DynamicCache(ddp_cache_data=data, config=model.config)


@torch.no_grad()
def final_logits(model, ids, key, value):
    positions = torch.arange(key.shape[1], device="cuda")
    cache = make_cache(model, key, value, positions)
    prefix = key.shape[1]
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    output = model.model(input_ids=input_ids,
                         attention_mask=torch.ones((1, prefix + len(ids)), device="cuda", dtype=torch.long),
                         position_ids=torch.arange(prefix, prefix + len(ids), device="cuda")[None],
                         past_key_values=cache, use_cache=False)
    return model.lm_head(output.last_hidden_state[:, -1])[0].float()


@torch.no_grad()
def capture_llama(model, full_ids, body_length):
    captured_k, captured_v, handles = {}, {}, []

    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            value = module.v_proj(hidden).view(shape)
            captured_k[index] = key[0, :body_length].cpu()
            captured_v[index] = value[0, :body_length].cpu()
        return apply

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)
        output = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             position_ids=torch.arange(len(full_ids), device="cuda")[None], use_cache=False)
        full_logits = model.lm_head(output.last_hidden_state[:, -1])[0].float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return (torch.stack([captured_k[index] for index in range(28)]),
            torch.stack([captured_v[index] for index in range(28)]), full_logits)


def load_state(module, checkpoint, expected_kind=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if expected_kind is not None and payload.get("kind") != expected_kind:
        raise RuntimeError(f"Expected checkpoint kind={expected_kind}, got {payload.get('kind')}")
    module.load_state_dict(payload["state"], strict=True)
    return module.cuda().eval().requires_grad_(False), payload


def representation(prediction, target):
    prediction, target = prediction.float(), target.float()
    return {
        "nmse": ((prediction - target).square().mean() /
                 target.square().mean().clamp_min(1e-8)).item(),
        "cosine": F.cosine_similarity(prediction, target, dim=-1).mean().item(),
    }


def choice_kl(student, teacher, temperature):
    student = F.log_softmax(student.float() / temperature, dim=-1)
    target = F.log_softmax(teacher.float() / temperature, dim=-1)
    return (F.kl_div(student, target, reduction="sum", log_target=True) * temperature ** 2).item()


def paired(first, second):
    both = sum(a and b for a, b in zip(first, second))
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    return {"both_correct": both, "only_first_correct": only_first,
            "only_second_correct": only_second,
            "both_wrong": len(first) - both - only_first - only_second,
            "second_minus_first": (only_second - only_first) / len(first)}


def aggregate(records, reverse_branch, condition):
    rows = [record["reverse_branches"][reverse_branch][condition] for record in records]
    native = [record["llama_native_oracle32"] for record in records]
    return {
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "agreement_with_llama_native_oracle32": sum(
            row["prediction"] == oracle["prediction"] for row, oracle in zip(rows, native)) / len(rows),
        "choice_kl_to_llama_native_oracle32": sum(row["choice_kl_to_oracle"] for row in rows) / len(rows),
        "k_nmse_to_llama_native": sum(row["k_representation"]["nmse"] for row in rows) / len(rows),
        "v_nmse_to_llama_native": sum(row["v_representation"]["nmse"] for row in rows) / len(rows),
        "k_cosine_to_llama_native": sum(row["k_representation"]["cosine"] for row in rows) / len(rows),
        "v_cosine_to_llama_native": sum(row["v_representation"]["cosine"] for row in rows) / len(rows),
    }


@torch.no_grad()
def run(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    forward_root, reverse_root = Path(cfg["forward_root"]), Path(cfg["reverse_root"])
    manifest = read_json(forward_root / "runs/study/manifests/test.json")
    rows = manifest["rows"][:cfg["test_samples"]]
    if len(rows) != cfg["test_samples"]:
        raise RuntimeError("Wrong test sample count")

    forward_base, forward_base_payload = load_state(
        GemmaToQwen(cfg["forward_depth_hidden_dim"], cfg["forward_depth_output_dim"]),
        cfg["forward_stage_a_checkpoint"])
    forward_adapter, forward_residual_payload = load_state(
        ResidualAdapter(36, cfg["adapter_rank"]), cfg["forward_residual_checkpoint"], "residual")
    reverse_base, reverse_base_payload = load_state(
        QwenToLlama(cfg["reverse_hidden_dim"]), cfg["reverse_stage_a_checkpoint"])
    reverse_adapter, reverse_residual_payload = load_state(
        ResidualAdapter(28, cfg["adapter_rank"]), cfg["reverse_residual_checkpoint"], "residual")

    llama_tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["llama"], local_files_only=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["qwen"], local_files_only=True)
    llama = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["llama"], local_files_only=True, dtype=torch.float16,
        attn_implementation=cfg["attention_implementation"]).cuda().eval().requires_grad_(False)
    geometry = (llama.config.num_hidden_layers, llama.config.num_key_value_heads, llama.config.head_dim)
    if geometry != (28, 8, 128):
        raise RuntimeError(f"Unexpected Llama geometry: {geometry}")

    records = []
    for number, row in enumerate(rows, 1):
        sample_id = row["id"]
        source = torch.load(forward_root / "runs/study/source_selected_cache/test" / f"{sample_id}.pt",
                            map_location="cpu", weights_only=True)
        pair = torch.load(forward_root / "runs/study/pair_cache/test" / f"{sample_id}.pt",
                          map_location="cpu", weights_only=True)
        source_k = source["source_k"][:, 1:].cuda()[None]
        source_v = source["source_v"][:, 1:].cuda()[None]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            qwen_stage_a_k, qwen_stage_a_v = forward_base(source_k, source_v)
            qwen_stage_b_k, qwen_stage_b_v = forward_adapter(qwen_stage_a_k, qwen_stage_a_v)
        qwen_inputs = {
            "qwen_native_bridge": (pair["target_k"][:, 1:].cuda()[None],
                                   pair["target_v"][:, 1:].cuda()[None]),
            "gemma_stage_a": (qwen_stage_a_k, qwen_stage_a_v),
            "gemma_stage_b_residual": (qwen_stage_b_k, qwen_stage_b_v),
        }

        llama_encoded = encode_llama(llama_tokenizer, row)
        llama_k, llama_v, llama_full_logits = capture_llama(
            llama, llama_encoded["full"], len(llama_encoded["body"]))
        llama_spans = token_spans(llama_tokenizer, row["body"], llama_encoded["body"])
        qwen_spans = {span.index: span for span in token_spans(
            qwen_tokenizer, row["body"], row["encoded"]["qwen"]["body"])}
        option_start, option_end = row["options_char_span"]
        llama_options = [span for span in llama_spans if min(span.end, option_end) > max(span.start, option_start)]
        target_indices = []
        for anchor in pair["metadata"]["anchors"]:
            source_span = qwen_spans[anchor["target_index"]]
            target_indices.append(target_token(anchor["anchor_char"], source_span, llama_options).index)
        if len(target_indices) != 32:
            raise RuntimeError("Expected exactly 32 Llama target anchors")
        llama_native_k, llama_native_v = llama_k[:, target_indices].cuda(), llama_v[:, target_indices].cuda()
        question_k = llama_k[:, :llama_encoded["question_prefix_length"]].cuda()
        question_v = llama_v[:, :llama_encoded["question_prefix_length"]].cuda()
        choice_index = torch.tensor(llama_encoded["choice_ids"], device="cuda")
        native_key = torch.cat((question_k, llama_native_k), dim=1)
        native_value = torch.cat((question_v, llama_native_v), dim=1)
        native_logits = final_logits(llama, llama_encoded["receiver_answer"], native_key, native_value)[choice_index]
        native_prediction = int(native_logits.argmax())

        reverse_outputs = {"reverse_stage_a": {}, "reverse_residual": {}}
        for condition, (qwen_k, qwen_v) in qwen_inputs.items():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                llama_stage_a_k, llama_stage_a_v = reverse_base(qwen_k, qwen_v)
                llama_residual_k, llama_residual_v = reverse_adapter(llama_stage_a_k, llama_stage_a_v)
            for branch, (mapped_k, mapped_v) in {
                "reverse_stage_a": (llama_stage_a_k[0], llama_stage_a_v[0]),
                "reverse_residual": (llama_residual_k[0], llama_residual_v[0]),
            }.items():
                key = torch.cat((question_k, mapped_k), dim=1)
                value = torch.cat((question_v, mapped_v), dim=1)
                logits = final_logits(llama, llama_encoded["receiver_answer"], key, value)[choice_index]
                prediction = int(logits.argmax())
                reverse_outputs[branch][condition] = {
                    "prediction": prediction,
                    "correct": prediction == row["gold_index"],
                    "choice_logits": logits.cpu().tolist(),
                    "choice_kl_to_oracle": choice_kl(logits, native_logits, cfg["temperature"]),
                    "k_representation": representation(mapped_k, llama_native_k),
                    "v_representation": representation(mapped_v, llama_native_v),
                }

        qwen_native_k, qwen_native_v = qwen_inputs["qwen_native_bridge"]
        qwen_rep = {}
        for name, (key, value) in qwen_inputs.items():
            qwen_rep[name] = {
                "k": representation(key[0], qwen_native_k[0]),
                "v": representation(value[0], qwen_native_v[0]),
            }
        records.append({
            "id": sample_id,
            "gold_index": row["gold_index"],
            "llama_target_indices": target_indices,
            "llama_native_oracle32": {
                "prediction": native_prediction,
                "correct": native_prediction == row["gold_index"],
                "choice_logits": native_logits.cpu().tolist(),
            },
            "qwen_space_representation": qwen_rep,
            "reverse_branches": reverse_outputs,
        })
        log(f"Composability audit: {number}/{len(rows)} id={sample_id}")

    metrics = {}
    for branch in ("reverse_stage_a", "reverse_residual"):
        metrics[branch] = {condition: aggregate(records, branch, condition)
                           for condition in ("qwen_native_bridge", "gemma_stage_a", "gemma_stage_b_residual")}
    native_correct = [record["llama_native_oracle32"]["correct"] for record in records]
    pairwise = {}
    for branch in ("reverse_stage_a", "reverse_residual"):
        a = [record["reverse_branches"][branch]["gemma_stage_a"]["correct"] for record in records]
        b = [record["reverse_branches"][branch]["gemma_stage_b_residual"]["correct"] for record in records]
        bridge = [record["reverse_branches"][branch]["qwen_native_bridge"]["correct"] for record in records]
        pairwise[branch] = {
            "stage_a_vs_stage_b": paired(a, b),
            "native_bridge_vs_stage_b": paired(bridge, b),
            "llama_native_oracle_vs_stage_b": paired(native_correct, b),
        }

    forward_comparison = read_json(forward_root / "runs/study/results/comparison.json")
    result = {
        "status": "completed",
        "protocol": cfg["protocol"],
        "sample_count": len(records),
        "definition": "Freeze Gemma->Qwen Stage-A/Residual Stage-B and Qwen->Llama Stage-A/Residual writers; vary only the Qwen-space KV passed between them.",
        "checkpoints": {
            name: {"path": cfg[name], "sha256": file_sha256(cfg[name])}
            for name in ("forward_stage_a_checkpoint", "forward_residual_checkpoint",
                         "reverse_stage_a_checkpoint", "reverse_residual_checkpoint")
        },
        "upstream_qwen_metrics_from_frozen_forward_experiment": {
            "stage_a": forward_comparison["metrics"]["stage_a"],
            "stage_b_residual": forward_comparison["metrics"]["residual"],
            "functional_accuracy_gain": (forward_comparison["metrics"]["residual"]["accuracy"] -
                                           forward_comparison["metrics"]["stage_a"]["accuracy"]),
        },
        "llama_native_oracle32": {
            "correct": sum(native_correct), "accuracy": sum(native_correct) / len(native_correct)
        },
        "metrics": metrics,
        "pairwise_correctness": pairwise,
        "primary_deltas": {
            branch: {
                "stage_b_minus_stage_a_accuracy": (metrics[branch]["gemma_stage_b_residual"]["accuracy"] -
                                                     metrics[branch]["gemma_stage_a"]["accuracy"]),
                "stage_b_minus_stage_a_agreement": (
                    metrics[branch]["gemma_stage_b_residual"]["agreement_with_llama_native_oracle32"] -
                    metrics[branch]["gemma_stage_a"]["agreement_with_llama_native_oracle32"]),
                "stage_b_minus_stage_a_choice_kl": (
                    metrics[branch]["gemma_stage_b_residual"]["choice_kl_to_llama_native_oracle32"] -
                    metrics[branch]["gemma_stage_a"]["choice_kl_to_llama_native_oracle32"]),
            } for branch in ("reverse_stage_a", "reverse_residual")
        },
    }
    out = ROOT / "runs/study/results"
    save_json(out / "composability_summary.json", result)
    with (out / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    save_json(out / "checkpoint_manifest.json", result["checkpoints"])
    log(json.dumps(result["primary_deltas"], ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    args = parser.parse_args()
    cfg = read_json(args.config)
    torch.manual_seed(cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()
