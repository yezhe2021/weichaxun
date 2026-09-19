import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from translators import LlamaToQwen, QwenToGemma, ResidualAdapter

ROOT = Path(__file__).resolve().parent


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


def text_config(model_or_config):
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "text_config", config)


def text_backbone(model):
    return getattr(model.model, "language_model", model.model)


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
        raise RuntimeError("Offset-tokenization mismatch")
    offsets = [(0, 0)] * len(bos) + [tuple(item) for item in encoded["offset_mapping"]]
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


def encode_gemma(tokenizer, row):
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    body = list(tokenizer(row["body"], add_special_tokens=False)["input_ids"])
    full = enc(row["full_prompt"])
    question = enc(row["question_prefix"])
    if body + enc(row["answer_prefix"]) != full:
        raise RuntimeError("Gemma body/Answer boundary is not compositional")
    if full[:len(question)] != question:
        raise RuntimeError("Gemma Question boundary is not compositional")
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    choice_ids = []
    for label in row["labels"]:
        continuation = enc(row["full_prompt"] + " " + label)
        if continuation[:len(full)] != full or len(continuation) != len(full) + 1:
            raise RuntimeError(f"Unstable Gemma choice token: {label}")
        choice_ids.append(continuation[-1])
    if len(set(choice_ids)) != len(choice_ids):
        raise RuntimeError("Duplicate Gemma choice token IDs")
    return {"body": bos + body, "full": bos + full,
            "question_prefix_length": len(bos) + len(question),
            "receiver_answer": enc(row["receiver_answer"]), "choice_ids": choice_ids}


def rotate_half(tensor):
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rotary_embeddings(model, tensor, positions, layer_index):
    backbone = text_backbone(model)
    positions = positions.to(tensor.device).unsqueeze(0)
    config = text_config(model)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None or not isinstance(getattr(backbone.rotary_emb, "rope_type", None), dict):
        return backbone.rotary_emb(tensor.unsqueeze(0), positions)
    return backbone.rotary_emb(tensor.unsqueeze(0), positions, layer_types[layer_index])


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


@torch.no_grad()
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
def capture_gemma(model, full_ids, body_length):
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
            value = module.v_proj(hidden).view(shape)
            captured_k[index], captured_v[index] = key[0, :body_length].cpu(), value[0, :body_length].cpu()
        return apply

    for index, layer in enumerate(backbone.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)
        output = backbone(input_ids=ids, attention_mask=torch.ones_like(ids),
                          position_ids=torch.arange(len(full_ids), device="cuda")[None], use_cache=False)
        full_logits = model.lm_head(output.last_hidden_state[:, -1])[0].float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack([captured_k[i] for i in range(34)]), \
           torch.stack([captured_v[i] for i in range(34)]), full_logits


def load_state(module, checkpoint, expected_kind=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if expected_kind is not None and payload.get("kind") != expected_kind:
        raise RuntimeError(f"Expected checkpoint kind={expected_kind}, got {payload.get('kind')}")
    module.load_state_dict(payload["state"], strict=True)
    return module.cuda().eval().requires_grad_(False)


def representation(prediction, target):
    prediction, target = prediction.float(), target.float()
    return {"nmse": ((prediction - target).square().mean() /
                     target.square().mean().clamp_min(1e-8)).item(),
            "cosine": F.cosine_similarity(prediction, target, dim=-1).mean().item()}


def choice_kl(student, teacher, temperature):
    student = F.log_softmax(student.float() / temperature, dim=-1)
    teacher = F.log_softmax(teacher.float() / temperature, dim=-1)
    return (F.kl_div(student, teacher, reduction="sum", log_target=True) * temperature ** 2).item()


def paired(first, second):
    both = sum(a and b for a, b in zip(first, second))
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    return {"both_correct": both, "only_first_correct": only_first,
            "only_second_correct": only_second,
            "both_wrong": len(first) - both - only_first - only_second,
            "second_minus_first": (only_second - only_first) / len(first)}


def aggregate(records, branch, condition):
    rows = [item["downstream_branches"][branch][condition] for item in records]
    oracle = [item["gemma_native_oracle32"] for item in records]
    count = len(rows)
    return {"correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / count,
            "agreement_with_gemma_native_oracle32": sum(
                row["prediction"] == target["prediction"] for row, target in zip(rows, oracle)) / count,
            "choice_kl_to_gemma_native_oracle32": sum(row["choice_kl_to_oracle"] for row in rows) / count,
            "k_nmse_to_gemma_native": sum(row["k_representation"]["nmse"] for row in rows) / count,
            "v_nmse_to_gemma_native": sum(row["v_representation"]["nmse"] for row in rows) / count,
            "k_cosine_to_gemma_native": sum(row["k_representation"]["cosine"] for row in rows) / count,
            "v_cosine_to_gemma_native": sum(row["v_representation"]["cosine"] for row in rows) / count}


@torch.no_grad()
def run(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    upstream_root = Path(cfg["upstream_root"])
    manifest = read_json(upstream_root / "runs/study/manifests/test.json")
    rows = manifest["rows"][:cfg["test_samples"]]
    if len(rows) != cfg["test_samples"]:
        raise RuntimeError("Wrong test sample count")

    upstream_base = load_state(LlamaToQwen(cfg["mlp_hidden_dim"]), cfg["upstream_stage_a_checkpoint"])
    upstream_adapter = load_state(ResidualAdapter(36, 8, 128, cfg["adapter_rank"]),
                                  cfg["upstream_residual_checkpoint"], "residual")
    downstream_base = load_state(QwenToGemma(cfg["mlp_hidden_dim"], cfg["depth_output_dim"]),
                                 cfg["downstream_stage_a_checkpoint"])
    downstream_adapter = load_state(ResidualAdapter(34, 4, 256, cfg["adapter_rank"]),
                                    cfg["downstream_residual_checkpoint"], "residual")

    gemma_tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["gemma"], local_files_only=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["qwen"], local_files_only=True)
    gemma = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["gemma"], local_files_only=True, dtype=torch.float32,
        attn_implementation=cfg["attention_implementation"])
    del gemma.model.vision_tower
    del gemma.model.multi_modal_projector
    gemma = gemma.cuda().eval().requires_grad_(False)
    geometry = (text_config(gemma).num_hidden_layers, text_config(gemma).num_key_value_heads,
                text_config(gemma).head_dim)
    if geometry != (34, 4, 256):
        raise RuntimeError(f"Unexpected Gemma geometry: {geometry}")

    records = []
    for number, row in enumerate(rows, 1):
        sample_id = row["id"]
        source = torch.load(upstream_root / "runs/study/source_selected_cache/test" / f"{sample_id}.pt",
                            map_location="cpu", weights_only=True)
        pair = torch.load(upstream_root / "runs/study/pair_cache/test" / f"{sample_id}.pt",
                          map_location="cpu", weights_only=True)
        source_k = source["source_k"][:, 1:].cuda()[None]
        source_v = source["source_v"][:, 1:].cuda()[None]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            qwen_stage_a_k, qwen_stage_a_v = upstream_base(source_k, source_v)
            qwen_stage_b_k, qwen_stage_b_v = upstream_adapter(qwen_stage_a_k, qwen_stage_a_v)
        qwen_inputs = {
            "qwen_native_bridge": (pair["target_k"][:, 1:].cuda()[None],
                                   pair["target_v"][:, 1:].cuda()[None]),
            "llama_stage_a": (qwen_stage_a_k, qwen_stage_a_v),
            "llama_stage_b_residual": (qwen_stage_b_k, qwen_stage_b_v),
        }

        encoded = encode_gemma(gemma_tokenizer, row)
        gemma_k, gemma_v, _ = capture_gemma(gemma, encoded["full"], len(encoded["body"]))
        gemma_spans = token_spans(gemma_tokenizer, row["body"], encoded["body"])
        qwen_spans = {span.index: span for span in token_spans(
            qwen_tokenizer, row["body"], row["encoded"]["qwen"]["body"])}
        option_start, option_end = row["options_char_span"]
        gemma_options = [span for span in gemma_spans
                         if min(span.end, option_end) > max(span.start, option_start)]
        target_indices = [target_token(anchor["anchor_char"], qwen_spans[anchor["target_index"]],
                                       gemma_options).index for anchor in pair["metadata"]["anchors"]]
        gemma_native_k, gemma_native_v = gemma_k[:, target_indices].cuda(), gemma_v[:, target_indices].cuda()
        question_k = gemma_k[:, :encoded["question_prefix_length"]].cuda()
        question_v = gemma_v[:, :encoded["question_prefix_length"]].cuda()
        choice_index = torch.tensor(encoded["choice_ids"], device="cuda")
        oracle_logits = final_logits(gemma, encoded["receiver_answer"],
                                     torch.cat((question_k, gemma_native_k), dim=1),
                                     torch.cat((question_v, gemma_native_v), dim=1))[choice_index]
        oracle_prediction = int(oracle_logits.argmax())

        branches = {"downstream_stage_a": {}, "downstream_residual": {}}
        for condition, (qwen_k, qwen_v) in qwen_inputs.items():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                gemma_a_k, gemma_a_v = downstream_base(qwen_k, qwen_v)
                gemma_b_k, gemma_b_v = downstream_adapter(gemma_a_k, gemma_a_v)
            for branch, (mapped_k, mapped_v) in {
                "downstream_stage_a": (gemma_a_k[0], gemma_a_v[0]),
                "downstream_residual": (gemma_b_k[0], gemma_b_v[0]),
            }.items():
                logits = final_logits(gemma, encoded["receiver_answer"],
                                      torch.cat((question_k, mapped_k), dim=1),
                                      torch.cat((question_v, mapped_v), dim=1))[choice_index]
                prediction = int(logits.argmax())
                branches[branch][condition] = {
                    "prediction": prediction, "correct": prediction == row["gold_index"],
                    "choice_logits": logits.cpu().tolist(),
                    "choice_kl_to_oracle": choice_kl(logits, oracle_logits, cfg["temperature"]),
                    "k_representation": representation(mapped_k, gemma_native_k),
                    "v_representation": representation(mapped_v, gemma_native_v)}

        qwen_native_k, qwen_native_v = qwen_inputs["qwen_native_bridge"]
        qwen_rep = {name: {"k": representation(key[0], qwen_native_k[0]),
                           "v": representation(value[0], qwen_native_v[0])}
                    for name, (key, value) in qwen_inputs.items()}
        records.append({"id": sample_id, "gold_index": row["gold_index"],
                        "gemma_target_indices": target_indices,
                        "gemma_native_oracle32": {"prediction": oracle_prediction,
                                                  "correct": oracle_prediction == row["gold_index"],
                                                  "choice_logits": oracle_logits.cpu().tolist()},
                        "qwen_space_representation": qwen_rep,
                        "downstream_branches": branches})
        print(f"Llama-Qwen-Gemma audit: {number}/{len(rows)} id={sample_id}", flush=True)

    conditions = ("qwen_native_bridge", "llama_stage_a", "llama_stage_b_residual")
    metrics = {branch: {condition: aggregate(records, branch, condition) for condition in conditions}
               for branch in ("downstream_stage_a", "downstream_residual")}
    oracle_correct = [row["gemma_native_oracle32"]["correct"] for row in records]
    pairwise = {}
    for branch in metrics:
        stage_a = [row["downstream_branches"][branch]["llama_stage_a"]["correct"] for row in records]
        stage_b = [row["downstream_branches"][branch]["llama_stage_b_residual"]["correct"] for row in records]
        bridge = [row["downstream_branches"][branch]["qwen_native_bridge"]["correct"] for row in records]
        pairwise[branch] = {"stage_a_vs_stage_b": paired(stage_a, stage_b),
                            "native_bridge_vs_stage_b": paired(bridge, stage_b),
                            "gemma_oracle_vs_stage_b": paired(oracle_correct, stage_b)}

    upstream_result = read_json(upstream_root / "runs/study/results/comparison.json")
    result = {
        "status": "completed", "protocol": cfg["protocol"], "sample_count": len(records),
        "definition": "Freeze Llama->Qwen Stage-A/Residual Stage-B and Qwen->Gemma Stage-A/Residual writers; vary only the Qwen Hub state passed between them.",
        "checkpoints": {name: {"path": cfg[name], "sha256": file_sha256(cfg[name])}
                        for name in ("upstream_stage_a_checkpoint", "upstream_residual_checkpoint",
                                     "downstream_stage_a_checkpoint", "downstream_residual_checkpoint")},
        "upstream_qwen_metrics": {"stage_a": upstream_result["metrics"]["stage_a"],
                                  "stage_b_residual": upstream_result["metrics"]["residual"],
                                  "functional_accuracy_gain": (upstream_result["metrics"]["residual"]["accuracy"] -
                                                               upstream_result["metrics"]["stage_a"]["accuracy"])},
        "gemma_native_oracle32": {"correct": sum(oracle_correct),
                                  "accuracy": sum(oracle_correct) / len(oracle_correct)},
        "metrics": metrics, "pairwise_correctness": pairwise,
        "primary_deltas": {branch: {
            "stage_b_minus_stage_a_accuracy": (metrics[branch]["llama_stage_b_residual"]["accuracy"] -
                                                 metrics[branch]["llama_stage_a"]["accuracy"]),
            "stage_b_minus_stage_a_agreement": (
                metrics[branch]["llama_stage_b_residual"]["agreement_with_gemma_native_oracle32"] -
                metrics[branch]["llama_stage_a"]["agreement_with_gemma_native_oracle32"]),
            "stage_b_minus_stage_a_choice_kl": (
                metrics[branch]["llama_stage_b_residual"]["choice_kl_to_gemma_native_oracle32"] -
                metrics[branch]["llama_stage_a"]["choice_kl_to_gemma_native_oracle32"])}
            for branch in metrics}}
    output = ROOT / "runs/study/results"
    save_json(output / "composability_summary.json", result)
    with (output / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(output / "checkpoint_manifest.json", result["checkpoints"])
    print(json.dumps(result["primary_deltas"], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    args = parser.parse_args()
    torch.manual_seed(1234)
    run(read_json(args.config))


if __name__ == "__main__":
    main()
