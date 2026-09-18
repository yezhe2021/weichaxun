import hashlib
import json
import math
import random
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as parquet
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parent
LABELS = "ABCDEFGHIJ"


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def save_tensor(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary); temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def text_config(model_or_config):
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "text_config", config)


def text_backbone(model):
    return getattr(model.model, "language_model", model.model)


def tokenizer(path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def load_model(cfg, family):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    dtype = torch.float32 if family == "gemma" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg["models"][family], local_files_only=True, dtype=dtype,
        attn_implementation=cfg["attention_implementation"])
    if family == "gemma":
        del model.model.vision_tower
        del model.model.multi_modal_projector
    model = model.cuda().eval().requires_grad_(False)
    config = text_config(model)
    observed = (config.num_hidden_layers, config.num_key_value_heads, config.head_dim)
    expected = {"llama": (28, 8, 128), "qwen": (36, 8, 128), "gemma": (34, 4, 256)}[family]
    if observed != expected:
        raise RuntimeError(f"{family} geometry mismatch: {observed} != {expected}")
    return model


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def token_set(text):
    return set(normalize(text).split())


def openbook_training_questions(path, seed):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    return [item["question"]["stem"] for item in rows[:4096]]


def raw_dataset_rows(cfg, dataset):
    if dataset == "arc_challenge":
        member = "ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl"
        with zipfile.ZipFile(cfg["datasets"][dataset]) as archive:
            lines = archive.read(member).decode("utf-8").splitlines()
        output = []
        for item in map(json.loads, lines):
            choices = sorted(item["question"]["choices"], key=lambda x: x["label"])
            labels = [choice["label"] for choice in choices]
            if labels != list(LABELS[:len(labels)]) or len(choices) != 4:
                continue
            output.append({"id": f"arc_{item['id']}", "category": "arc_challenge",
                           "question": item["question"]["stem"],
                           "options": [choice["text"] for choice in choices],
                           "gold_index": labels.index(item["answerKey"])})
        return output
    if dataset == "mmlu_pro":
        items = parquet.read_table(cfg["datasets"][dataset]).to_pylist()
        return [{"id": f"mmlupro_{item['question_id']}", "category": item["category"],
                 "question": item["question"], "options": list(item["options"]),
                 "gold_index": int(item["answer_index"])} for item in items]
    raise ValueError(dataset)


def select_rows(cfg, dataset, limit):
    old_questions = openbook_training_questions(cfg["datasets"]["openbook_train"], cfg["seed"])
    old_normalized = {normalize(question) for question in old_questions}
    old_sets = [token_set(question) for question in old_questions]
    candidates = raw_dataset_rows(cfg, dataset)
    if dataset == "mmlu_pro":
        groups = {}
        for row in candidates:
            groups.setdefault(row["category"], []).append(row)
        rng = random.Random(cfg["seed"] + 17)
        for rows in groups.values(): rng.shuffle(rows)
        ordered = []
        while any(groups.values()):
            for name in sorted(groups):
                if groups[name]: ordered.append(groups[name].pop())
        candidates = ordered
    else:
        random.Random(cfg["seed"] + 11).shuffle(candidates)

    selected, rejected, similarities = [], 0, []
    for row in candidates:
        normalized = normalize(row["question"])
        words = token_set(row["question"])
        exact = normalized in old_normalized
        maximum = max((len(words & other) / max(len(words | other), 1) for other in old_sets), default=0.0)
        if exact or maximum >= 0.9:
            rejected += 1
            continue
        similarities.append(maximum); selected.append(row)
        if len(selected) == limit: break
    if len(selected) != limit:
        raise RuntimeError(f"Insufficient leak-free {dataset} rows: {len(selected)}")
    audit = {"dataset": dataset, "raw_candidates": len(raw_dataset_rows(cfg, dataset)),
             "selected": len(selected), "excluded_before_selection": rejected,
             "max_question_jaccard": max(similarities),
             "mean_question_jaccard": sum(similarities) / len(similarities),
             "openbook_reference_count": len(old_questions),
             "selection": "seed1234; ARC exact four-choice; MMLU-Pro category round-robin; leakage threshold 0.9"}
    return selected, audit


def serialize(row):
    options = [str(value).strip() for value in row["options"]]
    if not 2 <= len(options) <= len(LABELS): raise RuntimeError("Unsupported option count")
    labels = list(LABELS[:len(options)])
    if not 0 <= int(row["gold_index"]) < len(options): raise RuntimeError("Invalid gold index")
    question = str(row["question"]).strip()
    question_header = "Question:\n"
    question_start, question_end = len(question_header), len(question_header) + len(question)
    options_header = "\n\nOptions:\n"
    options_start = question_end + len(options_header)
    options_text = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
    options_end = options_start + len(options_text)
    body = question_header + question + options_header + options_text + "\n\n"
    answer_prefix = "Answer:"
    return {"body": body, "full_prompt": body + answer_prefix,
            "question_prefix": body[:options_start],
            "receiver_answer": body[options_end:] + answer_prefix,
            "answer_prefix": answer_prefix, "options_char_span": [options_start, options_end],
            "question_char_span": [question_start, question_end], "labels": labels,
            "gold_index": int(row["gold_index"]), "options": options}


def encode(tok, serialized):
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    body_encoding = tok(serialized["body"], add_special_tokens=False, return_offsets_mapping=True)
    body_plain = list(body_encoding["input_ids"])
    full_plain = enc(serialized["full_prompt"])
    answer = enc(serialized["answer_prefix"])
    question_plain = enc(serialized["question_prefix"])
    if body_plain + answer != full_plain: raise RuntimeError("Noncompositional Answer boundary")
    if full_plain[:len(question_plain)] != question_plain: raise RuntimeError("Noncompositional Question boundary")
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    choice_ids = []
    for label in serialized["labels"]:
        continuation = enc(serialized["full_prompt"] + " " + label)
        if continuation[:len(full_plain)] != full_plain or len(continuation) != len(full_plain) + 1:
            raise RuntimeError(f"Unstable choice token {label}")
        choice_ids.append(continuation[-1])
    if len(set(choice_ids)) != len(choice_ids): raise RuntimeError("Duplicate choice IDs")
    start, end = serialized["options_char_span"]
    option_indices = [len(bos) + index for index, (left, right) in enumerate(body_encoding["offset_mapping"])
                      if min(int(right), end) > max(int(left), start)]
    return {"body": bos + body_plain, "full": bos + full_plain,
            "question_prefix_length": len(bos) + len(question_plain),
            "receiver_answer": enc(serialized["receiver_answer"]),
            "choice_ids": choice_ids, "option_token_indices": option_indices,
            "offsets": [(0, 0)] * len(bos) + [list(pair) for pair in body_encoding["offset_mapping"]]}


@dataclass(frozen=True)
class Span:
    index: int
    start: int
    end: int

    @property
    def special(self): return self.start == self.end


def spans_from_encoded(encoded):
    return [Span(index, int(pair[0]), int(pair[1])) for index, pair in enumerate(encoded["offsets"])]


def span_distance(point, span):
    if span.start <= point < span.end: return 0
    return span.start - point if point < span.start else point - span.end + 1


def overlap(first, second):
    return max(0, min(first.end, second.end) - max(first.start, second.start))


def region_bounds(characters, regions):
    return [(math.floor(index * characters / regions), math.floor((index + 1) * characters / regions))
            for index in range(regions)]


def source_anchor(region, spans, importance):
    start, end = region
    candidates = [span for span in spans if overlap(span, Span(-1, start, end)) > 0]
    fallback = not candidates
    if not candidates:
        center = (start + end - 1) // 2
        candidates = sorted(spans, key=lambda span: (span_distance(center, span), span.index))[:1]
    chosen = max(candidates, key=lambda span: (float(importance[span.index]), -span.index))
    lo, hi = max(chosen.start, start), min(chosen.end, end)
    anchor = (lo + hi - 1) // 2 if hi > lo else min(max((start + end - 1) // 2, 0), end - 1)
    return chosen, anchor, fallback


def target_token(anchor, source, spans):
    candidates = [span for span in spans if span.index and not span.special]
    containing = [span for span in candidates if span.start <= anchor < span.end]
    pool = containing or candidates
    chosen = min(pool, key=lambda span: (span_distance(anchor, span), -overlap(source, span),
                                         span.end - span.start, span.index))
    return chosen


def build_anchors(text, source_spans, target_spans, importance, regions, region_start, region_end):
    width = region_end - region_start
    if width <= 0 or not source_spans or not target_spans: raise RuntimeError("Invalid anchor region")
    records = []
    for left, right in region_bounds(width, regions):
        bounds = (region_start + left, region_start + right)
        source, anchor, fallback = source_anchor(bounds, source_spans, importance)
        target = target_token(anchor, source, target_spans)
        records.append({"source_index": source.index, "target_index": target.index,
                        "anchor_char": anchor, "source_fallback": fallback,
                        "source_text": text[source.start:source.end],
                        "target_text": text[target.start:target.end]})
    return records


def rotate_half(tensor):
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rotary_embeddings(model, tensor, positions, layer_index):
    backbone = text_backbone(model); positions = positions.to(tensor.device).unsqueeze(0)
    config = text_config(model); layer_types = getattr(config, "layer_types", None)
    if layer_types is None or not isinstance(getattr(backbone.rotary_emb, "rope_type", None), dict):
        return backbone.rotary_emb(tensor.unsqueeze(0), positions)
    return backbone.rotary_emb(tensor.unsqueeze(0), positions, layer_types[layer_index])


def apply_rope(model, tensor, positions, layer_index):
    cos, sin = rotary_embeddings(model, tensor, positions, layer_index)
    return tensor * cos[0, :, None] + rotate_half(tensor) * sin[0, :, None]


@torch.no_grad()
def capture_decision(model, full_ids, body_length, choice_ids):
    backbone = text_backbone(model); captured_k, captured_v, importance, handles = {}, {}, {}, []

    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            if hasattr(module, "k_norm"):
                try: key = module.k_norm(key.transpose(1, 2)).transpose(1, 2)
                except (RuntimeError, ValueError): key = module.k_norm(key)
            value = module.v_proj(hidden).view(shape)
            query = module.q_proj(hidden[:, -1:]).view(1, 1, module.config.num_attention_heads, module.head_dim)
            if hasattr(module, "q_norm"):
                try: query = module.q_norm(query.transpose(1, 2)).transpose(1, 2)
                except (RuntimeError, ValueError): query = module.q_norm(query)
            supplied = kwargs.get("position_embeddings")
            if supplied is None:
                body_key = apply_rope(model, key[0, :body_length], torch.arange(body_length, device="cuda"), index)
                decision_query = apply_rope(model, query[0], torch.tensor([len(full_ids)-1], device="cuda"), index)[0]
            else:
                cos, sin = supplied
                body_key = key[0, :body_length] * cos[0, :body_length, None] + rotate_half(key[0, :body_length]) * sin[0, :body_length, None]
                decision_query = (query[0] * cos[0, -1:, None] + rotate_half(query[0]) * sin[0, -1:, None])[0]
            repeat = module.config.num_attention_heads // module.config.num_key_value_heads
            repeated = body_key.repeat_interleave(repeat, dim=1)
            scaling = getattr(module, "scaling", 1.0 / math.sqrt(module.head_dim))
            scores = torch.einsum("hd,thd->ht", decision_query.float(), repeated.float()) * scaling
            candidate = torch.softmax(scores[:, 1:], dim=-1).mean(0)
            importance[index] = torch.cat((torch.zeros(1, device=hidden.device), candidate)).cpu()
            captured_k[index], captured_v[index] = key[0, :body_length].cpu(), value[0, :body_length].cpu()
        return apply

    for index, layer in enumerate(backbone.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)
        output = backbone(input_ids=ids, attention_mask=torch.ones_like(ids),
                          position_ids=torch.arange(len(full_ids), device="cuda")[None], use_cache=False)
        logits = model.lm_head(output.last_hidden_state[:, -1])[0].float()
    finally:
        for handle in handles: handle.remove()
    layers = len(backbone.layers)
    score = torch.stack([importance[index] for index in range(layers)]).mean(0)
    score[0] = 0; score /= score.sum().clamp_min(1e-12)
    return (torch.stack([captured_k[index] for index in range(layers)]),
            torch.stack([captured_v[index] for index in range(layers)]), score,
            logits[torch.tensor(choice_ids, device=logits.device)].cpu())


def make_cache(model, key, value):
    positions = torch.arange(key.shape[1], device="cuda")
    dtype = next(model.parameters()).dtype
    key, value = key.to(dtype), value.to(dtype)
    rotated = torch.stack([apply_rope(model, layer, positions, index) for index, layer in enumerate(key)])
    data = [(rotated[layer].permute(1, 0, 2).unsqueeze(0),
             value[layer].permute(1, 0, 2).unsqueeze(0)) for layer in range(key.shape[0])]
    return DynamicCache(ddp_cache_data=data, config=text_config(model))


@torch.no_grad()
def final_logits(model, ids, key=None, value=None):
    cache = make_cache(model, key, value) if key is not None else None
    prefix = key.shape[1] if key is not None else 0
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    output = text_backbone(model)(input_ids=input_ids,
        attention_mask=torch.ones((1, prefix + len(ids)), device="cuda", dtype=torch.long),
        position_ids=torch.arange(prefix, prefix + len(ids), device="cuda")[None],
        past_key_values=cache, use_cache=False)
    return model.lm_head(output.last_hidden_state[:, -1])[0].float()
