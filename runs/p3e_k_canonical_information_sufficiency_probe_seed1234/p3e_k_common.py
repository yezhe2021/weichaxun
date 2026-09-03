import json
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(value):
    return " ".join(re.sub(r"[^\w\s]", " ", str(value).lower()).split())


def token_span_targets(offsets, answer_ranges):
    targets = []
    for left, right in answer_ranges:
        hits = [
            index for index, (start, end) in enumerate(offsets)
            if start != end and start < right and end > left
        ]
        if hits:
            targets.append((hits[0], hits[-1]))
    return sorted(set(targets))


def answer_occurrences(evidence, answer):
    if not answer or normalize_answer(answer) in {"yes", "no"}:
        return []
    return [
        (match.start(), match.end())
        for match in re.finditer(re.escape(answer), evidence, flags=re.IGNORECASE)
    ]


def sentence_annotations(evidence, offsets, supporting_facts):
    sentence_ids = torch.full((len(offsets),), -1, dtype=torch.long)
    sentence_keys, spans = [], []
    current_title = None
    cursor = 0
    for line in evidence.splitlines(keepends=True):
        clean = line.rstrip("\r\n")
        if clean.startswith("TITLE:"):
            current_title = clean.split(":", 1)[1].strip()
        match = re.match(r"^\[(\d+)\]\s*", clean)
        if match and current_title is not None:
            sentence_id = len(sentence_keys)
            sentence_keys.append([current_title, int(match.group(1))])
            left = cursor + match.end()
            right = cursor + len(clean)
            spans.append((left, right))
            for token_index, (start, end) in enumerate(offsets):
                if start != end and start < right and end > left:
                    sentence_ids[token_index] = sentence_id
        cursor += len(line)
    gold_keys = {(str(title), int(index)) for title, index in supporting_facts}
    gold_sentence_ids = [
        index for index, key in enumerate(sentence_keys)
        if (str(key[0]), int(key[1])) in gold_keys
    ]
    return sentence_ids, sentence_keys, spans, gold_sentence_ids


class ProbeCache:
    def __init__(
        self, canonical_index, native_index, sidecar_index, data_path, capacity=2,
    ):
        self.canonical_root = Path(canonical_index).parent
        self.native_root = Path(native_index).parent
        self.sidecar_root = Path(sidecar_index).parent
        self.canonical_entries = read_json(canonical_index)["entries"]
        self.native_entries = read_json(native_index)["entries"]
        self.sidecar_entries = read_json(sidecar_index)["entries"]
        self.rows = read_jsonl(data_path)
        self.canonical_entries = self.canonical_entries[:len(self.rows)]
        self.native_entries = self.native_entries[:len(self.rows)]
        self.sidecar_entries = self.sidecar_entries[:len(self.rows)]
        if not (
            len(self.canonical_entries)
            == len(self.native_entries)
            == len(self.sidecar_entries)
            == len(self.rows)
        ):
            raise RuntimeError("Probe cache lengths differ")
        for index, row in enumerate(self.rows):
            ids = (
                self.canonical_entries[index]["id"],
                self.native_entries[index]["id"],
                self.sidecar_entries[index]["id"],
                row["id"],
            )
            if len(set(ids)) != 1:
                raise RuntimeError(f"Probe cache ID mismatch at {index}: {ids}")
        self.capacity = int(capacity)
        self.loaded = OrderedDict()

    @staticmethod
    def _path(root, entry):
        path = Path(entry["file"])
        return path if path.is_absolute() else root / path

    def __len__(self):
        return len(self.rows)

    def load(self, index):
        if index not in self.loaded:
            canonical = torch.load(
                self._path(self.canonical_root, self.canonical_entries[index]),
                map_location="cpu", weights_only=False,
            )
            native = torch.load(
                self._path(self.native_root, self.native_entries[index]),
                map_location="cpu", weights_only=False,
            )
            sidecar = torch.load(
                self._path(self.sidecar_root, self.sidecar_entries[index]),
                map_location="cpu", weights_only=False,
            )
            tokens = canonical["keys"].shape[1]
            if canonical["keys"].shape != canonical["values"].shape:
                raise RuntimeError("Canonical K/V mismatch")
            if canonical["keys"].shape != (16, tokens, 16, 128):
                raise RuntimeError(f"Unexpected Canonical shape {tuple(canonical['keys'].shape)}")
            if native["keys"].shape != (16, tokens, 1024):
                raise RuntimeError(f"Unexpected Native shape {tuple(native['keys'].shape)}")
            if sidecar["full_text_hidden"].shape != (tokens, 2560):
                raise RuntimeError("Full-text representation token mismatch")
            if sidecar["question"].shape != (2560,):
                raise RuntimeError("Question vector shape mismatch")
            if sidecar["sentence_ids"].numel() != tokens:
                raise RuntimeError("Sentence ID token mismatch")
            self.loaded[index] = {
                "row": self.rows[index],
                "canonical_keys": canonical["keys"].float(),
                "canonical_values": canonical["values"].float(),
                "native_keys": native["keys"].reshape(16, tokens, 8, 128).float(),
                "native_values": native["values"].reshape(16, tokens, 8, 128).float(),
                "mask": canonical["mask"].bool(),
                "support_mask": canonical["support_mask"].bool(),
                "token_ids": torch.as_tensor(native["metadata"]["token_ids"], dtype=torch.long),
                "offsets": native["metadata"]["offsets"],
                "evidence": native["evidence"],
                **sidecar,
            }
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]


class RMSNorm(nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = float(eps)

    def forward(self, value):
        value = value.float()
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps) * self.weight


class CanonicalStructuredFrontEnd(nn.Module):
    def __init__(self, question_dim=2560, layers=16, heads=16, head_dim=128, unit_dim=16):
        super().__init__()
        self.layers, self.heads, self.head_dim, self.unit_dim = (
            int(layers), int(heads), int(head_dim), int(unit_dim)
        )
        self.k_norm = RMSNorm(head_dim)
        self.v_norm = RMSNorm(head_dim)
        self.unit_mlp = nn.Sequential(
            nn.Linear(head_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, unit_dim),
        )
        self.layer_identity = nn.Parameter(torch.empty(layers, unit_dim))
        self.head_identity = nn.Parameter(torch.empty(heads, unit_dim))
        self.question_projection = nn.Linear(question_dim, unit_dim)
        self.token_projection = nn.Sequential(
            nn.Linear(layers * heads * unit_dim, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
        )
        nn.init.normal_(self.layer_identity, std=0.02)
        nn.init.normal_(self.head_identity, std=0.02)

    def forward(self, keys, values, question):
        if keys.shape != values.shape or keys.ndim != 4:
            raise RuntimeError("Structured K/V shape mismatch")
        if keys.shape[0] != self.layers or keys.shape[-2:] != (self.heads, self.head_dim):
            raise RuntimeError(f"Expected [16,T,16,128], got {tuple(keys.shape)}")
        units = self.unit_mlp(torch.cat((self.k_norm(keys), self.v_norm(values)), dim=-1))
        units = (
            units
            + self.layer_identity[:, None, None, :]
            + self.head_identity[None, None, :, :]
        )
        conditioned = units * torch.tanh(self.question_projection(question))[None, None, None, :]
        tokens = conditioned.permute(1, 0, 2, 3).contiguous().reshape(keys.shape[1], -1)
        return self.token_projection(tokens)


class InformationSufficiencyProbe(nn.Module):
    def __init__(self, mode, max_tokens=1024):
        super().__init__()
        if mode not in {"canonical", "native", "zero", "text"}:
            raise ValueError(mode)
        self.mode = mode
        if mode in {"canonical", "native", "zero"}:
            self.structured_frontend = CanonicalStructuredFrontEnd()
            self.text_frontend = None
        else:
            self.structured_frontend = None
            self.text_frontend = nn.Sequential(
                RMSNorm(2560),
                nn.Linear(2560, 512),
                nn.SiLU(),
                nn.LayerNorm(512),
            )
        self.cls_token = nn.Parameter(torch.zeros(512))
        self.cls_question = nn.Sequential(
            RMSNorm(2560),
            nn.Linear(2560, 512),
        )
        self.position_embeddings = nn.Parameter(torch.empty(max_tokens + 1, 512))
        encoder = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_transformer = nn.TransformerEncoder(
            encoder, num_layers=4, enable_nested_tensor=False,
        )
        self.support_head = nn.Linear(512, 1)
        self.start_head = nn.Linear(512, 1)
        self.end_head = nn.Linear(512, 1)
        self.yesno_head = nn.Linear(512, 2)
        nn.init.normal_(self.position_embeddings, std=0.01)
        nn.init.normal_(self.cls_token, std=0.02)

    @staticmethod
    def duplicate_native(keys, values):
        return (
            keys.repeat_interleave(2, dim=2),
            values.repeat_interleave(2, dim=2),
        )

    def forward(self, payload, device):
        question = payload["question"].float().to(device)
        mask = payload["mask"].bool().to(device)
        if self.mode == "text":
            token_states = self.text_frontend(payload["full_text_hidden"].float().to(device))
        else:
            if self.mode == "native":
                keys, values = self.duplicate_native(
                    payload["native_keys"].to(device),
                    payload["native_values"].to(device),
                )
            else:
                keys = payload["canonical_keys"].to(device)
                values = payload["canonical_values"].to(device)
                if self.mode == "zero":
                    keys, values = torch.zeros_like(keys), torch.zeros_like(values)
            token_states = self.structured_frontend(keys, values, question)
        cls = self.cls_token + self.cls_question(question)
        sequence = torch.cat((cls[None], token_states), dim=0)
        sequence = sequence + self.position_embeddings[:sequence.shape[0]]
        sequence_mask = torch.cat((
            torch.ones(1, dtype=torch.bool, device=device), mask,
        ))
        encoded = self.token_transformer(
            sequence[None], src_key_padding_mask=(~sequence_mask)[None],
        )[0]
        cls_state, token_state = encoded[0], encoded[1:]
        invalid = ~mask
        start = self.start_head(token_state).squeeze(-1).masked_fill(invalid, -1e9)
        end = self.end_head(token_state).squeeze(-1).masked_fill(invalid, -1e9)
        return {
            "support": self.support_head(token_state).squeeze(-1),
            "start": start,
            "end": end,
            "yesno": self.yesno_head(cls_state),
            "mask": mask,
        }

    def metadata(self):
        return {
            "mode": self.mode,
            "unit_dim": 16 if self.structured_frontend is not None else None,
            "structured_token_dim": 4096 if self.structured_frontend is not None else None,
            "token_hidden": 512,
            "token_transformer_layers": 4,
            "token_transformer_heads": 8,
            "token_transformer_ffn": 2048,
            "layer_head_pooling": False,
            "native_8_to_16": "lossless_adjacent_duplication" if self.mode == "native" else None,
        }


def marginal_span_loss(start_logits, end_logits, spans):
    if not spans:
        raise RuntimeError("No valid answer spans")
    start_log = F.log_softmax(start_logits, dim=-1)
    end_log = F.log_softmax(end_logits, dim=-1)
    pair_scores = torch.stack([start_log[start] + end_log[end] for start, end in spans])
    return -torch.logsumexp(pair_scores, dim=0)


def probe_loss(output, payload):
    mask = output["mask"]
    support = payload["support_mask"].float().to(output["support"].device)
    positive = support[mask].sum()
    negative = mask.sum() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    support_loss = F.binary_cross_entropy_with_logits(
        output["support"][mask], support[mask], pos_weight=pos_weight,
    )
    kind = payload["answer_kind"]
    span_loss = torch.zeros((), device=output["support"].device)
    yesno_loss = torch.zeros_like(span_loss)
    if kind == "span":
        span_loss = marginal_span_loss(output["start"], output["end"], payload["answer_spans"])
    elif kind in {"yes", "no"}:
        target = torch.tensor([0 if kind == "yes" else 1], device=output["support"].device)
        yesno_loss = F.cross_entropy(output["yesno"][None], target)
    return support_loss + span_loss + yesno_loss, {
        "support_loss": support_loss,
        "span_loss": span_loss,
        "yesno_loss": yesno_loss,
    }


def top_spans(start_logits, end_logits, top_k=5, max_answer_tokens=24):
    starts = start_logits.topk(min(24, start_logits.numel())).indices.tolist()
    ends = end_logits.topk(min(24, end_logits.numel())).indices.tolist()
    candidates = []
    for start in starts:
        for end in ends:
            if start <= end < start + max_answer_tokens:
                candidates.append((float(start_logits[start] + end_logits[end]), start, end))
    candidates.sort(reverse=True)
    return [(start, end, score) for score, start, end in candidates[:top_k]]


def decode_span(evidence, offsets, start, end):
    if start >= len(offsets) or end >= len(offsets):
        return ""
    left, right = offsets[start][0], offsets[end][1]
    return evidence[left:right].strip() if right > left else ""


def token_f1_span(predicted, gold_spans):
    predicted_tokens = set(range(predicted[0], predicted[1] + 1))
    best = 0.0
    for start, end in gold_spans:
        gold = set(range(start, end + 1))
        overlap = len(predicted_tokens & gold)
        if overlap:
            precision = overlap / len(predicted_tokens)
            recall = overlap / len(gold)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def binary_metrics(scores, targets, threshold=0.5):
    scores = scores.float().cpu()
    targets = targets.bool().cpu()
    predictions = scores >= threshold
    tp = int((predictions & targets).sum())
    fp = int((predictions & ~targets).sum())
    fn = int((~predictions & targets).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order].float()
    positives = int(targets.sum())
    if positives:
        cumulative = sorted_targets.cumsum(0)
        ranks = torch.arange(1, len(scores) + 1)
        auprc = float(((cumulative / ranks) * sorted_targets).sum() / positives)
    else:
        auprc = 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "auprc": auprc}


def sentence_metrics(scores, sentence_ids, gold_sentence_ids):
    valid_ids = sorted({int(value) for value in sentence_ids.tolist() if value >= 0})
    sentence_scores = {}
    for sentence_id in valid_ids:
        sentence_scores[sentence_id] = float(scores[sentence_ids == sentence_id].max())
    ranked = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    gold = set(int(value) for value in gold_sentence_ids)
    top2 = set(ranked[:2])
    recall_at2 = len(top2 & gold) / max(len(gold), 1)
    predicted = {key for key, score in sentence_scores.items() if score >= 0.5}
    tp = len(predicted & gold)
    precision = tp / max(len(predicted), 1)
    recall = tp / max(len(gold), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"recall_at_2": recall_at2, "f1": f1, "scores": sentence_scores}


def build_hard_negatives(rows, lengths):
    mapping = []
    for index, row in enumerate(rows):
        candidates = []
        titles = {normalize_answer(title) for title in row.get("supporting_titles", [])}
        for other_index, other in enumerate(rows):
            if other_index == index:
                continue
            if other.get("type") != row.get("type"):
                continue
            if normalize_answer(other["answer"]) == normalize_answer(row["answer"]):
                continue
            if titles & {normalize_answer(title) for title in other.get("supporting_titles", [])}:
                continue
            candidates.append((
                abs(lengths[index] - lengths[other_index]),
                abs(len(row["answer"].split()) - len(other["answer"].split())),
                other_index,
            ))
        if not candidates:
            raise RuntimeError(f"No hard negative for {row['id']}")
        mapping.append(min(candidates)[2])
    return mapping


def retarget_payload(question_payload, memory_payload):
    result = dict(memory_payload)
    row = question_payload["row"]
    evidence = memory_payload["evidence"]
    offsets = memory_payload["offsets"]
    answer = normalize_answer(row["answer"])
    if answer in {"yes", "no"}:
        result["answer_kind"] = answer
        result["answer_spans"] = []
    else:
        ranges = answer_occurrences(evidence, row["answer"])
        spans = token_span_targets(offsets, ranges)
        result["answer_kind"] = "span" if spans else "span_absent"
        result["answer_spans"] = spans
    current_facts = {
        (str(title), int(index)) for title, index in row.get("supporting_facts", [])
    }
    gold_sentence_ids = [
        index for index, key in enumerate(memory_payload["sentence_keys"])
        if (str(key[0]), int(key[1])) in current_facts
    ]
    support = torch.zeros_like(memory_payload["support_mask"])
    for sentence_id in gold_sentence_ids:
        support |= memory_payload["sentence_ids"] == sentence_id
    result["support_mask"] = support
    result["gold_sentence_ids"] = gold_sentence_ids
    result["question"] = question_payload["question"]
    result["row"] = row
    return result
