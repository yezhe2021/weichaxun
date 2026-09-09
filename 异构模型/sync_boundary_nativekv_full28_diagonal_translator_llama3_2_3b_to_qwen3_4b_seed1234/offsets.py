from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSpan:
    index: int
    token_id: int
    start: int
    end: int

    @property
    def special(self):
        return self.start == self.end


def token_spans(tokenizer, text, expected_ids):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    ids = bos + list(encoded["input_ids"])
    offsets = [(0, 0)] * len(bos) + [tuple(x) for x in encoded["offset_mapping"]]
    if ids != list(expected_ids):
        raise RuntimeError("Offset-tokenization IDs do not match Native KV manifest")
    spans = [TokenSpan(i, token_id, int(a), int(b)) for i, (token_id, (a, b)) in enumerate(zip(ids, offsets))]
    if any(s.start < 0 or s.end < s.start or s.end > len(text) for s in spans):
        raise RuntimeError("Invalid tokenizer offset mapping")
    return spans


def span_distance(point, span):
    if span.start <= point < span.end:
        return 0
    return span.start - point if point < span.start else point - span.end + 1


def overlap(a, b):
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def span_iou(a, b):
    intersection = overlap(a, b)
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union if union else 0.0
