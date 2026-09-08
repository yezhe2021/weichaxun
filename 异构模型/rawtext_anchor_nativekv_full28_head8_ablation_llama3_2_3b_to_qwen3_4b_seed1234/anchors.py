import math

from offsets import overlap, span_distance, span_iou


def region_bounds(characters, regions):
    if characters < 1: raise ValueError("Empty raw text")
    return [(math.floor(i * characters / regions), math.floor((i + 1) * characters / regions))
            for i in range(regions)]


def eligible(spans):
    # Position zero is a separate model-native bootstrap channel. It is never an anchor candidate.
    return [span for span in spans if span.index != 0 and not span.special]


def source_anchor(region, spans, importance):
    start, end = region
    candidates = [span for span in eligible(spans) if min(span.end, end) > max(span.start, start)]
    fallback = False
    if not candidates:
        fallback = True
        center = (start + end - 1) // 2
        candidates = sorted(eligible(spans), key=lambda span: (span_distance(center, span), span.index))[:1]
    chosen = max(candidates, key=lambda span: (float(importance[span.index]),
                                               overlap(span, type("R", (), {"start": start, "end": end})()),
                                               -span.index))
    lo, hi = max(start, chosen.start), min(end, chosen.end)
    anchor = (lo + hi - 1) // 2 if hi > lo else min(max((start + end - 1) // 2, 0), end - 1)
    return chosen, anchor, fallback


def target_token(anchor, source, spans):
    candidates = eligible(spans)
    containing = [span for span in candidates if span.start <= anchor < span.end]
    fallback = not containing
    pool = containing if containing else candidates
    chosen = min(pool, key=lambda span: (span_distance(anchor, span), -overlap(source, span),
                                         span.end - span.start, span.index))
    return chosen, {"contains_anchor": not fallback, "char_distance": span_distance(anchor, chosen),
                    "span_overlap": overlap(source, chosen), "span_iou": span_iou(source, chosen)}


def build_anchors(text, source_spans, target_spans, source_importance, regions):
    records = []
    for region_id, bounds in enumerate(region_bounds(len(text), regions)):
        source, anchor, source_fallback = source_anchor(bounds, source_spans, source_importance)
        target, alignment = target_token(anchor, source, target_spans)
        records.append({"region": region_id, "region_start": bounds[0], "region_end": bounds[1],
                        "anchor_char": anchor, "anchor_normalized": (anchor + .5) / len(text),
                        "source_index": source.index, "source_start": source.start, "source_end": source.end,
                        "source_text": text[source.start:source.end], "source_region_fallback": source_fallback,
                        "target_index": target.index, "target_start": target.start, "target_end": target.end,
                        "target_text": text[target.start:target.end], **alignment})
    return records


def self_anchors(text, spans, importance, regions):
    return build_anchors(text, spans, spans, importance, regions)


def selected_mass(importance, indices):
    denominator = importance[1:].sum().clamp_min(1e-12).item()
    return importance[sorted(set(indices))].sum().item() / denominator
