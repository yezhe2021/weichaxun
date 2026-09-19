import argparse
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from p3e_k_common import (
    InformationSufficiencyProbe,
    ProbeCache,
    binary_metrics,
    build_hard_negatives,
    decode_span,
    normalize_answer,
    retarget_payload,
    sentence_metrics,
    token_f1_span,
    top_spans,
    write_json,
    write_jsonl,
)


def answer_scores(prediction, answer):
    predicted = normalize_answer(prediction)
    target = normalize_answer(answer)
    exact = float(predicted == target)
    predicted_tokens, target_tokens = predicted.split(), target.split()
    if not predicted_tokens or not target_tokens:
        return exact, float(predicted_tokens == target_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(target_tokens)).values())
    if not overlap:
        return exact, 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(target_tokens)
    return exact, 2 * precision * recall / (precision + recall)


def load_probe(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mode = checkpoint["mode"]
    probe = InformationSufficiencyProbe(mode).to(device)
    probe.load_state_dict(checkpoint["probe"])
    probe.requires_grad_(False)
    probe.eval()
    return probe, checkpoint


def evaluate_output(output, target_payload, source_payload):
    probabilities = output["support"].sigmoid().detach().cpu()
    current_support = target_payload["support_mask"].bool()
    current_available = bool(current_support.any())
    current_token = (
        binary_metrics(probabilities, current_support)
        if current_available else None
    )
    current_sentence = (
        sentence_metrics(
            probabilities,
            target_payload["sentence_ids"],
            target_payload["gold_sentence_ids"],
        )
        if target_payload["gold_sentence_ids"] else None
    )
    source_support = source_payload["support_mask"].bool()
    source_token = binary_metrics(probabilities, source_support)
    source_sentence = sentence_metrics(
        probabilities,
        source_payload["sentence_ids"],
        source_payload["gold_sentence_ids"],
    )
    spans = top_spans(output["start"], output["end"], top_k=5)
    best_start, best_end = spans[0][:2] if spans else (0, 0)
    span_prediction = decode_span(
        source_payload["evidence"], source_payload["offsets"],
        best_start, best_end,
    )
    kind = target_payload["answer_kind"]
    if kind in {"yes", "no"}:
        prediction = "yes" if int(output["yesno"].argmax()) == 0 else "no"
        yesno_correct = float(prediction == kind)
        span_em = span_f1 = topk_hit = None
    else:
        prediction = span_prediction
        yesno_correct = None
        gold_spans = target_payload["answer_spans"]
        span_em = float((best_start, best_end) in set(map(tuple, gold_spans))) if gold_spans else 0.0
        span_f1 = token_f1_span((best_start, best_end), gold_spans) if gold_spans else 0.0
        topk_hit = float(any(
            (start, end) in set(map(tuple, gold_spans))
            for start, end, _ in spans
        )) if gold_spans else 0.0
    current_em, current_f1 = answer_scores(prediction, target_payload["row"]["answer"])
    source_em, source_f1 = answer_scores(prediction, source_payload["row"]["answer"])
    source_gold_spans = source_payload["answer_spans"]
    source_span_em = (
        float((best_start, best_end) in set(map(tuple, source_gold_spans)))
        if source_gold_spans else 0.0
    )
    return {
        "prediction": prediction,
        "predicted_span_text": span_prediction,
        "predicted_start": best_start,
        "predicted_end": best_end,
        "top5_spans": [
            {"start": start, "end": end, "score": score}
            for start, end, score in spans
        ],
        "current_answer_em": current_em,
        "current_answer_f1": current_f1,
        "source_answer_em": source_em,
        "source_answer_f1": source_f1,
        "span_em": span_em,
        "span_token_f1": span_f1,
        "span_top5_hit": topk_hit,
        "yesno_correct": yesno_correct,
        "support_current_available": current_available,
        "support_token_current": current_token,
        "support_sentence_current": current_sentence,
        "support_token_source": source_token,
        "support_sentence_source": source_sentence,
        "source_span_em": source_span_em,
        "yesno_logits": output["yesno"].float().cpu().tolist(),
    }


def mean_available(rows, path):
    values = []
    for row in rows:
        value = row
        for key in path:
            if value is None:
                break
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def summarize(records, condition):
    rows = [row for row in records if row["condition"] == condition]
    span_rows = [row for row in rows if row["target_answer_kind"] == "span"]
    yesno_rows = [row for row in rows if row["target_answer_kind"] in {"yes", "no"}]
    return {
        "n": len(rows),
        "current_answer_em": mean_available(rows, ["metrics", "current_answer_em"]),
        "current_answer_f1": mean_available(rows, ["metrics", "current_answer_f1"]),
        "source_answer_em": mean_available(rows, ["metrics", "source_answer_em"]),
        "source_answer_f1": mean_available(rows, ["metrics", "source_answer_f1"]),
        "support_token_f1": mean_available(rows, ["metrics", "support_token_current", "f1"]),
        "support_token_auprc": mean_available(rows, ["metrics", "support_token_current", "auprc"]),
        "support_sentence_recall_at_2": mean_available(rows, ["metrics", "support_sentence_current", "recall_at_2"]),
        "support_sentence_f1": mean_available(rows, ["metrics", "support_sentence_current", "f1"]),
        "source_support_token_f1": mean_available(rows, ["metrics", "support_token_source", "f1"]),
        "source_support_sentence_recall_at_2": mean_available(rows, ["metrics", "support_sentence_source", "recall_at_2"]),
        "span_samples": len(span_rows),
        "span_em": mean_available(span_rows, ["metrics", "span_em"]),
        "span_token_f1": mean_available(span_rows, ["metrics", "span_token_f1"]),
        "span_top5_hit": mean_available(span_rows, ["metrics", "span_top5_hit"]),
        "yesno_samples": len(yesno_rows),
        "yesno_accuracy": mean_available(yesno_rows, ["metrics", "yesno_correct"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-index", required=True)
    parser.add_argument("--native-index", required=True)
    parser.add_argument("--sidecar-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--text-probe", required=True)
    parser.add_argument("--native-probe", required=True)
    parser.add_argument("--canonical-probe", required=True)
    parser.add_argument("--zero-probe", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = ProbeCache(
        args.canonical_index, args.native_index,
        args.sidecar_index, args.data, capacity=3,
    )
    count = min(args.max_samples, len(cache))
    probes = {
        "full_text_representation": load_probe(args.text_probe, device)[0],
        "sender_native_kv": load_probe(args.native_probe, device)[0],
        "learned_canonical_kv": load_probe(args.canonical_probe, device)[0],
        "question_only_zero_memory": load_probe(args.zero_probe, device)[0],
    }
    lengths = [
        int(entry.get("shape", [0, cache.load(index)["mask"].numel()])[1])
        for index, entry in enumerate(cache.canonical_entries[:count])
    ]
    hard = build_hard_negatives(cache.rows[:count], lengths)
    sample_shuffle = [
        (index + 1 + args.seed % max(count - 1, 1)) % count
        for index in range(count)
    ]
    records = []
    with torch.inference_mode():
        for index in tqdm(range(count), desc="p3e_k_eval"):
            current = cache.load(index)
            for condition, probe_name, source_index in [
                ("full_text_representation", "full_text_representation", index),
                ("sender_native_kv", "sender_native_kv", index),
                ("learned_canonical_kv", "learned_canonical_kv", index),
                ("question_only_zero_memory", "question_only_zero_memory", index),
                ("canonical_sample_shuffled", "learned_canonical_kv", sample_shuffle[index]),
                ("canonical_hard_shuffled", "learned_canonical_kv", hard[index]),
            ]:
                source = cache.load(source_index)
                target = current if source_index == index else retarget_payload(current, source)
                result = probes[probe_name](target, device)
                metrics = evaluate_output(result, target, source)
                records.append({
                    "id": current["row"]["id"],
                    "source_id": source["row"]["id"],
                    "condition": condition,
                    "type": current["row"].get("type"),
                    "question": current["row"]["question"],
                    "current_answer": current["row"]["answer"],
                    "source_answer": source["row"]["answer"],
                    "target_answer_kind": target["answer_kind"],
                    "metrics": metrics,
                })
    write_jsonl(output / "per_sample_results.jsonl", records)
    conditions = sorted({row["condition"] for row in records})
    summaries = {condition: summarize(records, condition) for condition in conditions}
    canonical = summaries["learned_canonical_kv"]
    sample = summaries["canonical_sample_shuffled"]
    hard_summary = summaries["canonical_hard_shuffled"]
    result = {
        "status": "complete",
        "experiment": "P3-E-K Canonical Information Sufficiency Probe",
        "diagnostic_only_not_reader": True,
        "samples": count,
        "conditions": summaries,
        "canonical_correct_minus_sample_shuffled_current_answer_f1": (
            canonical["current_answer_f1"] - sample["current_answer_f1"]
        ),
        "canonical_correct_minus_hard_shuffled_current_answer_f1": (
            canonical["current_answer_f1"] - hard_summary["current_answer_f1"]
        ),
        "interpretation_rules": {
            "A": "Canonical near Native and Full-text: information sufficient; execution interface is bottleneck.",
            "B": "Native high, Canonical low: C2 Writer/Canonical interface loses critical content.",
            "C": "Native and Canonical low, Full-text high: selected KV representation is insufficient for this probe.",
            "D": "Supporting localization high, answer span low: facts exist but relational/multi-hop composition is not decoded.",
        },
        "probe_checkpoints": {
            "text": args.text_probe,
            "native": args.native_probe,
            "canonical": args.canonical_probe,
            "zero": args.zero_probe,
        },
    }
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
