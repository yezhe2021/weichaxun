import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import write_json, write_jsonl
from p3e_n2_common import ReadoutCache, payload_to_device
from p3e_n3_common import load_probe


def binary_counts(prediction, target, valid):
    prediction = prediction[valid].bool()
    target = target[valid].bool()
    return (
        int((prediction & target).sum()),
        int((prediction & ~target).sum()),
        int((~prediction & target).sum()),
    )


def token_f1(predicted_span, gold_spans):
    if not gold_spans:
        return 0.0
    predicted = set(range(predicted_span[0], predicted_span[1] + 1))
    best = 0.0
    for start, end in gold_spans:
        gold = set(range(start, end + 1))
        overlap = len(predicted & gold)
        if overlap:
            precision = overlap / len(predicted)
            recall = overlap / len(gold)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def best_span(start_logits, end_logits, valid, max_answer_length=20):
    best = (-1e30, 0, 0)
    for start in range(len(valid)):
        if not valid[start]:
            continue
        for end in range(start, min(len(valid), start + max_answer_length)):
            if valid[end]:
                score = float(start_logits[start] + end_logits[end])
                if score > best[0]:
                    best = (score, start, end)
    return best[1], best[2]


def score(outputs, labels):
    valid = labels["valid_mask"].detach().cpu()
    support = labels["support_token_mask"].detach().cpu()
    support_prediction = (
        outputs["support_logits"].detach().cpu().sigmoid() >= 0.5
    )
    tp, fp, fn = binary_counts(support_prediction, support, valid)
    spans = labels["answer_spans"]
    predicted_span = best_span(
        outputs["start_logits"].detach().cpu(),
        outputs["end_logits"].detach().cpu(),
        valid.tolist(),
    )
    answer = str(labels["source_answer"]).strip().lower()
    yesno_correct = None
    if answer in ("yes", "no"):
        prediction = int(outputs["yesno_logits"].argmax(-1).item())
        yesno_correct = float(prediction == (0 if answer == "yes" else 1))
    return {
        "support_tp": tp,
        "support_fp": fp,
        "support_fn": fn,
        "predicted_span": list(predicted_span),
        "gold_spans": [list(span) for span in spans],
        "span_em": float(predicted_span in spans),
        "span_token_f1": token_f1(predicted_span, spans),
        "yesno_correct": yesno_correct,
    }


def aggregate(rows, condition):
    tp = sum(row[condition]["support_tp"] for row in rows)
    fp = sum(row[condition]["support_fp"] for row in rows)
    fn = sum(row[condition]["support_fn"] for row in rows)
    span_rows = [row for row in rows if row[condition]["gold_spans"]]
    yesno_rows = [
        row for row in rows
        if row[condition]["yesno_correct"] is not None
    ]
    return {
        "n": len(rows),
        "support_token_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "span_n": len(span_rows),
        "span_em": sum(row[condition]["span_em"] for row in span_rows)
        / max(1, len(span_rows)),
        "span_token_f1": sum(
            row[condition]["span_token_f1"] for row in span_rows
        ) / max(1, len(span_rows)),
        "yesno_n": len(yesno_rows),
        "yesno_accuracy": sum(
            row[condition]["yesno_correct"] for row in yesno_rows
        ) / max(1, len(yesno_rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = ReadoutCache(args.cache)
    probe, checkpoint = load_probe(args.checkpoint, device)
    probe.eval()
    count = min(args.max_samples, len(cache))
    rows = []
    with torch.inference_mode():
        for index in tqdm(
            range(count), desc=f"p3e_n3_eval_{checkpoint['mode']}"
        ):
            payload = cache.load(index)
            row = {"id": payload["row"]["id"]}
            for result_name, cache_name, zero in (
                ("correct_current", "correct", False),
                ("shuffled_source", "shuffled", False),
                ("zero_current", "correct", True),
            ):
                readout, attention, labels = payload_to_device(
                    payload, cache_name, device, zero=zero
                )
                outputs = probe(readout, attention, labels["valid_mask"])
                row[result_name] = score(outputs, labels)
            rows.append(row)

    write_jsonl(output / "per_sample_predictions.jsonl", rows)
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-N3 Attention-vs-Readout Sufficiency Ablation",
            "mode": checkpoint["mode"],
            "samples": count,
            "correct_current": aggregate(rows, "correct_current"),
            "shuffled_source": aggregate(rows, "shuffled_source"),
            "zero_current": aggregate(rows, "zero_current"),
            "checkpoint": args.checkpoint,
        },
    )


if __name__ == "__main__":
    main()

