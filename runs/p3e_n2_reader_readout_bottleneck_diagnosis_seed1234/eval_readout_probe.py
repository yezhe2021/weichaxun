import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import write_json, write_jsonl
from p3e_n2_common import (
    ReadoutCache,
    ReadoutContentProbe,
    payload_to_device,
)


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
    length = len(valid)
    best = (-1e30, 0, 0)
    for start in range(length):
        if not valid[start]:
            continue
        for end in range(start, min(length, start + max_answer_length)):
            if valid[end]:
                score = float(start_logits[start] + end_logits[end])
                if score > best[0]:
                    best = (score, start, end)
    return best[1], best[2]


def evaluate_labels(outputs, labels):
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


def aggregate(rows, prefix):
    tp = sum(row[prefix]["support_tp"] for row in rows)
    fp = sum(row[prefix]["support_fp"] for row in rows)
    fn = sum(row[prefix]["support_fn"] for row in rows)
    support_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    span_rows = [row for row in rows if row[prefix]["gold_spans"]]
    yesno_rows = [
        row for row in rows if row[prefix]["yesno_correct"] is not None
    ]
    return {
        "n": len(rows),
        "support_token_f1": support_f1,
        "span_n": len(span_rows),
        "span_em": sum(row[prefix]["span_em"] for row in span_rows)
        / max(1, len(span_rows)),
        "span_token_f1": sum(row[prefix]["span_token_f1"] for row in span_rows)
        / max(1, len(span_rows)),
        "yesno_n": len(yesno_rows),
        "yesno_accuracy": sum(
            row[prefix]["yesno_correct"] for row in yesno_rows
        )
        / max(1, len(yesno_rows)),
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
    count = min(args.max_samples, len(cache))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    probe = ReadoutContentProbe().to(device)
    probe.load_state_dict(checkpoint["probe"])
    probe.eval()
    rows = []
    with torch.inference_mode():
        for index in tqdm(range(count), desc="p3e_n2_probe_eval"):
            payload = cache.load(index)
            row = {"id": payload["row"]["id"]}
            correct_r, correct_a, correct_labels = payload_to_device(
                payload, "correct", device
            )
            shuffled_r, shuffled_a, shuffled_labels = payload_to_device(
                payload, "shuffled", device
            )
            zero_r, zero_a, zero_labels = payload_to_device(
                payload, "correct", device, zero=True
            )
            correct_outputs = probe(
                correct_r, correct_a, correct_labels["valid_mask"]
            )
            shuffled_outputs = probe(
                shuffled_r, shuffled_a, shuffled_labels["valid_mask"]
            )
            zero_outputs = probe(zero_r, zero_a, zero_labels["valid_mask"])
            row["correct_current"] = evaluate_labels(
                correct_outputs, correct_labels
            )
            row["shuffled_source"] = evaluate_labels(
                shuffled_outputs, shuffled_labels
            )
            row["zero_current"] = evaluate_labels(zero_outputs, zero_labels)
            rows.append(row)
    write_jsonl(output / "per_sample_predictions.jsonl", rows)
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-N2 Reader readout content diagnosis",
            "samples": count,
            "correct_current": aggregate(rows, "correct_current"),
            "shuffled_source": aggregate(rows, "shuffled_source"),
            "zero_current": aggregate(rows, "zero_current"),
            "probe_checkpoint": args.checkpoint,
            "input_excludes_native_kv": True,
            "input_contains_only_reader_readout_and_attention": True,
        },
    )


if __name__ == "__main__":
    main()
