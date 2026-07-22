import argparse
import random
import re
from pathlib import Path

from p3d3_common import evidence_block, normalize_answer, read_json, seed_everything, write_json, write_jsonl


def format_document(title, sentences, support_indices):
    text = f"TITLE: {title}\n"
    spans = []
    for index, sentence in enumerate(sentences):
        prefix = f"[{index}] "
        start = len(text) + len(prefix)
        text += prefix + sentence.strip() + "\n"
        if index in support_indices:
            spans.append([start, start + len(sentence.strip())])
    return text.rstrip(), spans


def all_occurrences(text, answer):
    if not answer or normalize_answer(answer) in {"yes", "no"}:
        return []
    return [[match.start(), match.end()] for match in re.finditer(re.escape(answer), text, flags=re.I)]


def bridge_entity(question_type, titles, documents):
    if question_type != "bridge" or len(titles) != 2:
        return ""
    left, right = titles
    if normalize_answer(left) in normalize_answer(documents[right]):
        return left
    if normalize_answer(right) in normalize_answer(documents[left]):
        return right
    return left


def convert(row):
    support = {}
    ordered_titles = []
    for title, sentence_index in row.get("supporting_facts", []):
        if title not in support:
            support[title] = set(); ordered_titles.append(title)
        support[title].add(int(sentence_index))
    contexts = {title: sentences for title, sentences in row.get("context", [])}
    if len(ordered_titles) != 2 or any(title not in contexts for title in ordered_titles):
        return None
    first, second = ordered_titles
    evidence_a, spans_a = format_document(first, contexts[first], support[first])
    evidence_b, spans_b = format_document(second, contexts[second], support[second])
    prefix_a = len("EVIDENCE A\n")
    prefix_b = prefix_a + len(evidence_a) + len("\n\nEVIDENCE B\n")
    support_spans = [[start + prefix_a, end + prefix_a] for start, end in spans_a]
    support_spans += [[start + prefix_b, end + prefix_b] for start, end in spans_b]
    converted = {
        "id": row["_id"], "question": row["question"], "answer": row["answer"],
        "type": row.get("type", "unknown"), "level": row.get("level", "unknown"),
        "evidence_a": evidence_a, "evidence_b": evidence_b,
        "supporting_titles": ordered_titles,
        "supporting_facts": row.get("supporting_facts", []),
        "support_char_spans": support_spans,
        "bridge_entity": bridge_entity(row.get("type"), ordered_titles, {first: evidence_a, second: evidence_b}),
    }
    answer_spans = all_occurrences(evidence_block(converted), row["answer"])
    converted["answer_char_spans"] = answer_spans
    converted["answer_type"] = "yes_no" if normalize_answer(row["answer"]) in {"yes", "no"} else ("extractive" if answer_spans else "open")
    return converted


def balanced_sample(rows, count, seed):
    rng = random.Random(seed)
    converted = [item for item in (convert(row) for row in rows) if item is not None]
    groups = {kind: [row for row in converted if row["type"] == kind] for kind in ("bridge", "comparison")}
    for values in groups.values(): rng.shuffle(values)
    target_bridge = min(len(groups["bridge"]), (count + 1) // 2)
    target_comparison = min(len(groups["comparison"]), count // 2)
    selected = groups["bridge"][:target_bridge] + groups["comparison"][:target_comparison]
    if len(selected) < count:
        used = {row["id"] for row in selected}
        remainder = [row for row in converted if row["id"] not in used]
        rng.shuffle(remainder); selected.extend(remainder[:count - len(selected)])
    rng.shuffle(selected)
    if len(selected) != count: raise RuntimeError(f"Only {len(selected)} usable rows for requested {count}")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True); parser.add_argument("--validation", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--train-samples", type=int, default=64); parser.add_argument("--validation-samples", type=int, default=64); parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(); seed_everything(args.seed); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    train = balanced_sample(read_json(args.train), args.train_samples, args.seed)
    validation = balanced_sample(read_json(args.validation), args.validation_samples, args.seed + 1)
    if {row["id"] for row in train} & {row["id"] for row in validation}: raise RuntimeError("Train/validation IDs overlap")
    write_jsonl(output / "train.jsonl", train); write_jsonl(output / "validation.jsonl", validation)
    result = {"status": "complete", "seed": args.seed, "train": len(train), "validation": len(validation), "splits_are_official_and_disjoint": True,
              "train_types": {kind: sum(row["type"] == kind for row in train) for kind in ("bridge", "comparison")},
              "validation_types": {kind: sum(row["type"] == kind for row in validation) for kind in ("bridge", "comparison")}}
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__": main()
