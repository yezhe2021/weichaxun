import argparse
import random
import re
from pathlib import Path

from experiment import normalize_answer, read_json, seed_everything, write_json, write_jsonl


def format_evidence(row, scope):
    support = {}
    support_order = []
    for title, sentence_index in row.get("supporting_facts", []):
        if title not in support:
            support[title] = set()
            support_order.append(title)
        support[title].add(int(sentence_index))
    contexts = row.get("context", [])
    context_map = {title: sentences for title, sentences in contexts}
    if len(support_order) != 2 or any(title not in context_map for title in support_order):
        return None
    selected_titles = support_order if scope == "gold_docs" else [title for title, _ in contexts]
    chunks = []
    support_spans = []
    cursor = 0
    for document_index, title in enumerate(selected_titles):
        header = f"DOCUMENT {document_index + 1}\nTITLE: {title}\n"
        chunks.append(header)
        cursor += len(header)
        for sentence_index, sentence in enumerate(context_map[title]):
            sentence = sentence.strip()
            prefix = f"[{sentence_index}] "
            chunks.append(prefix)
            cursor += len(prefix)
            start = cursor
            chunks.append(sentence)
            cursor += len(sentence)
            if sentence_index in support.get(title, set()):
                support_spans.append([start, cursor])
            chunks.append("\n")
            cursor += 1
        if document_index + 1 < len(selected_titles):
            chunks.append("\n")
            cursor += 1
    return "".join(chunks).rstrip(), support_spans, support_order


def answer_occurrences(text, answer):
    if not answer or normalize_answer(answer) in {"yes", "no"}:
        return []
    return [[match.start(), match.end()] for match in re.finditer(re.escape(answer), text, flags=re.I)]


def bridge_entity(question_type, supporting_titles, evidence):
    if question_type != "bridge":
        return ""
    for title in supporting_titles:
        if normalize_answer(title) in normalize_answer(evidence):
            return title
    return ""


def convert(row, scope):
    formatted = format_evidence(row, scope)
    if formatted is None:
        return None
    evidence, support_spans, supporting_titles = formatted
    answer_spans = answer_occurrences(evidence, row["answer"])
    answer_normalized = normalize_answer(row["answer"])
    answer_type = "yes_no" if answer_normalized in {"yes", "no"} else "text"
    return {
        "id": row["_id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row.get("type", "unknown"),
        "level": row.get("level", "unknown"),
        "evidence": evidence,
        "evidence_scope": scope,
        "supporting_titles": supporting_titles,
        "supporting_facts": row.get("supporting_facts", []),
        "support_char_spans": support_spans,
        "answer_char_spans": answer_spans,
        "answer_type": answer_type,
        "bridge_entity": bridge_entity(row.get("type"), supporting_titles, evidence),
    }


def balanced_sample(rows, count, seed, scope):
    converted = [item for item in (convert(row, scope) for row in rows) if item is not None]
    rng = random.Random(seed)
    groups = {kind: [row for row in converted if row["type"] == kind] for kind in ("bridge", "comparison")}
    for values in groups.values():
        rng.shuffle(values)
    selected = groups["bridge"][: (count + 1) // 2] + groups["comparison"][: count // 2]
    if len(selected) < count:
        used = {row["id"] for row in selected}
        remainder = [row for row in converted if row["id"] not in used]
        rng.shuffle(remainder)
        selected.extend(remainder[: count - len(selected)])
    rng.shuffle(selected)
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} usable rows for requested count {count}")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--evidence-scope", choices=["gold_docs", "all_context"], default="gold_docs")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    train = balanced_sample(read_json(args.train), args.train_samples, args.seed, args.evidence_scope)
    validation = balanced_sample(read_json(args.validation), args.validation_samples, args.seed + 1, args.evidence_scope)
    if {row["id"] for row in train} & {row["id"] for row in validation}:
        raise RuntimeError("Official train and validation samples overlap")
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    write_json(output / "SUCCESS.json", {
        "status": "complete",
        "seed": args.seed,
        "evidence_scope": args.evidence_scope,
        "train": len(train),
        "validation": len(validation),
        "train_types": {kind: sum(row["type"] == kind for row in train) for kind in ("bridge", "comparison")},
        "validation_types": {kind: sum(row["type"] == kind for row in validation) for kind in ("bridge", "comparison")},
        "official_splits_disjoint": True,
    })


if __name__ == "__main__":
    main()
