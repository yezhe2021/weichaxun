import argparse
import json
import random
from collections import Counter
from pathlib import Path

from hotpot_common import write_json, write_jsonl


def supporting_evidence(row):
    context = {title: sentences for title, sentences in row["context"]}
    grouped = []
    for title, sentence_index in row["supporting_facts"]:
        if title not in context or sentence_index >= len(context[title]):
            continue
        if not grouped or grouped[-1][0] != title:
            existing = next((item for item in grouped if item[0] == title), None)
            if existing is None:
                existing = [title, []]; grouped.append(existing)
        else:
            existing = grouped[-1]
        sentence = context[title][sentence_index].strip()
        if sentence and sentence not in existing[1]:
            existing[1].append(sentence)
    rendered = [f"[{title}] " + " ".join(sentences) for title, sentences in grouped if sentences]
    if not rendered:
        raise RuntimeError(f"No supporting evidence for {row['_id']}")
    if len(rendered) == 1:
        return rendered[0], "[No additional supporting sentence.]"
    return rendered[0], "\n".join(rendered[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=64); parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    with open(args.raw, encoding="utf-8") as handle:
        source = json.load(handle)
    rng = random.Random(args.seed); rng.shuffle(source)
    quota = {"bridge": args.samples // 2, "comparison": args.samples - args.samples // 2}
    selected = []
    for row in source:
        kind = row.get("type", "unknown")
        if quota.get(kind, 0) <= 0:
            continue
        evidence_a, evidence_b = supporting_evidence(row)
        evidence = evidence_a + "\n" + evidence_b
        answer = str(row["answer"])
        answer_type = "yes_no" if answer.casefold() in {"yes", "no"} else ("extractive" if answer.casefold() in evidence.casefold() else "non_extractive")
        selected.append({
            "id": row["_id"], "question": row["question"], "answer": answer,
            "type": kind, "level": row.get("level"), "answer_type": answer_type,
            "evidence_a": evidence_a, "evidence_b": evidence_b,
            "supporting_facts": row["supporting_facts"],
        })
        quota[kind] -= 1
        if len(selected) == args.samples:
            break
    if len(selected) != args.samples:
        raise RuntimeError(f"Selected {len(selected)} of {args.samples} requested samples")
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "dev64.jsonl", selected)
    write_json(output / "SUCCESS.json", {
        "status": "complete", "samples": len(selected), "seed": args.seed,
        "type_counts": Counter(row["type"] for row in selected),
        "answer_type_counts": Counter(row["answer_type"] for row in selected),
        "source": str(Path(args.raw).resolve()),
    })


if __name__ == "__main__":
    main()
