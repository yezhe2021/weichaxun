import argparse
import json
import random
from collections import Counter
from pathlib import Path

from p3a_common import write_json, write_jsonl


def evidence(row):
    context = {title: sentences for title, sentences in row["context"]}; grouped = {}
    order = []
    for title, index in row["supporting_facts"]:
        if title not in context or index >= len(context[title]): continue
        if title not in grouped: grouped[title] = []; order.append(title)
        sentence = context[title][index].strip()
        if sentence and sentence not in grouped[title]: grouped[title].append(sentence)
    rendered = [f"[{title}] " + " ".join(grouped[title]) for title in order if grouped[title]]
    if not rendered: raise RuntimeError(f"No support for {row['_id']}")
    return rendered[0], "\n".join(rendered[1:]) if len(rendered) > 1 else "[No additional supporting sentence.]"


def select(source, count, seed):
    random.Random(seed).shuffle(source); quota = {"bridge": count // 2, "comparison": count - count // 2}; rows = []
    for item in source:
        kind = item.get("type", "unknown")
        if quota.get(kind, 0) <= 0: continue
        a, b = evidence(item); answer = str(item["answer"]); joined = (a + "\n" + b).casefold()
        answer_type = "yes_no" if answer.casefold() in {"yes", "no"} else ("extractive" if answer.casefold() in joined else "non_extractive")
        rows.append({"id": item["_id"], "question": item["question"], "answer": answer, "type": kind, "level": item.get("level"), "answer_type": answer_type, "evidence_a": a, "evidence_b": b, "supporting_facts": item["supporting_facts"]})
        quota[kind] -= 1
        if len(rows) == count: break
    if len(rows) != count: raise RuntimeError(f"Selected {len(rows)} of {count}")
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--train-raw", required=True); parser.add_argument("--dev-raw", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--train-samples", type=int, default=512); parser.add_argument("--dev-samples", type=int, default=500); parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    with open(args.train_raw, encoding="utf-8") as handle: train = select(json.load(handle), args.train_samples, args.seed)
    with open(args.dev_raw, encoding="utf-8") as handle: dev = select(json.load(handle), args.dev_samples, args.seed + 1)
    write_jsonl(output / "train512.jsonl", train); write_jsonl(output / "dev500.jsonl", dev)
    write_json(output / "SUCCESS.json", {"status": "complete", "train": len(train), "dev": len(dev), "train_types": Counter(r["type"] for r in train), "dev_types": Counter(r["type"] for r in dev), "dev_answer_types": Counter(r["answer_type"] for r in dev)})


if __name__ == "__main__": main()
