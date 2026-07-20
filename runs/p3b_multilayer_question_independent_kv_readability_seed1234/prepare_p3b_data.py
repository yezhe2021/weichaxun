import argparse
import random
from collections import Counter
from pathlib import Path

from p3b_common import all_occurrences, evidence_block, load_jsonl, write_json, write_jsonl


def partition(rows):
    extractive, excluded = [], []
    for row in rows:
        item = dict(row)
        evidence = evidence_block(item)
        ranges = all_occurrences(evidence, item["answer"])
        item["answer_char_spans"] = ranges
        if item.get("answer_type") == "extractive" and ranges:
            extractive.append(item)
        else:
            excluded.append(item)
    return extractive, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--validation-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    train_rows, train_excluded = partition(load_jsonl(args.train_source))
    test_rows, test_excluded = partition(load_jsonl(args.test_source))
    random.Random(args.seed).shuffle(train_rows)
    if len(train_rows) <= args.validation_size:
        raise RuntimeError("Not enough extractive training samples")
    validation = train_rows[: args.validation_size]
    train = train_rows[args.validation_size :]

    output = Path(args.out)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    write_jsonl(output / "test.jsonl", test_rows)
    write_jsonl(output / "excluded_train.jsonl", train_excluded)
    write_jsonl(output / "excluded_test.jsonl", test_excluded)
    summary = {
        "status": "complete",
        "seed": args.seed,
        "train": len(train),
        "validation": len(validation),
        "test": len(test_rows),
        "excluded_train": len(train_excluded),
        "excluded_test": len(test_excluded),
        "train_types": Counter(row["type"] for row in train),
        "validation_types": Counter(row["type"] for row in validation),
        "test_types": Counter(row["type"] for row in test_rows),
    }
    write_json(output / "SUCCESS.json", summary)


if __name__ == "__main__":
    main()
