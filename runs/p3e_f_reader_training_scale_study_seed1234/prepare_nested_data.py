import argparse
import random
from pathlib import Path

from transformers import AutoTokenizer

from p3d3_common import aliases_overlap, evidence_block, normalize_answer
from p3e_f_common import read_json, read_jsonl, sha256_rows, write_json, write_jsonl
from prepare_p3d3_data import convert


def choose_block(pool, used, needed, seed, existing):
    rng = random.Random(seed)
    available = [row for row in pool if row["id"] not in used]
    groups = {kind: [row for row in available if row["type"] == kind] for kind in ("bridge", "comparison")}
    for group in groups.values():
        rng.shuffle(group)
    current = {kind: sum(row["type"] == kind for row in existing) for kind in groups}
    selected = []
    for _ in range(needed):
        kind = min(groups, key=lambda name: (current[name], name))
        if not groups[kind]:
            kind = next(name for name in groups if groups[name])
        row = groups[kind].pop()
        selected.append(row)
        current[kind] += 1
    rng.shuffle(selected)
    return selected


def negative_for(index, rows, lengths, candidates):
    row = rows[index]
    titles = {normalize_answer(title) for title in row.get("supporting_titles", [])}
    bridge = normalize_answer(row.get("bridge_entity", ""))
    ranked = []
    for other_index in candidates:
        if other_index == index:
            continue
        other = rows[other_index]
        if other["type"] != row["type"] or other["answer_type"] != row["answer_type"]:
            continue
        if aliases_overlap(row["answer"], other["answer"]):
            continue
        if titles & {normalize_answer(title) for title in other.get("supporting_titles", [])}:
            continue
        other_bridge = normalize_answer(other.get("bridge_entity", ""))
        if bridge and other_bridge and bridge == other_bridge:
            continue
        if normalize_answer(row["answer"]) in normalize_answer(evidence_block(other)):
            continue
        ranked.append((abs(lengths[index] - lengths[other_index]),
                       abs(len(str(row["answer"]).split()) - len(str(other["answer"]).split())),
                       other_index))
    if not ranked:
        raise RuntimeError(f"No leakage-safe hard negative for {row['id']}")
    return min(ranked)[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", required=True)
    parser.add_argument("--existing512", required=True)
    parser.add_argument("--validation64", required=True)
    parser.add_argument("--existing-cache-index", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-evidence-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    first512 = read_jsonl(args.existing512)
    validation = read_jsonl(args.validation64)
    excluded = {row["id"] for row in first512} | {row["id"] for row in validation}
    converted = [item for item in (convert(row) for row in read_json(args.raw_train)) if item is not None]
    usable = []
    token_lengths = {}
    for row in converted:
        if row["id"] in excluded:
            continue
        length = len(tokenizer(evidence_block(row), add_special_tokens=True, truncation=False)["input_ids"])
        if length <= args.max_evidence_tokens:
            token_lengths[row["id"]] = length
            usable.append(row)

    block1024 = choose_block(usable, excluded, 512, args.seed + 1024, first512)
    used1024 = excluded | {row["id"] for row in block1024}
    rows1024 = first512 + block1024
    block2048 = choose_block(usable, used1024, 1024, args.seed + 2048, rows1024)
    rows2048 = rows1024 + block2048
    if [row["id"] for row in rows2048[:512]] != [row["id"] for row in first512]:
        raise RuntimeError("The original train512 prefix changed")
    if [row["id"] for row in rows2048[:1024]] != [row["id"] for row in rows1024]:
        raise RuntimeError("The train1024 prefix changed")

    lengths = []
    for row in rows2048:
        lengths.append(token_lengths.get(row["id"]) or
                       len(tokenizer(evidence_block(row), add_special_tokens=True, truncation=False)["input_ids"]))
    old_index = read_json(args.existing_cache_index)
    old_negatives = [int(entry["hard_negative_index"]) for entry in old_index["entries"]]
    if len(old_negatives) != 512:
        raise RuntimeError("Expected exactly 512 historical hard negatives")
    negatives = old_negatives[:]
    for index in range(512, 1024):
        negatives.append(negative_for(index, rows2048, lengths, range(1024)))
    for index in range(1024, 2048):
        negatives.append(negative_for(index, rows2048, lengths, range(2048)))
    if any(index >= 512 for index in negatives[:512]):
        raise RuntimeError("Historical train512 negatives escaped the first prefix")
    if any(index >= 1024 for index in negatives[:1024]):
        raise RuntimeError("Train1024 negatives escaped its prefix")

    write_jsonl(output / "train1024.jsonl", rows1024)
    write_jsonl(output / "train2048.jsonl", rows2048)
    write_json(output / "hard_negatives.json", {"train512": negatives[:512], "train1024": negatives[:1024], "train2048": negatives})
    manifest = {
        "status": "complete", "seed": args.seed, "nested": True,
        "sizes": [512, 1024, 2048],
        "prefix_sha256": {"512": sha256_rows(first512), "1024": sha256_rows(rows1024), "2048": sha256_rows(rows2048)},
        "types": {str(size): {kind: sum(row["type"] == kind for row in rows2048[:size])
                              for kind in ("bridge", "comparison")} for size in (512, 1024, 2048)},
        "max_evidence_tokens": max(lengths), "validation_ids_excluded": True,
        "existing512": args.existing512, "fixed_validation64": args.validation64,
    }
    write_json(output / "SUCCESS.json", manifest)


if __name__ == "__main__":
    main()
