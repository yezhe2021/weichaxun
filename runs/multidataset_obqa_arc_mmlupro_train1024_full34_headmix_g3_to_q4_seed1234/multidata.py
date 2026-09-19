import json
import random
import re
import zipfile
from pathlib import Path

import pyarrow.parquet as parquet

LABELS = "ABCDEFGHIJ"


def _openbook_rows(cfg, split):
    path = Path(cfg["datasets"]["openbookqa"][split])
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for item in items:
        choices = sorted(item["question"]["choices"], key=lambda value: value["label"])
        labels = [value["label"] for value in choices]
        output.append({"id": f"obqa_{item['id']}", "dataset": "openbookqa",
                       "category": "openbookqa_official_main", "question": item["question"]["stem"],
                       "options": [value["text"] for value in choices],
                       "gold_index": labels.index(item["answerKey"])})
    random.Random(cfg["seed"] + {"train": 101, "validation": 102, "test": 103}[split]).shuffle(output)
    return output


def _arc_rows(cfg, split):
    names = {"train": "Train", "validation": "Dev", "test": "Test"}
    member = f"ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-{names[split]}.jsonl"
    with zipfile.ZipFile(cfg["datasets"]["arc_challenge_zip"]) as archive:
        items = [json.loads(line) for line in archive.read(member).decode("utf-8").splitlines() if line.strip()]
    output = []
    for item in items:
        choices = item["question"]["choices"]
        labels = [str(value["label"]) for value in choices]
        answer = str(item["answerKey"])
        if len(choices) == 4 and answer in labels:
            output.append({"id": f"arc_{split}_{item['id']}", "dataset": "arc_challenge",
                           "category": "arc_challenge", "question": item["question"]["stem"],
                           "options": [value["text"] for value in choices], "gold_index": labels.index(answer)})
    random.Random(cfg["seed"] + {"train": 201, "validation": 202, "test": 203}[split]).shuffle(output)
    return output


def _mmlu_rows(cfg, split):
    items = parquet.read_table(cfg["datasets"]["mmlu_pro_test"]).to_pylist()
    groups = {}
    for item in items:
        groups.setdefault(str(item["category"]), []).append(item)
    rng = random.Random(cfg["seed"] + 301)
    for values in groups.values():
        rng.shuffle(values)
    ordered = []
    while any(groups.values()):
        for name in sorted(groups):
            if groups[name]:
                ordered.append(groups[name].pop())
    begin, end = {"train": (0, 2048), "validation": (4096, 5120), "test": (8192, 9216)}[split]
    return [{"id": f"mmlupro_{split}_{item['question_id']}", "dataset": "mmlu_pro",
             "category": str(item["category"]), "question": item["question"],
             "options": list(item["options"]), "gold_index": int(item["answer_index"])}
            for item in ordered[begin:end]]


def official_rows(cfg, split):
    chosen, seen = {}, set()
    for current in ("train", "validation", "test"):
        rows = []
        pools = ((_openbook_rows(cfg, current), "openbookqa"),
                 (_arc_rows(cfg, current), "arc_challenge"),
                 (_mmlu_rows(cfg, current), "mmlu_pro"))
        for values, name in pools:
            accepted = []
            for row in values:
                normalized = re.sub(r"\W+", " ", row["question"].lower()).strip()
                if normalized in seen:
                    continue
                seen.add(normalized)
                accepted.append(row)
                if len(accepted) == cfg["per_dataset_samples"][current]:
                    break
            if len(accepted) != cfg["per_dataset_samples"][current]:
                raise RuntimeError(f"Insufficient leak-free {name} {current}: {len(accepted)}")
            rows.extend(accepted)
        random.Random(cfg["seed"] + {"train": 401, "validation": 402, "test": 403}[current]).shuffle(rows)
        chosen[current] = rows
    rows = chosen[split]
    if len(rows) != cfg[f"{split}_samples"]:
        raise RuntimeError(f"Balanced split count mismatch: {split} {len(rows)}")
    return rows

