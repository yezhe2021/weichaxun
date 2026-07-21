import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from p3d_common import answer_scores, read_json, write_json, write_jsonl


def target_rows(index_path):
    path = Path(index_path); index = read_json(path); rows = {}
    for entry in index["entries"]:
        payload = torch.load(path.parent / entry["file"], map_location="cpu", weights_only=False)
        row = payload["row"]; rows[str(row.get("id"))] = row
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--old", required=True); parser.add_argument("--cache", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); wanted = target_rows(args.cache); records = []
    with open(args.old, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); sample_id = str(row.get("id", row.get("sample_id")))
            if sample_id not in wanted or row.get("condition") != "correct": continue
            prediction = row.get("prediction", ""); exact, f1 = answer_scores(prediction, wanted[sample_id]["answer"])
            records.append({**row, "sample_id": sample_id, "exact_match": exact, "f1": f1, "question_type": wanted[sample_id].get("type", "unknown")})
    grouped = defaultdict(list)
    for row in records: grouped[row["question_type"]].append(row)
    metrics = {"n": len(records), "exact_match": sum(row["exact_match"] for row in records) / max(1, len(records)), "f1": sum(row["f1"] for row in records) / max(1, len(records))}
    for kind in ("bridge", "comparison"):
        if grouped[kind]: metrics[kind] = {"n": len(grouped[kind]), "exact_match": sum(row["exact_match"] for row in grouped[kind]) / len(grouped[kind]), "f1": sum(row["f1"] for row in grouped[kind]) / len(grouped[kind])}
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); write_jsonl(output / "per_sample_generation.jsonl", records)
    write_json(output / "SUCCESS.json", {"status": "complete" if records else "unavailable", "metrics": metrics, "source": args.old})


if __name__ == "__main__": main()
