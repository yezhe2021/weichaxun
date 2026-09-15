from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def encode_context(raw, tokenizer):
    ids, offsets, cursor, supporting, selected = [], [], 0, [], []
    gold = {(str(title), int(index)) for title, index in raw["supporting_facts"]}

    def add(text, is_support=False):
        nonlocal cursor
        encoded = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        begin = len(ids)
        ids.extend(encoded.input_ids)
        offsets.extend([(cursor + a, cursor + b) for a, b in encoded.offset_mapping])
        if is_support:
            selected.extend(range(begin, len(ids)))
            supporting.append([cursor, cursor + len(text)])
        cursor += len(text)

    for title, sentences in raw["context"]:
        add(f"Document: {title}\n")
        for index, sentence in enumerate(sentences):
            add(f"Sentence {index}: ")
            add(sentence, (str(title), index) in gold)
            add("\n")
    return ids, offsets, selected, supporting


def alignment(target_spans, source_spans):
    coverage_counts, used_counts, rows, uncovered = [], {}, [], 0
    overlap_total, target_total = 0, 0
    for ta, tb in target_spans:
        values = {}
        for index, (sa, sb) in enumerate(source_spans):
            overlap = max(0, min(tb, sb) - max(ta, sa))
            if overlap:
                values[index] = overlap
                used_counts[index] = used_counts.get(index, 0) + 1
        total = sum(values.values())
        rows.append({str(k): v / total for k, v in values.items()} if total else {})
        coverage_counts.append(len(values))
        uncovered += int(total == 0)
        overlap_total += min(total, max(tb - ta, 0))
        target_total += max(tb - ta, 0)
    used = len(used_counts)
    return rows, {
        "coverage_counts": coverage_counts,
        "uncovered_target_tokens": uncovered,
        "one_to_many_ratio": sum(x > 1 for x in coverage_counts) / max(len(rows), 1),
        "many_to_one_ratio": sum(x > 1 for x in used_counts.values()) / max(used, 1),
        "character_coverage": overlap_total / max(target_total, 1),
    }


def build(cfg, mode, rows):
    tok4 = AutoTokenizer.from_pretrained(
        cfg["model_4b"], local_files_only=True, use_fast=True
    )
    tok35 = AutoTokenizer.from_pretrained(
        cfg["model_q35"], local_files_only=True, use_fast=True
    )
    train = {x["_id"]: x for x in load(cfg["hotpot_train"])}
    dev = {x["_id"]: x for x in load(cfg["hotpot_dev"])}
    audit, metadata = [], []
    for split, samples in rows.items():
        raw_map = train if split == "train" else dev
        for sample in samples:
            raw = raw_map[sample["id"]]
            ids4, offsets4, selected4, supporting = encode_context(raw, tok4)
            ids35, offsets35, _, _ = encode_context(raw, tok35)
            limit = cfg["max_evidence_tokens"]
            targets = selected4[:limit]
            rows_a, metrics = alignment([offsets4[i] for i in targets], offsets35)
            audit.append({
                "id": sample["id"], "split": split,
                "q35_context_tokens": len(ids35), "q4_context_tokens": len(ids4),
                "q35_selected_tokens": len({int(k) for row in rows_a for k in row}),
                "q4_selected_tokens": len(targets),
                "coverage_count_mean": sum(metrics["coverage_counts"]) / max(len(targets), 1),
                **{k: v for k, v in metrics.items() if k != "coverage_counts"},
                "truncated_tokens": max(0, len(selected4) - limit),
            })
            metadata.append({
                "id": sample["id"], "split": split,
                "supporting_sentence_char_spans": supporting,
                "q35_token_offsets": offsets35, "q4_token_offsets": offsets4,
                "q35_selected_token_indices": sorted(
                    {int(k) for row in rows_a for k in row}
                ),
                "q4_selected_token_indices": targets,
                "q4_target_position_ids": sample["selected_position_ids"],
                "alignment_rows": rows_a,
                "valid_target_mask": [bool(row) for row in rows_a],
            })
    root = Path(cfg["work_dir"]) / "artifacts" / mode
    save(root / "alignment_audit.json", audit)
    save(root / "alignment_metadata.json", metadata)
