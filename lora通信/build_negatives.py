import argparse
from pathlib import Path

from experiment import normalize_answer, read_json, write_json


def aliases_overlap(left, right):
    left, right = normalize_answer(left), normalize_answer(right)
    if not left or not right:
        return False
    return left == right or (min(len(left), len(right)) >= 4 and (left in right or right in left))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-answer-type-fallback", action="store_true")
    args = parser.parse_args()
    index = read_json(args.memory)
    entries = index["entries"]
    mapping, fallback_ids = [], []
    for current_index, current in enumerate(entries):
        current_titles = {normalize_answer(title) for title in current["supporting_titles"]}
        def collect(require_answer_type):
            candidates = []
            for candidate_index, candidate in enumerate(entries):
                if candidate_index == current_index or candidate["type"] != current["type"]:
                    continue
                if require_answer_type and candidate["answer_type"] != current["answer_type"]:
                    continue
                if aliases_overlap(current["answer"], candidate["answer"]):
                    continue
                if current["answer_type"] != "yes_no" and normalize_answer(current["answer"]) in candidate.get("evidence_normalized", ""):
                    continue
                candidate_titles = {normalize_answer(title) for title in candidate["supporting_titles"]}
                if current_titles & candidate_titles:
                    continue
                if current["bridge_entity"] and normalize_answer(current["bridge_entity"]) == normalize_answer(candidate["bridge_entity"]):
                    continue
                candidates.append((
                    abs(int(current["tokens"]) - int(candidate["tokens"])),
                    abs(len(str(current["answer"]).split()) - len(str(candidate["answer"]).split())),
                    candidate["id"],
                    candidate_index,
                ))
            return candidates

        candidates = collect(require_answer_type=True)
        if not candidates and args.allow_answer_type_fallback:
            candidates = collect(require_answer_type=False)
            if candidates:
                fallback_ids.append(current["id"])
        if not candidates:
            raise RuntimeError(f"No strict hard negative for sample {current['id']}")
        _, _, _, selected = min(candidates)
        mapping.append(selected)
    result = {
        "status": "complete",
        "memory_index": str(Path(args.memory).resolve()),
        "samples": len(entries),
        "mapping": mapping,
        "strict_mapping": not fallback_ids,
        "answer_type_fallback_count": len(fallback_ids),
        "answer_type_fallback_ids": fallback_ids,
        "constraints": [
            "same_question_type",
            "same_answer_type",
            "different_answer_alias",
            "no_supporting_title_overlap",
            "different_bridge_entity",
            "candidate_evidence_excludes_current_text_answer",
            "minimum_memory_token_length_gap",
        ],
    }
    write_json(args.out, result)


if __name__ == "__main__":
    main()
