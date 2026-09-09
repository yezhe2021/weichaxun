import argparse
from pathlib import Path

from p3d3_common import hard_negative_mapping
from p3e_b_common import SenderNativeHeadwiseCache
from p3e_d_common import CONDITIONS, git_commit, sha256, supporting_text, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", required=True)
    parser.add_argument("--writer", required=True)
    parser.add_argument("--canonical-reader", required=True)
    parser.add_argument("--native-reader", required=True)
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cache = SenderNativeHeadwiseCache(args.memory, capacity=args.max_samples)
    if len(cache) < args.max_samples:
        raise RuntimeError(f"Requested {args.max_samples} validation samples, cache only has {len(cache)}")
    negatives = hard_negative_mapping(cache)
    samples = []
    seen = set()
    for index in range(args.max_samples):
        payload = cache.load(index)
        row = payload["row"]
        if row["id"] in seen:
            raise RuntimeError(f"Duplicate validation id: {row['id']}")
        seen.add(row["id"])
        source = cache.load(negatives[index])["row"]
        support = supporting_text(row)
        samples.append({
            "index": index,
            "id": row["id"],
            "type": row["type"],
            "answer": row["answer"],
            "hard_negative_index": negatives[index],
            "hard_negative_id": source["id"],
            "hard_negative_answer": source["answer"],
            "native_tokens": int(payload["keys"].shape[1]),
            "supporting_text_utf8_bytes": len(support.encode("utf-8")),
        })
    type_counts = {kind: sum(sample["type"] == kind for sample in samples) for kind in ("bridge", "comparison")}
    root = Path(args.out)
    manifest = {
        "status": "prepared",
        "experiment": "P3-E-D Current-System Performance Check",
        "evaluation_only": True,
        "paper_conclusion": False,
        "seed": args.seed,
        "conditions": CONDITIONS,
        "samples": samples,
        "sample_count": len(samples),
        "type_counts": type_counts,
        "assets": {
            "memory": args.memory,
            "sender_model": args.sender_model,
            "receiver_model": args.receiver_model,
            "writer": {"path": args.writer, "sha256": sha256(args.writer)},
            "canonical_reader": {"path": args.canonical_reader, "sha256": sha256(args.canonical_reader)},
            "native_reader": {"path": args.native_reader, "sha256": sha256(args.native_reader)},
        },
        "protocol": {
            "full_evidence": "exact Evidence A+B complete documents used by the Sender",
            "supporting_text": "official supporting sentences selected by support_char_spans",
            "summary_max_new_tokens": 512,
            "receiver_max_new_tokens": 32,
            "receiver_input_truncation": False,
            "decoding": "greedy",
        },
        "git_commit": git_commit(Path(args.out).parents[1]),
    }
    write_json(root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
