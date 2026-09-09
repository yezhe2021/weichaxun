import argparse
from pathlib import Path

import torch

from p3d3_common import file_sha256, write_json
from p3e_m_common import FreshReaderMemory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-base", required=True)
    parser.add_argument("--train-conditioned", required=True)
    parser.add_argument("--validation-base", required=True)
    parser.add_argument("--validation-conditioned", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split_results = {}
    for split, base, conditioned in (
        ("train", args.train_base, args.train_conditioned),
        ("validation", args.validation_base, args.validation_conditioned),
    ):
        cache = FreshReaderMemory(base, conditioned)
        checks = []
        for index in range(len(cache)):
            a = cache.evidence_only(index)
            b = cache.question_conditioned(index)
            if a["row"]["id"] != b["row"]["id"]:
                raise RuntimeError("Sample IDs are not aligned")
            if a["keys"].shape != b["keys"].shape:
                raise RuntimeError("Native KV shapes differ between A and B")
            if a["keys"].shape[-2:] != (8, 128):
                raise RuntimeError("Native KV is not [16,T,8,128]")
            if a["metadata"]["token_ids"] != b["metadata"]["token_ids"]:
                raise RuntimeError("Evidence token IDs changed under Q conditioning")
            if b["metadata"]["prefix_token_count"] <= 0:
                raise RuntimeError("Q-conditioned Sender has no Question prefix")
            if b["metadata"]["question_tokens_transmitted"] != 0:
                raise RuntimeError("Question KV leaked into transmitted memory")
            checks.append(
                {
                    "id": a["row"]["id"],
                    "tokens": int(a["keys"].shape[1]),
                    "hard_index": cache.hard_index(index),
                }
            )
        split_results[split] = {
            "samples": len(cache),
            "all_ids_aligned": True,
            "all_evidence_token_ids_identical": True,
            "all_question_tokens_excluded": True,
            "shape": "[16,T,8,128]",
            "first": checks[0],
        }
    write_json(
        args.out,
        {
            "status": "complete",
            "experiment": "P3-E-M input audit",
            "splits": split_results,
            "train_base_sha256": file_sha256(args.train_base),
            "train_conditioned_sha256": file_sha256(args.train_conditioned),
            "validation_base_sha256": file_sha256(args.validation_base),
            "validation_conditioned_sha256": file_sha256(args.validation_conditioned),
            "writer_loaded": False,
            "canonical_memory_used": False,
        },
    )


if __name__ == "__main__":
    main()
