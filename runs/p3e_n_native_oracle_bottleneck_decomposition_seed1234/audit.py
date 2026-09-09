import argparse

from p3d3_common import file_sha256, read_json, write_json
from p3e_n_common import ReceiverNativeConditionedCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--reader-b-training", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    splits = {}
    for name, path, expected in (
        ("train", args.train, 512),
        ("validation", args.validation, 64),
    ):
        cache = ReceiverNativeConditionedCache(path)
        if len(cache) != expected:
            raise RuntimeError(f"{name} expected {expected} samples, got {len(cache)}")
        for index in range(len(cache)):
            correct = cache.correct(index)
            shuffled = cache.shuffled(index)
            for payload in (correct, shuffled):
                if payload["keys"].shape[-2:] != (8, 128):
                    raise RuntimeError("4B Native KV geometry mismatch")
                if payload["metadata"]["question_tokens_transmitted"] != 0:
                    raise RuntimeError("Question KV leaked into memory")
                if payload["metadata"]["native_model"] != "Qwen3-4B":
                    raise RuntimeError("Cache is not Qwen3-4B Native KV")
        splits[name] = {"samples": len(cache), "shape": "[16,T,8,128]"}

    reader_b = read_json(args.reader_b_training)
    if reader_b["epochs"] != 5 or reader_b["loss"] != "answer-token mean NLL only":
        raise RuntimeError("P3-E-M Reader B protocol does not match P3-E-N")
    write_json(
        args.out,
        {
            "status": "complete",
            "experiment": "P3-E-N protocol audit",
            "splits": splits,
            "shared_initialization": args.initialization,
            "shared_initialization_sha256": file_sha256(args.initialization),
            "reader_b_training_reference": reader_b,
            "same_epochs": 5,
            "same_loss": "answer-token mean NLL only",
            "writer_loaded": False,
            "canonical_projection_used": False,
        },
    )


if __name__ == "__main__":
    main()
