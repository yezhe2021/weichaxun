import argparse
import statistics
from pathlib import Path

from p3d_common import file_sha256, read_json, write_json


def select_stable(p3c, layer_config):
    candidates = []
    for seed in (1234, 2345, 3456):
        branch = p3c / "branches" / layer_config / f"seed{seed}"
        result = read_json(branch / "fresh_probe/SUCCESS.json")
        candidates.append({"seed": seed, "retention": result["retention"], "gap": result["causal_gaps"]["correct_minus_zero_f1"], "branch": branch})
    median = statistics.median(item["retention"] for item in candidates)
    chosen = min(candidates, key=lambda item: (abs(item["retention"] - median), -item["gap"], item["seed"]))
    branch, writer = chosen["branch"], chosen["branch"] / "writer/writer_best.pt"
    cache = {split: str((branch / "cache" / split / "index.json").resolve()) for split in ("train", "validation", "test")}
    index = read_json(cache["train"])
    return {
        "layer_config": layer_config, "seed": chosen["seed"], "selection_rule": "retention_closest_to_three_seed_median",
        "median_retention": median, "selected_retention": chosen["retention"], "writer_checkpoint": str(writer.resolve()),
        "writer_sha256": file_sha256(writer), "canonical_cache": cache,
        "groups": index["layers"], "memory_dim": index["canonical_dim"], "original_layer_indices": index["original_layer_indices"],
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--p3b", required=True); parser.add_argument("--p3c", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); p3b, p3c = Path(args.p3b), Path(args.p3c)
    uniform16 = select_stable(p3c, "uniform16"); all36 = select_stable(p3c, "all36")
    native_cache = {split: str((p3b / "cache" / split / "index.json").resolve()) for split in ("train", "validation", "test")}
    protocol = {
        "status": "complete", "main_protocol": "canonical16", "question_independent": True,
        "canonical16": uniform16, "canonical36": all36,
        "native16": {"groups": 16, "memory_dim": 1024, "original_layer_indices": uniform16["original_layer_indices"], "cache": native_cache},
        "writer_frozen": True, "sender_frozen": True, "receiver_backbone_frozen": True,
    }
    write_json(args.out, protocol)


if __name__ == "__main__": main()
