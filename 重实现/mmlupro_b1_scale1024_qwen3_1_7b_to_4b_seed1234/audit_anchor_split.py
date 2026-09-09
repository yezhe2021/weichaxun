from __future__ import annotations

import argparse
from pathlib import Path

from common import load_json, read_jsonl, save_json


def identities(rows):
    return [(row["id"], row["prefix_token_sha256"]) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    current_root = Path(cfg["work_dir"]) / "artifacts/manifests"
    anchor_root = Path(cfg["data"]["anchor_manifest_dir"])
    current = {split: read_jsonl(current_root / f"{split}.jsonl") for split in ("train", "validation", "test")}
    anchor = {split: read_jsonl(anchor_root / f"{split}.jsonl") for split in ("train", "validation", "test")}
    checks = {
        "test_exact_identity_and_order": identities(current["test"]) == identities(anchor["test"]),
        "old_train_is_new_train_prefix": identities(current["train"][:len(anchor["train"])]) == identities(anchor["train"]),
        "old_validation_is_new_validation_prefix": identities(current["validation"][:len(anchor["validation"])]) == identities(anchor["validation"]),
        "counts_exact": {split: len(rows) for split, rows in current.items()} == cfg["data"]["splits"],
        "all_splits_disjoint": len({row["id"] for rows in current.values() for row in rows}) == sum(len(rows) for rows in current.values()),
    }
    save_json(current_root / "anchor_audit.json", {"passed": all(checks.values()), "checks": checks})
    if not all(checks.values()):
        raise RuntimeError(f"anchored scale-up audit failed: {checks}")
    print(f"anchored scale-up audit passed: {checks}", flush=True)


if __name__ == "__main__":
    main()
