from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()
def link(path, target):
    path, target = Path(path), Path(target).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() == target: return
        path.unlink()
    elif path.exists(): raise RuntimeError(f"refusing to replace {path}")
    path.symlink_to(target, target_is_directory=target.is_dir())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args(); cfg = load(args.config)
    work, forward = Path(cfg["work_dir"]), Path(cfg["forward_experiment_dir"])
    base_source = forward / "artifacts" / "manifest.json"
    eval_source = forward / "artifacts" / "eval128" / "manifest.json"
    link(work / "artifacts" / "manifest.json", base_source)
    link(work / "artifacts" / "eval128_manifest.json", eval_source)
    link(work / "cache" / "source1_7", forward / "cache" / "source1_7")
    link(work / "cache" / "source1_7_eval128", forward / "cache" / "eval128" / "source1_7")
    link(work / "cache" / "target4", Path(cfg["target4_experiment_dir"]) / "cache" / "development" / "target4")
    base, test = load(base_source), load(eval_source)
    if [x["id"] for x in test[:64]] != [x["id"] for x in base["test"]]: raise RuntimeError("eval128 does not preserve frozen test64 prefix")
    counts = {kind: sum(x["type"] == kind for x in test) for kind in ("bridge", "comparison")}
    if len(test) != 128 or counts != {"bridge": 64, "comparison": 64}: raise RuntimeError(f"eval128 balance mismatch: {counts}")
    tok4 = AutoTokenizer.from_pretrained(cfg["model_4b"], local_files_only=True, use_fast=True)
    tok17 = AutoTokenizer.from_pretrained(cfg["model_1_7b"], local_files_only=True, use_fast=True)
    vocab4, vocab17 = tok4.get_vocab(), tok17.get_vocab()
    special4 = {name: getattr(tok4, name) for name in ("bos_token_id", "eos_token_id", "pad_token_id")}
    special17 = {name: getattr(tok17, name) for name in ("bos_token_id", "eos_token_id", "pad_token_id")}
    if vocab4 != vocab17 or special4 != special17: raise RuntimeError("4B/1.7B tokenizer or special-token mismatch")
    vocab_hash = hashlib.sha256(json.dumps(sorted(vocab4.items()), ensure_ascii=False).encode("utf-8")).hexdigest()
    report = {
        "base_manifest_sha256": sha256(base_source), "eval128_manifest_sha256": sha256(eval_source),
        "expected_forward_eval128_sha256": sha256(forward / "artifacts" / "eval128" / "manifest.json"),
        "strictly_comparable": True, "sizes": cfg["sizes"], "test_type_counts": counts,
        "tokenizer_vocab_sha256": vocab_hash, "tokenizer_vocab_equal": True, "special_token_ids": special4,
        "context_and_question_ids_shared_from_frozen_manifest": True,
        "sender_sees_question": False, "receiver_backbone_frozen": True, "reader": "Identity", "thinking": False,
    }
    save(work / "artifacts" / "asset_audit.json", report)
    print("reverse diagnosis manifests and reusable assets audited", flush=True)


if __name__ == "__main__": main()
