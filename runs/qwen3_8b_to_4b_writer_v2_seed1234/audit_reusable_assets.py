from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_hash(root):
    root = Path(root); digest = hashlib.sha256(); files = []
    for pattern in ("tokenizer*.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        files.extend(root.glob(pattern))
    for path in sorted(set(files), key=lambda item: item.name):
        digest.update(path.name.encode()); digest.update(bytes.fromhex(sha256(path)))
    if not files: raise RuntimeError(f"no tokenizer files under {root}")
    return {"sha256": digest.hexdigest(), "files": sorted(path.name for path in set(files))}


def tensor_load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def ensure_link(link, target):
    link, target = Path(link), Path(target).resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target: return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def validate_cache(cfg, old, manifest):
    expected = cfg["sizes"]; errors = []; counts = {}
    for family in ("source8", "target4", "teacher4"):
        counts[family] = {}
        for split, rows in manifest.items():
            root = old / "cache" / "development" / family / split
            files = list(root.glob("*.pt")) if root.exists() else []
            counts[family][split] = len(files)
            if len(files) < expected[split]: errors.append(f"{family}/{split}: {len(files)} < {expected[split]}")
            required = {row["id"] for row in rows}
            if family == "source8": required |= {row["shuffle_id"] for row in rows}
            missing = sorted(sample_id for sample_id in required if not (root / f"{sample_id}.pt").is_file())
            if missing: errors.append(f"{family}/{split}: missing {len(missing)} required ids")
    if errors: return counts, errors

    for split, rows in manifest.items():
        for index, sample in enumerate(rows, 1):
            source = tensor_load(old / "cache" / "development" / "source8" / split / f"{sample['id']}.pt")
            target = tensor_load(old / "cache" / "development" / "target4" / split / f"{sample['id']}.pt")
            teacher = tensor_load(old / "cache" / "development" / "teacher4" / split / f"{sample['id']}.pt")
            prefix = f"{split}/{sample['id']}"
            if source.get("id") != sample["id"]: errors.append(f"{prefix}: source id")
            if target.get("id") != sample["id"]: errors.append(f"{prefix}: target id")
            if teacher.get("id") != sample["id"]: errors.append(f"{prefix}: teacher id")
            if int(source.get("context_length", -1)) != len(sample["context_input_ids"]): errors.append(f"{prefix}: context length")
            if list(teacher.get("gold", [])) != list(sample["answer_token_ids"]): errors.append(f"{prefix}: teacher gold")
            length = len(sample["context_input_ids"])
            if tuple(source["pre_key"].shape) != (36, length, 8, 128): errors.append(f"{prefix}: source pre_key shape")
            if tuple(source["value"].shape) != (36, length, 8, 128): errors.append(f"{prefix}: source value shape")
            if tuple(target["pre_key"].shape) != (36, 128, 8, 128): errors.append(f"{prefix}: target pre_key shape")
            if tuple(target["value"].shape) != (36, 128, 8, 128): errors.append(f"{prefix}: target value shape")
            position_shape = tuple(target["positions"].shape) if hasattr(target["positions"], "shape") else (len(target["positions"]),)
            if position_shape != (128,): errors.append(f"{prefix}: positions shape")
            if target["query_first"].shape[0] != 36 or target["query_first"].shape[-1] != 128: errors.append(f"{prefix}: query_first shape")
            if teacher["logits"].shape[0] != len(sample["answer_token_ids"]): errors.append(f"{prefix}: teacher logits length")
            if errors: break
        if errors: break
    return counts, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repair-missing", action="store_true")
    args = parser.parse_args(); cfg = load_json(args.config)
    work, old = Path(cfg["work_dir"]), Path(cfg["reusable_experiment_dir"])
    required = [
        old / "artifacts" / "manifest.json", old / "artifacts" / "protocol.json",
        old / "artifacts" / "development" / "scales.pt",
        old / "artifacts" / "development" / "linear_stage_a" / "best.pt",
    ]
    missing_assets = [str(path) for path in required if not path.is_file()]
    if missing_assets: raise RuntimeError(f"missing reusable assets: {missing_assets}")
    manifest = load_json(required[0])
    counts, errors = validate_cache(cfg, old, manifest)
    if errors and args.repair_missing:
        old_script = old / "assets.py"; old_config = old / "config.json"
        for action in ("source", "target", "teacher"):
            subprocess.run([sys.executable, "-u", str(old_script), "--config", str(old_config), "--mode", "development", action], cwd=old, check=True)
        counts, errors = validate_cache(cfg, old, manifest)
    if errors: raise RuntimeError("reusable cache audit failed: " + "; ".join(errors[:20]))

    scales = tensor_load(required[2]); checkpoint = tensor_load(required[3]); state = checkpoint.get("writer", checkpoint)
    scale_shapes = {name: list(value.shape) for name, value in scales.items()}
    valid_scale_shapes = ([36, 1024], [36, 8, 128])
    if any(shape not in valid_scale_shapes for shape in scale_shapes.values()):
        raise RuntimeError(f"scale shape mismatch: {scale_shapes}")
    checkpoint_shapes = {name: list(state[name].shape) for name in ("weight_k", "weight_v")}
    if checkpoint_shapes != {"weight_k": [36, 1024, 1024], "weight_v": [36, 1024, 1024]}:
        raise RuntimeError(f"linear checkpoint shape mismatch: {checkpoint_shapes}")

    ensure_link(work / "artifacts" / "manifest.json", required[0])
    ensure_link(work / "artifacts" / "protocol.json", required[1])
    for mode in ("smoke", "development"):
        for family in ("source8", "target4", "teacher4"):
            ensure_link(work / "cache" / mode / family, old / "cache" / "development" / family)
        ensure_link(work / "artifacts" / mode / "scales.pt", required[2])
    reused = work / "reused"; reused.mkdir(parents=True, exist_ok=True)
    copied_checkpoint = reused / "linear_stage_a_best.pt"
    if not copied_checkpoint.exists():
        shutil.copy2(required[3], copied_checkpoint); copied_checkpoint.chmod(0o444)

    from data import SYSTEM
    report = {
        "passed": True, "counts": counts,
        "manifest_sha256": sha256(required[0]), "protocol_sha256": sha256(required[1]),
        "prompt_sha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
        "model_config_sha256": {"qwen3_4b": sha256(Path(cfg["model_4b"]) / "config.json"), "qwen3_8b": sha256(Path(cfg["model_8b"]) / "config.json")},
        "tokenizers": {"qwen3_4b": tokenizer_hash(cfg["model_4b"]), "qwen3_8b": tokenizer_hash(cfg["model_8b"])},
        "scale_shapes": scale_shapes, "linear_checkpoint_shapes": checkpoint_shapes,
        "linear_checkpoint_sha256": sha256(copied_checkpoint),
        "links": {family: str((work / "cache" / "development" / family).resolve()) for family in ("source8", "target4", "teacher4")},
    }
    save_json(work / "artifacts" / "reusable_asset_audit.json", report)
    print(json.dumps({"passed": True, "counts": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
