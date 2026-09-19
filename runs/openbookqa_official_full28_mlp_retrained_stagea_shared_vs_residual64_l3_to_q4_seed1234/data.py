from pathlib import Path

import torch

from common import digest, read_json, run_root, save_json


def asset_run(cfg):
    return Path(cfg["asset_experiment"]) / "runs" / "study"


def asset_config(cfg):
    return read_json(asset_run(cfg) / "run_config.json")


def validate_assets(cfg):
    source = asset_config(cfg)
    fields = ("models", "dataset", "regions", "max_body_tokens", "protocol", "split_policy")
    mismatches = {field: {"expected": cfg.get(field), "asset": source.get(field)}
                  for field in fields if cfg.get(field) != source.get(field)}
    if mismatches:
        raise RuntimeError(f"Asset protocol mismatch: {mismatches}")
    return source


def manifest_rows(cfg, split):
    source = validate_assets(cfg)
    payload = read_json(asset_run(cfg) / "manifests" / f"{split}.json")
    if payload.get("signature") != source["signature"]:
        raise RuntimeError("Asset manifest signature mismatch")
    rows = payload["rows"]
    if len(rows) < cfg[f"{split}_samples"]:
        raise RuntimeError(f"Asset {split} count mismatch")
    return rows[:cfg[f"{split}_samples"]]


def _load(cfg, path, label):
    source = validate_assets(cfg)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("signature") != source["signature"]:
        raise RuntimeError(f"{label} asset signature mismatch")
    return payload


def load_source(cfg, split, row):
    return _load(cfg, asset_run(cfg) / "source_selected_cache" / split / f"{row['id']}.pt", "source")


def load_pair(cfg, split, row):
    return _load(cfg, asset_run(cfg) / "pair_cache" / split / f"{row['id']}.pt", "pair")


def load_teacher(cfg, split, row):
    return _load(cfg, asset_run(cfg) / "oracle_teacher" / split / f"{row['id']}.pt", "teacher")


def audit_assets(cfg):
    source = validate_assets(cfg)
    rows = {split: manifest_rows(cfg, split) for split in ("train", "validation", "test")}
    ids = {split: {row["id"] for row in values} for split, values in rows.items()}
    overlap = {
        "train_validation": sorted(ids["train"] & ids["validation"]),
        "train_test": sorted(ids["train"] & ids["test"]),
        "validation_test": sorted(ids["validation"] & ids["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Official split overlap: {overlap}")
    result = {
        "status": "passed",
        "asset_experiment": cfg["asset_experiment"],
        "asset_signature": source["signature"],
        "protocol": cfg["protocol"],
        "counts": {split: len(values) for split, values in rows.items()},
        "ids_sha256": {split: digest([row["id"] for row in values]) for split, values in rows.items()},
        "official_splits_disjoint": True,
        "stage_a_reuse": False,
        "cache_reuse_only": True,
    }
    save_json(run_root(cfg) / "audit" / "asset_audit.json", result)
    return result
