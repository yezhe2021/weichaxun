from pathlib import Path

import torch

from common import read_json, run_root, save_json, log


def source_cfg(cfg):
    return read_json(Path(cfg["source_root"]) / "run_config.json")


def pair_source_cfg(cfg):
    return read_json(Path(cfg["pair_source_root"]) / "run_config.json")


def rows(cfg, split):
    payload = read_json(Path(cfg["source_root"]) / "manifests" / f"{split}.json")
    count = cfg[f"{split}_samples"]
    if payload["signature"] != source_cfg(cfg)["signature"] or len(payload["rows"]) < count:
        raise RuntimeError(f"Invalid source manifest: {split}")
    return payload["rows"][:count]


def pair_path(cfg, split, row):
    return Path(cfg["pair_source_root"]) / "pair_cache" / split / f"{row['id']}.pt"


def load_pair(cfg, split, row):
    payload = torch.load(pair_path(cfg, split, row), map_location="cpu", weights_only=True)
    if payload["signature"] != pair_source_cfg(cfg)["signature"]:
        raise RuntimeError("Reused pair cache signature mismatch")
    for name, layers in (("source_k", 28), ("source_v", 28), ("target_k", 36), ("target_v", 36)):
        if tuple(payload[name].shape) != (layers, 33, 8, 128):
            raise RuntimeError(f"Bad {name} shape: {tuple(payload[name].shape)}")
    return payload


def audit_pairs(cfg):
    report = {
        "pair_source_root": cfg["pair_source_root"],
        "reuse_without_copy": True,
        "source_schema": "[layers,33,8,128] = token0 + 32 content anchors",
        "translator_schema": "[layers,32,8,128] content anchors only",
        "receiver_schema": "Qwen Native token0 + 32 translated/native anchors; suffix starts at 33",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        selected = rows(cfg, split)
        for number, row in enumerate(selected, 1):
            if not pair_path(cfg, split, row).exists():
                raise FileNotFoundError(pair_path(cfg, split, row))
            if number == 1 or number == len(selected):
                load_pair(cfg, split, row)
        report["splits"][split] = {"count": len(selected), "first_id": selected[0]["id"], "last_id": selected[-1]["id"]}
        log(f"Pair reuse audit {split}: {len(selected)} files present")
    save_json(run_root(cfg) / "audit" / "pair_reuse_audit.json", report)
    return report
