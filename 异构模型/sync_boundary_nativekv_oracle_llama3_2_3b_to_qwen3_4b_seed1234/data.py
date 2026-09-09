from pathlib import Path

import torch

from common import digest, read_json


def source_cfg(cfg): return read_json(Path(cfg["source_root"]) / "run_config.json")


def importance_cfg(cfg): return read_json(Path(cfg["importance_root"]) / "run_config.json")


def rawanchor_cfg(cfg): return read_json(Path(cfg["rawanchor_root"]) / "run_config.json")


def rows(cfg):
    payload = read_json(Path(cfg["source_root"]) / "manifests" / "test.json")
    if payload["signature"] != source_cfg(cfg)["signature"] or len(payload["rows"]) < cfg["test_samples"]:
        raise RuntimeError("Invalid test manifest")
    return payload["rows"][:cfg["test_samples"]]


def native(cfg, family, row):
    path = Path(cfg["source_root"]) / "cache" / family / "test" / f"{row['id']}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != source_cfg(cfg)["signature"]:
        raise RuntimeError("Native cache signature mismatch")
    if payload["tokens_hash"] != digest(row["encoded"][family]):
        raise RuntimeError("Native token hash mismatch")
    return payload


def importance(cfg, family, row):
    path = Path(cfg["importance_root"]) / "selection_cache" / family / "test" / f"{row['id']}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != importance_cfg(cfg)["signature"]:
        raise RuntimeError("Importance cache signature mismatch")
    return payload["importance"].float()


def rawanchor(cfg, row):
    path = Path(cfg["rawanchor_root"]) / "pair_cache" / "test" / f"{row['id']}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != rawanchor_cfg(cfg)["signature"]:
        raise RuntimeError("RawAnchor cache signature mismatch")
    return payload
