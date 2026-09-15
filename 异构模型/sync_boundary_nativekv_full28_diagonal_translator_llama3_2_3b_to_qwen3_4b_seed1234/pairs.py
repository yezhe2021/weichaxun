from pathlib import Path

import torch

from common import digest, read_json, run_root, save_json, save_tensor, tokenizer, log
from offsets import token_spans
from units import build_synchronized_units, select_units


def source_cfg(cfg): return read_json(Path(cfg["source_root"]) / "run_config.json")


def importance_cfg(cfg): return read_json(Path(cfg["importance_root"]) / "run_config.json")


def rows(cfg, split):
    payload = read_json(Path(cfg["source_root"]) / "manifests" / f"{split}.json")
    count = cfg[f"{split}_samples"]
    if payload["signature"] != source_cfg(cfg)["signature"] or len(payload["rows"]) < count:
        raise RuntimeError(f"Invalid source manifest: {split}")
    return payload["rows"][:count]


def native(cfg, family, split, row):
    payload = torch.load(Path(cfg["source_root"]) / "cache" / family / split / f"{row['id']}.pt",
                         map_location="cpu", weights_only=True)
    if payload["signature"] != source_cfg(cfg)["signature"] or payload["tokens_hash"] != digest(row["encoded"][family]):
        raise RuntimeError("Native cache mismatch")
    return payload


def importance(cfg, family, split, row):
    payload = torch.load(Path(cfg["importance_root"]) / "selection_cache" / family / split / f"{row['id']}.pt",
                         map_location="cpu", weights_only=True)
    if payload["signature"] != importance_cfg(cfg)["signature"]: raise RuntimeError("Importance cache mismatch")
    return payload["importance"].float()


def pair_path(cfg, split, row): return run_root(cfg) / "pair_cache" / split / f"{row['id']}.pt"


def prepare_pairs(cfg):
    toks = {family: tokenizer(path) for family, path in cfg["models"].items()}
    audit = {"selection": cfg["selection"], "alignment": "UTF-8 synchronized causal right boundary",
             "token0_excluded_from_writer": True, "splits": {}}
    for split in ("train", "validation", "test"):
        manifest, shortfall, duplicates = [], 0, []; selected_rows = rows(cfg, split)
        for number, row in enumerate(selected_rows, 1):
            path = pair_path(cfg, split, row)
            if path.exists():
                cached = torch.load(path, map_location="cpu", weights_only=True)
                if cached.get("signature") == cfg["signature"]:
                    manifest.append(cached["metadata"]); continue
            spans = {family: token_spans(toks[family], row["prefix_text"], row["encoded"][family]["prefix"])
                     for family in ("llama", "qwen")}
            units = build_synchronized_units(row["prefix_text"], spans["llama"], spans["qwen"])
            chosen = select_units(units, importance(cfg, "llama", split, row),
                                  "llama", "max", cfg["regions"], True)
            source_indices = [unit.llama_right for unit in chosen]
            target_indices = [unit.qwen_right for unit in chosen]
            unique = len(set(unit.unit_id for unit in chosen)); duplicates.append(32 - unique); shortfall += len(units) < 32
            source, target = native(cfg, "llama", split, row), native(cfg, "qwen", split, row)
            metadata = {"id": row["id"], "gold_index": row["gold_index"],
                        "units": [unit.json() for unit in chosen], "source_indices": source_indices,
                        "target_indices": target_indices, "unique_units": unique,
                        "qwen_suffix": row["encoded"]["qwen"]["suffix"],
                        "qwen_choice_ids": row["encoded"]["qwen"]["choice_ids"]}
            save_tensor(path, {"signature": cfg["signature"], "metadata": metadata,
                "source_k": source["k"][:, [0] + source_indices].contiguous(),
                "source_v": source["v"][:, [0] + source_indices].contiguous(),
                "target_k": target["k"][:, [0] + target_indices].contiguous(),
                "target_v": target["v"][:, [0] + target_indices].contiguous(),
                "qwen_full_native_logits": target["native_logits"]})
            manifest.append(metadata)
            if number % 32 == 0: log(f"Sync pair dataset {split}: {number}/{len(selected_rows)}")
        save_json(run_root(cfg) / "pair_manifests" / f"{split}.json", {"signature": cfg["signature"], "rows": manifest})
        audit["splits"][split] = {"count": len(selected_rows), "samples_below_32_units": shortfall,
            "mean_duplicate_entries": sum(duplicates) / max(len(duplicates), 1), "max_duplicate_entries": max(duplicates, default=0)}
    save_json(run_root(cfg) / "audit" / "pair_audit.json", audit); return audit


def load_pair(cfg, split, row):
    payload = torch.load(pair_path(cfg, split, row), map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]: raise RuntimeError("Sync pair signature mismatch")
    return payload


def audit_pairs(cfg): return prepare_pairs(cfg)
