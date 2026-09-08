from pathlib import Path

import torch

from anchors import build_anchors
from common import digest, read_json, run_root, save_json, save_tensor, tokenizer, log
from offsets import token_spans


def source_cfg(cfg): return read_json(Path(cfg["source_root"]) / "run_config.json")


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
    importance_cfg = read_json(Path(cfg["importance_root"]) / "run_config.json")
    payload = torch.load(Path(cfg["importance_root"]) / "selection_cache" / family / split / f"{row['id']}.pt",
                         map_location="cpu", weights_only=True)
    if payload["signature"] != importance_cfg["signature"]: raise RuntimeError("Importance cache mismatch")
    return payload["importance"].float()


def pair_path(cfg, split, row): return run_root(cfg) / "pair_cache" / split / f"{row['id']}.pt"


def prepare_pairs(cfg):
    toks = {family: tokenizer(path) for family, path in cfg["models"].items()}
    audit = {"id_match": True, "entries": 33, "pairing": "Llama-driven raw-text anchors"}
    for split in ("train", "validation", "test"):
        manifest = []; selected_rows = rows(cfg, split)
        for number, row in enumerate(selected_rows, 1):
            path = pair_path(cfg, split, row)
            if path.exists():
                cached = torch.load(path, map_location="cpu", weights_only=True)
                if cached.get("signature") == cfg["signature"]:
                    manifest.append(cached["metadata"]); continue
            spans = {family: token_spans(toks[family], row["prefix_text"], row["encoded"][family]["prefix"])
                     for family in ("llama", "qwen")}
            score = importance(cfg, "llama", split, row)
            anchors = build_anchors(row["prefix_text"], spans["llama"], spans["qwen"], score, cfg["regions"])
            source_indices = [0] + [anchor["source_index"] for anchor in anchors]
            target_indices = [0] + [anchor["target_index"] for anchor in anchors]
            source, target = native(cfg, "llama", split, row), native(cfg, "qwen", split, row)
            metadata = {"id": row["id"], "gold_index": row["gold_index"], "anchors": anchors,
                        "source_indices": source_indices, "target_indices": target_indices,
                        "qwen_suffix": row["encoded"]["qwen"]["suffix"],
                        "qwen_choice_ids": row["encoded"]["qwen"]["choice_ids"]}
            save_tensor(path, {"signature": cfg["signature"], "metadata": metadata,
                "source_k": source["k"][:, source_indices].contiguous(),
                "source_v": source["v"][:, source_indices].contiguous(),
                "target_k": target["k"][:, target_indices].contiguous(),
                "target_v": target["v"][:, target_indices].contiguous(),
                "qwen_full_native_logits": target["native_logits"]})
            manifest.append(metadata)
            if number % 32 == 0: log(f"pair dataset {split}: {number}/{len(selected_rows)}")
        save_json(run_root(cfg) / "pair_manifests" / f"{split}.json", {"signature": cfg["signature"], "rows": manifest})
    save_json(run_root(cfg) / "audit" / "pair_audit.json", audit)


def load_pair(cfg, split, row):
    payload = torch.load(pair_path(cfg, split, row), map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]: raise RuntimeError("Pair cache signature mismatch")
    return payload
