import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch


class LazyPairCache:
    def __init__(self, index_path, capacity=3):
        self.index_path = str(index_path)
        with open(index_path, encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.root = Path(index_path).parent
        self.entries = self.index["pair_files"]
        self.capacity = capacity
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index in self.loaded:
            self.loaded.move_to_end(index)
            return self.loaded[index]
        payload = torch.load(
            self.root / self.entries[index]["file"], map_location="cpu", weights_only=False
        )
        pair = {example["variant"]: example for example in payload["examples"]}
        self.loaded[index] = pair
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
        return pair


def pair_id_hash(pair_ids):
    return hashlib.sha256("\n".join(pair_ids).encode("utf-8")).hexdigest()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def verify_manifest_cache(manifest, split, cache):
    records = manifest[split]
    for record in records:
        index = int(record["index"])
        if cache.entries[index]["pair_id"] != record["pair_id"]:
            raise ValueError(
                f"Manifest/cache mismatch for {split}: {record['pair_id']} != "
                f"{cache.entries[index]['pair_id']}"
            )


__all__ = [
    "LazyPairCache",
    "load_manifest",
    "pair_id_hash",
    "verify_manifest_cache",
    "write_jsonl",
]
