import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def sha256_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class CanonicalCache:
    def __init__(self, index_path, data_path, capacity=2):
        self.index_path = Path(index_path)
        self.root = self.index_path.parent
        self.index = read_json(index_path)
        self.rows = read_jsonl(data_path)
        all_entries = self.index["entries"]
        if len(self.rows) > len(all_entries):
            raise RuntimeError("Data file has more rows than the Canonical index")
        # Scale-study datasets are strict prefixes of one shared 2048-entry cache.
        self.entries = all_entries[:len(self.rows)]
        for index, entry in enumerate(self.entries):
            if entry["id"] != self.rows[index]["id"]:
                raise RuntimeError(f"Canonical/data order mismatch at {index}")
        self.capacity = int(capacity)
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def _path(self, entry):
        path = Path(entry["file"])
        return path if path.is_absolute() else self.root / path

    def load(self, index):
        if index not in self.loaded:
            payload = torch.load(self._path(self.entries[index]), map_location="cpu", weights_only=False)
            payload["row"] = self.rows[index]
            self.loaded[index] = payload
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]


def memory_to(payload, device, oracle_support=False):
    keys = payload["keys"].float().to(device)
    values = payload["values"].float().to(device)
    valid = payload["mask"].to(device=device, dtype=torch.bool)
    support = payload["support_mask"].to(device=device, dtype=torch.bool)
    if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 16 or keys.shape[-2:] != (16, 128):
        raise RuntimeError(f"Expected Canonical K/V [16,T,16,128], got {tuple(keys.shape)}")
    if valid.numel() != keys.shape[1] or support.numel() != keys.shape[1]:
        raise RuntimeError("Canonical token mask mismatch")
    mask = valid & support if oracle_support else valid
    if not mask.any():
        raise RuntimeError("Canonical memory mask is empty")
    return {"keys": keys, "values": values, "mask": mask, "support_mask": support}


def prefix_digest(entries, count):
    digest = hashlib.sha256()
    for entry in entries[:count]:
        digest.update(entry["id"].encode())
        digest.update(entry["tensor_sha256"].encode())
    return digest.hexdigest()
