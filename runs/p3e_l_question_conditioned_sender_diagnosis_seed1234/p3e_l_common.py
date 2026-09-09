from collections import OrderedDict
from pathlib import Path

import torch

from p3d3_common import read_json


CONDITIONS = (
    "neutral_prefix",
    "wrong_question",
    "correct_question",
    "correct_question_hard_shuffled_evidence",
)


class ConditionedNativeCache:
    def __init__(self, index_path, capacity=2):
        self.path = Path(index_path)
        self.root = self.path.parent
        self.index = read_json(index_path)
        self.entries = self.index["entries"]
        self.capacity = capacity
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index not in self.loaded:
            self.loaded[index] = torch.load(
                self.root / self.entries[index]["file"],
                map_location="cpu",
                weights_only=False,
            )
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]


def condition_payload(bundle, condition):
    state = bundle["conditions"][condition]
    payload = {
        "row": bundle["row"],
        "keys": state["keys"],
        "values": state["values"],
        "metadata": state["metadata"],
        "sender_condition": condition,
        "source_id": state["source_id"],
        "source_answer": state["source_answer"],
    }
    if payload["keys"].shape != payload["values"].shape:
        raise RuntimeError("Conditioned K/V shapes differ")
    if payload["keys"].ndim != 4 or payload["keys"].shape[-2:] != (8, 128):
        raise RuntimeError(f"Expected [16,T,8,128], got {tuple(payload['keys'].shape)}")
    return payload


def native_memory(payload, device, oracle_support=False):
    keys = payload["keys"].float().to(device)
    values = payload["values"].float().to(device)
    valid = torch.as_tensor(payload["metadata"]["valid_mask"], dtype=torch.bool, device=device)
    support = torch.as_tensor(
        payload["metadata"]["support_token_mask"], dtype=torch.bool, device=device
    )
    mask = valid & support if oracle_support else valid
    return {"keys": keys, "values": values, "mask": mask, "support_mask": support}

