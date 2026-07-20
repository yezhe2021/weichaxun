import json
import random
from pathlib import Path

import torch
from transformers import Adafactor

from .core import PublicReader, PublicWriter


class HiddenCache:
    def __init__(self, index_path):
        self.index_path = Path(index_path)
        with self.index_path.open(encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.root = self.index_path.parent

    def __len__(self):
        return len(self.index["entries"])

    def load(self, index):
        return torch.load(self.root / self.index["entries"][index]["file"], map_location="cpu", weights_only=False)


def cached_memory(writer, cached, device, dtype=torch.float16):
    taps = tuple(value.unsqueeze(0).to(device=device, dtype=dtype) for value in cached["hidden_taps"])
    mask = cached["mask"].unsqueeze(0).to(device)
    return writer(taps, mask)


def build_reader(model, metadata=None):
    metadata = metadata or {
        "hidden_size": 2560, "active_layers": [3, 8, 12, 17, 21, 26, 30, 35],
        "query_heads": 32, "kv_heads": 8, "head_dim": 128, "max_gate": 1.0,
    }
    return PublicReader(**metadata)


def build_writer(hidden_size=2560):
    return PublicWriter(hidden_size=hidden_size, layers=8, kv_heads=8, head_dim=128)


def make_optimizer(parameters, name, lr, weight_decay):
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if name == "adafactor":
        return Adafactor(parameters, lr=lr, weight_decay=weight_decay, scale_parameter=False, relative_step=False)
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def save_checkpoint(path, writer, reader, args, epoch, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "epoch": epoch, "step": step, "args": vars(args),
        "writer": {name: value.detach().cpu() for name, value in writer.state_dict().items()},
        "writer_metadata": {"hidden_size": writer.hidden_size, "layers": writer.layer_count, "kv_heads": writer.kv_heads, "head_dim": writer.head_dim},
    }
    if reader is not None:
        payload["reader"] = {name: value.detach().cpu() for name, value in reader.state_dict().items()}
        payload["reader_metadata"] = reader.metadata()
    torch.save(payload, path)


def different_answer_partner(rows, index, seed):
    order = list(range(len(rows)))
    random.Random(seed + index).shuffle(order)
    answer = rows[index]["answer"].strip().casefold()
    return next(candidate for candidate in order if candidate != index and rows[candidate]["answer"].strip().casefold() != answer)


def assert_only_modules_have_grad(model, *modules):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen backbone received gradients")
    if not any(parameter.grad is not None for module in modules for parameter in module.parameters() if parameter.requires_grad):
        raise RuntimeError("No trainable Public-KV parameter received gradients")
