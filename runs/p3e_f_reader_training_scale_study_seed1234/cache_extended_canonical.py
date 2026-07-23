import argparse
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3d3_common import SELECTED_LAYERS, evidence_block
from p3e_c2_common import load_writer
from p3e_f_common import (read_json, read_jsonl, sha256_file, sha256_tensor,
                          prefix_digest, write_json)


def overlap_mask(offsets, spans):
    return [bool(end > start and any(end > left and start < right for left, right in spans))
            for start, end in offsets]


class NativeCapture:
    def __init__(self, model):
        self.model = model
        self.states = {}
        self.handles = []

    def __enter__(self):
        for layer_index in SELECTED_LAYERS:
            attention = self.model.model.layers[layer_index].self_attn

            def hook(module, args, kwargs, layer_index=layer_index):
                hidden = args[0] if args else kwargs["hidden_states"]
                shape = (*hidden.shape[:-1], -1, module.head_dim)
                keys = module.k_norm(module.k_proj(hidden).view(shape))
                values = module.v_proj(hidden).view(shape)
                self.states[layer_index] = (keys.detach(), values.detach())

            self.handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


def tensor_bundle_hash(payload):
    parts = {name: sha256_tensor(payload[name]) for name in ("keys", "values", "mask", "support_mask")}
    digest = hashlib.sha256()
    for name in sorted(parts):
        digest.update(name.encode())
        digest.update(parts[name].encode())
    return parts, digest.hexdigest()


def resolve_file(root, name):
    path = Path(name)
    return path if path.is_absolute() else root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--writer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--existing512-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.out)
    files = output / "files"
    files.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.data)
    if len(rows) != 2048:
        raise RuntimeError("Expected nested train2048 data")
    negative_map = read_json(args.negatives)["train2048"]

    existing_index_path = Path(args.existing512_index)
    existing_index = read_json(existing_index_path)
    existing_root = existing_index_path.parent
    entries = []
    for index, source_entry in enumerate(existing_index["entries"]):
        if index >= 512:
            break
        if source_entry["id"] != rows[index]["id"]:
            raise RuntimeError(f"Historical cache order mismatch at {index}")
        source_path = resolve_file(existing_root, source_entry["file"]).resolve()
        entries.append({
            "index": index, "id": rows[index]["id"], "file": str(source_path),
            "hard_negative_index": negative_map[index],
            "shape": source_entry["shape"], "dtype": source_entry["dtype"],
            "tensor_sha256": source_entry["tensor_sha256"], "reused_historical512": True,
        })

    pending = []
    for index in range(512, 2048):
        destination = files / f"sample_{index:05d}.pt"
        if destination.exists():
            payload = torch.load(destination, map_location="cpu", weights_only=False)
            if payload["id"] != rows[index]["id"]:
                raise RuntimeError(f"Existing cache ID mismatch at {index}")
            parts, combined = tensor_bundle_hash(payload)
            entries.append({"index": index, "id": payload["id"], "file": str(destination.resolve()),
                            "hard_negative_index": negative_map[index], "shape": list(payload["keys"].shape),
                            "dtype": str(payload["keys"].dtype), "tensor_components": parts,
                            "tensor_sha256": combined, "reused_historical512": False})
        else:
            pending.append(index)

    if pending:
        device = torch.device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, trust_remote_code=True, local_files_only=True
        ).to(device).eval()
        model.requires_grad_(False)
        writer, checkpoint = load_writer(args.writer, device)
        writer.requires_grad_(False)
        writer.eval()
        generated = {}
        with NativeCapture(model) as capture, torch.inference_mode():
            for index in tqdm(pending, desc="p3e_f_cache_c2_canonical"):
                row = rows[index]
                encoded = tokenizer(evidence_block(row), return_tensors="pt", return_offsets_mapping=True,
                                    truncation=False, add_special_tokens=True)
                offsets = encoded.pop("offset_mapping")[0].tolist()
                if len(offsets) > args.max_length:
                    raise RuntimeError(f"Evidence exceeds max length at {row['id']}: {len(offsets)}")
                inputs = {name: value.to(device) for name, value in encoded.items()}
                capture.states.clear()
                model(**inputs, use_cache=False)
                keys, values = [], []
                for layer_index in SELECTED_LAYERS:
                    key, value = capture.states[layer_index]
                    keys.append(key[0])
                    values.append(value[0])
                native_keys = torch.stack(keys)
                native_values = torch.stack(values)
                canonical_keys, canonical_values, _ = writer(native_keys, native_values)
                valid = torch.tensor([end > start for start, end in offsets], dtype=torch.bool)
                support = torch.tensor(overlap_mask(offsets, row["support_char_spans"]), dtype=torch.bool)
                payload = {"id": row["id"], "keys": canonical_keys.half().cpu(),
                           "values": canonical_values.half().cpu(), "mask": valid, "support_mask": support}
                destination = files / f"sample_{index:05d}.pt"
                torch.save(payload, destination)
                parts, combined = tensor_bundle_hash(payload)
                generated[index] = {"index": index, "id": row["id"], "file": str(destination.resolve()),
                                    "hard_negative_index": negative_map[index], "shape": list(payload["keys"].shape),
                                    "dtype": str(payload["keys"].dtype), "tensor_components": parts,
                                    "tensor_sha256": combined, "reused_historical512": False}
        del writer, model
        torch.cuda.empty_cache()
        entries = entries[:512] + [generated[index] for index in range(512, 2048)]

    entries.sort(key=lambda item: item["index"])
    if len(entries) != 2048:
        raise RuntimeError(f"Expected 2048 cache entries, got {len(entries)}")
    result = {
        "status": "complete", "samples": 2048, "format": "[16,T,16,128]",
        "entries": entries, "prefix_aggregate_sha256": {
            str(size): prefix_digest(entries, size) for size in (512, 1024, 2048)
        },
        "writer": args.writer, "writer_sha256": sha256_file(args.writer),
        "existing512_index": args.existing512_index, "data": args.data,
        "sender_loaded_during_reader_training": False,
    }
    write_json(output / "index.json", result)
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
