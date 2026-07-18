import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from canonical_modules import decoder_layers, full_attention_layers
from p2i_common import (
    LazyPairCache,
    full_text_prefixed_prompt,
    load_receiver,
    pack_answer,
    parse_dtype,
    resolve_device,
    student_prefixed_prompt,
)


def load_native_reader(root, model, checkpoint_path, device):
    sys.path.insert(0, str(Path(root).resolve()))
    from p2a_common import NativeKVExternalReader

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    reader = NativeKVExternalReader(
        model,
        max_gate=float(train_args["max_gate"]),
        gate_init=float(train_args["gate_init"]),
        query_rank=int(train_args["query_rank"]),
        output_rank=int(train_args["output_rank"]),
    ).to(device).eval()
    reader.load_state_dict(checkpoint["adapter"])
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    return reader


def sparse_logits(logits, labels, top_k):
    positions = torch.nonzero(labels[0] >= 0, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("No supervised answer position")
    first = int(positions[0])
    prediction_position = max(0, first - 1)
    values, indices = logits[0, prediction_position].float().topk(top_k)
    return {
        "position": prediction_position,
        "indices": indices.detach().cpu(),
        "values": values.detach().half().cpu(),
    }


def register_hidden_capture(model, selected):
    captured = {}
    handles = []
    for layer_index in selected:
        layer = decoder_layers(model)[layer_index]

        def hook(module, args, kwargs, layer_index=layer_index):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            captured[str(layer_index)] = hidden[:, -1].detach().half().cpu()

        handles.append(layer.register_forward_pre_hook(hook, with_kwargs=True))
    return captured, handles


@torch.inference_mode()
def question_only_states(model, tokenizer, row, device):
    encoded = tokenizer(
        student_prefixed_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False
    )
    captured, handles = register_hidden_capture(model, full_attention_layers(model))
    try:
        model(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    return captured


@torch.inference_mode()
def full_text_teacher(model, tokenizer, row, max_length, top_k, device):
    ids, mask, labels = pack_answer(
        tokenizer, full_text_prefixed_prompt(tokenizer, row), row["answer"], max_length, device
    )
    captured, handles = register_hidden_capture(model, full_attention_layers(model))
    try:
        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "answer_nll": float(output.loss.float().cpu()),
        "layer_deltas": {},
        "first_answer_topk": sparse_logits(output.logits, labels, top_k),
        "teacher_layer_states": captured,
        "question_only_layer_states": question_only_states(model, tokenizer, row, device),
        "teacher_kind": "full_text",
    }


@torch.inference_mode()
def native_teacher(model, tokenizer, reader, row, memory, max_length, top_k, device, dtype):
    from p2a_common import memory_to

    ids, mask, labels = pack_answer(
        tokenizer, student_prefixed_prompt(tokenizer, row), row["answer"], max_length, device
    )
    diagnostics = {"_capture_vectors": True, "_capture_training_tensors": True}
    captured, handles = register_hidden_capture(model, full_attention_layers(model))
    try:
        with reader.inject(model, memory_to(memory, device, dtype), diagnostics):
            output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    deltas = {
        layer: values["delta_vector"].half().cpu()
        for layer, values in diagnostics.items()
        if str(layer).isdigit() and "delta_vector" in values
    }
    routes = {
        layer: values["route_tensor"].detach().half().cpu()
        for layer, values in diagnostics.items()
        if str(layer).isdigit() and "route_tensor" in values
    }
    readouts = {
        layer: values["readout_vector"].detach().half().cpu()
        for layer, values in diagnostics.items()
        if str(layer).isdigit() and "readout_vector" in values
    }
    target_mass = {
        layer: float(values["target_mass_tensor"].detach().cpu())
        for layer, values in diagnostics.items()
        if str(layer).isdigit() and "target_mass_tensor" in values
    }
    return {
        "answer_nll": float(output.loss.float().cpu()),
        "layer_deltas": deltas,
        "layer_routes": routes,
        "layer_readouts": readouts,
        "target_attention_mass": target_mass,
        "first_answer_topk": sparse_logits(output.logits, labels, top_k),
        "teacher_layer_states": captured,
        "question_only_layer_states": question_only_states(model, tokenizer, row, device),
        "teacher_kind": "native_reader",
    }


def main():
    parser = argparse.ArgumentParser(description="Cache receiver-local functional teachers for P2-I")
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--rows-index", required=True, help="Pair-aligned cache supplying row text")
    parser.add_argument("--teacher-kind", choices=("full_text", "native_reader"), required=True)
    parser.add_argument("--native-index")
    parser.add_argument("--native-reader-checkpoint")
    parser.add_argument("--native-reader-root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    if args.teacher_kind == "native_reader" and not all(
        (args.native_index, args.native_reader_checkpoint, args.native_reader_root)
    ):
        raise ValueError("Native teacher requires index, checkpoint, and code root")
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    rows = LazyPairCache(args.rows_index, capacity=1)
    native = LazyPairCache(args.native_index, capacity=1) if args.native_index else None
    if native is not None:
        if len(native) != len(rows):
            raise ValueError("Native teacher and row caches differ in length")
        if [x["pair_id"] for x in native.entries] != [x["pair_id"] for x in rows.entries]:
            raise ValueError("Native teacher and row caches are not aligned")
    count = min(len(rows), args.max_pairs) if args.max_pairs > 0 else len(rows)
    model, tokenizer = load_receiver(args.receiver_model, device, dtype)
    reader = None
    if args.teacher_kind == "native_reader":
        reader = load_native_reader(
            args.native_reader_root, model, args.native_reader_checkpoint, device
        )

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    pair_files = []
    for index in tqdm(range(count), desc=f"cache_{args.teacher_kind}_teacher"):
        pair = rows.load(index)
        native_pair = native.load(index) if native is not None else None
        examples = []
        for variant in ("base", "counterfactual"):
            row = pair[variant]
            if reader is None:
                teacher = full_text_teacher(model, tokenizer, row, args.max_length, args.top_k, device)
            else:
                teacher = native_teacher(
                    model, tokenizer, reader, row, native_pair[variant]["memory"],
                    args.max_length, args.top_k, device, dtype,
                )
            examples.append(
                {
                    "pair_id": row["pair_id"],
                    "id": row["id"],
                    "variant": variant,
                    "answer": row["answer"],
                    "teacher": teacher,
                }
            )
        filename = f"pair_{index:05d}.pt"
        torch.save({"pair_id": examples[0]["pair_id"], "examples": examples}, output / filename)
        pair_files.append(
            {
                "pair_id": examples[0]["pair_id"],
                "file": filename,
                "base_answer": examples[0]["answer"],
                "counterfactual_answer": examples[1]["answer"],
            }
        )

    metadata = {
        "format_version": 1,
        "teacher_kind": args.teacher_kind,
        "receiver_model": args.receiver_model,
        "rows_index": args.rows_index,
        "native_index": args.native_index,
        "native_reader_checkpoint": args.native_reader_checkpoint,
        "pairs": count,
        "pair_files": pair_files,
    }
    with open(output / "index.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    with open(output / "CACHE_SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump({"status": "complete", **metadata}, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
