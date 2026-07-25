import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import file_sha256, load_receiver, question_prompt, write_json
from p3e_a_common import NativeHeadwiseReader, native_memory_to
from p3e_n_common import ReceiverNativeConditionedCache


def load_reader(model, path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["reader_metadata"]
    reader = NativeHeadwiseReader(
        model,
        metadata["selected_layers"],
        metadata["rank"],
        metadata["gate_init"],
    ).to(device)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False)
    reader.eval()
    return reader, checkpoint


def trace_readout(model, tokenizer, reader, row, memory):
    encoded = tokenizer(
        question_prompt(tokenizer, row),
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    trace = {}
    with reader.inject(model, memory, trace):
        model(**encoded, use_cache=False, return_dict=True)
    readouts, attentions = [], []
    for local, layer in enumerate(reader.selected_layers):
        calls = trace.get(layer, [])
        if len(calls) != 1:
            raise RuntimeError(f"Expected one Reader call at layer {layer}")
        call = calls[0]
        delta = call["delta"][0, -1].detach().float()
        gate = reader.branches[local].gate.detach().float()
        if gate.abs() >= 1e-4:
            denominator = gate
        else:
            denominator = gate.new_tensor(-1e-4 if gate.item() < 0 else 1e-4)
        readouts.append((delta / denominator).cpu())
        attention = call["attention"][0, -1].detach().float()
        attentions.append(attention.reshape(32, attention.shape[-1]).cpu())
    return torch.stack(readouts).half(), torch.stack(attentions).half()


def state(payload, readout, attention):
    return {
        "readout": readout,
        "attention": attention,
        "metadata": dict(payload["metadata"]),
        "source_id": payload["source_id"],
        "source_answer": payload["source_answer"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = ReceiverNativeConditionedCache(args.memory)
    total = min(args.max_samples or len(cache), len(cache))
    model, tokenizer = load_receiver(args.model, device)
    reader, checkpoint = load_reader(model, args.reader, device)
    entries = []
    with torch.inference_mode():
        for index in tqdm(range(total), desc="p3e_n2_cache_reader_readouts"):
            correct = cache.correct(index)
            shuffled = cache.shuffled(index)
            conditions = {}
            for name, payload in (("correct", correct), ("shuffled", shuffled)):
                readout, attention = trace_readout(
                    model,
                    tokenizer,
                    reader,
                    correct["row"],
                    native_memory_to(payload, device),
                )
                conditions[name] = state(payload, readout, attention)
            filename = f"sample_{index:05d}.pt"
            torch.save(
                {"row": correct["row"], "conditions": conditions},
                output / filename,
            )
            entries.append({"id": correct["row"]["id"], "file": filename})
    result = {
        "status": "complete",
        "experiment": "P3-E-N2 frozen Reader C readout cache",
        "samples": total,
        "entries": entries,
        "readout_shape": "[16,2560]",
        "attention_shape": "[16,32,T]",
        "readout": "post-native-o_proj before scalar gate",
        "query_position": "last Question prompt token",
        "reader_frozen": True,
        "receiver_frozen": True,
        "reader_checkpoint": args.reader,
        "reader_checkpoint_sha256": file_sha256(args.reader),
    }
    write_json(output / "index.json", result)
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
