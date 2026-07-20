import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3b_common import (
    SENDER_MODES,
    evidence_block,
    file_sha256,
    load_jsonl,
    sender_text,
    token_indices,
    token_span_targets,
    write_json,
)


class MultiLayerCapture:
    def __init__(self, model, evidence_indices):
        self.model = model
        self.indices = evidence_indices
        self.keys = [None] * len(model.model.layers)
        self.values = [None] * len(model.model.layers)
        self.hidden = [None] * len(model.model.layers)
        self.handles = []

    def __enter__(self):
        for index, layer in enumerate(self.model.model.layers):
            attention = layer.self_attn

            def attention_hook(module, args, kwargs, layer_index=index):
                states = kwargs.get("hidden_states", args[0] if args else None)
                if states is None:
                    raise RuntimeError("Attention hook did not receive hidden_states")
                shape = (*states.shape[:-1], -1, module.head_dim)
                keys = module.k_norm(module.k_proj(states).view(shape)).transpose(1, 2)
                values = module.v_proj(states).view(shape).transpose(1, 2)
                self.keys[layer_index] = keys[0, :, self.indices, :].transpose(0, 1).reshape(len(self.indices), -1).detach()
                self.values[layer_index] = values[0, :, self.indices, :].transpose(0, 1).reshape(len(self.indices), -1).detach()

            def layer_hook(module, args, output, layer_index=index):
                states = output[0] if isinstance(output, tuple) else output
                self.hidden[layer_index] = states[0, self.indices].detach()

            self.handles.append(attention.register_forward_pre_hook(attention_hook, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(layer_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def result(self):
        if any(value is None for group in (self.keys, self.values, self.hidden) for value in group):
            raise RuntimeError("Not all layer states were captured")
        return {
            "keys": torch.stack(self.keys).half().cpu().contiguous(),
            "values": torch.stack(self.values).half().cpu().contiguous(),
            "hidden": torch.stack(self.hidden).half().cpu().contiguous(),
        }


@torch.inference_mode()
def encode_question(model, tokenizer, question, device, max_length):
    encoded = tokenizer(f"QUESTION\n{question}", return_tensors="pt", add_special_tokens=True, truncation=False)
    if encoded.input_ids.shape[1] > max_length:
        raise RuntimeError("Question exceeds max length")
    captured = {}

    def hook(module, args, output):
        captured["state"] = output[0, -1].detach()

    handle = model.model.norm.register_forward_hook(hook)
    try:
        model(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()
    return captured["state"].half().cpu().contiguous()


@torch.inference_mode()
def encode_sender(model, tokenizer, row, mode, device, max_length):
    text, evidence_left, evidence_right = sender_text(row, mode)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
    )
    if encoded.input_ids.shape[1] > max_length:
        raise RuntimeError(f"{row['id']} exceeds {max_length} tokens in {mode}")
    full_offsets = encoded.offset_mapping[0].tolist()
    indices = token_indices(full_offsets, evidence_left, evidence_right)
    if not indices:
        raise RuntimeError(f"No evidence tokens for {row['id']} in {mode}")
    relative_offsets = [(max(0, full_offsets[i][0] - evidence_left), max(0, full_offsets[i][1] - evidence_left)) for i in indices]
    targets = token_span_targets(relative_offsets, row["answer_char_spans"])
    if not targets:
        raise RuntimeError(f"Answer span did not map to tokens for {row['id']} in {mode}")
    evidence = evidence_block(row)
    a_left = evidence.index(row["evidence_a"])
    a_right = a_left + len(row["evidence_a"])
    b_left = evidence.index(row["evidence_b"], a_right)
    b_right = b_left + len(row["evidence_b"])
    support_mask = torch.tensor(
        [any(start < right and end > left for left, right in ((a_left, a_right), (b_left, b_right))) for start, end in relative_offsets],
        dtype=torch.bool,
    )

    with MultiLayerCapture(model, indices) as capture:
        model(
            input_ids=encoded.input_ids.to(device),
            attention_mask=encoded.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
    states = capture.result()
    states.update(
        {
            "offsets": relative_offsets,
            "answer_token_spans": targets,
            "token_ids": encoded.input_ids[0, indices].cpu(),
            "valid_mask": torch.ones(len(indices), dtype=torch.bool),
            "support_token_mask": support_mask,
        }
    )
    return states, int(encoded.input_ids.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    if args.max_samples:
        rows = rows[: args.max_samples]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if len(model.model.layers) != 36:
        raise RuntimeError(f"Expected 36 sender layers, got {len(model.model.layers)}")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    entries, evidence_lengths, sender_lengths = [], [], {mode: [] for mode in SENDER_MODES}
    for index, row in enumerate(tqdm(rows, desc=f"p3b_cache_{output.name}")):
        question_state = encode_question(model, tokenizer, row["question"], device, args.max_length)
        modes = {}
        for mode in SENDER_MODES:
            modes[mode], length = encode_sender(model, tokenizer, row, mode, device, args.max_length)
            sender_lengths[mode].append(length)
        if modes["evidence_only"]["keys"].shape[0] != 36:
            raise RuntimeError("Layer axis drift")
        filename = f"sample_{index:05d}.pt"
        torch.save(
            {
                "row": row,
                "evidence": evidence_block(row),
                "question_state": question_state,
                "modes": modes,
            },
            output / filename,
        )
        entries.append({"id": row["id"], "file": filename, "answer": row["answer"], "type": row["type"]})
        evidence_lengths.extend([modes[mode]["keys"].shape[1] for mode in SENDER_MODES])

    metadata = {
        "status": "complete",
        "model": args.model,
        "model_sha256": file_sha256(Path(args.model) / "config.json"),
        "samples": len(entries),
        "entries": entries,
        "sender_modes": list(SENDER_MODES),
        "layers": 36,
        "kv_heads": 8,
        "head_dim": 128,
        "kv_flat_dim": 1024,
        "hidden_dim": int(model.config.hidden_size),
        "max_evidence_tokens": max(evidence_lengths),
        "max_sender_tokens": {mode: max(values) for mode, values in sender_lengths.items()},
        "question_in_evidence_only_sender": False,
    }
    write_json(output / "index.json", metadata)
    write_json(output / "SUCCESS.json", metadata)


if __name__ == "__main__":
    main()
