import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3d3_common import SELECTED_LAYERS, evidence_block, file_sha256, load_jsonl, write_json


def overlap_mask(offsets, spans):
    return [bool(end > start and any(end > left and start < right for left, right in spans)) for start, end in offsets]


class NativeCapture:
    def __init__(self, model, layers):
        self.model, self.layers, self.states, self.handles = model, list(layers), {}, []
    def __enter__(self):
        for layer_index in self.layers:
            attention = self.model.model.layers[layer_index].self_attn
            def hook(module, args, kwargs, layer_index=layer_index):
                hidden = args[0] if args else kwargs["hidden_states"]
                shape = (*hidden.shape[:-1], -1, module.head_dim)
                keys = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
                values = module.v_proj(hidden).view(shape).transpose(1, 2)
                self.states[layer_index] = (keys.detach(), values.detach())
            self.handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
        return self
    def __exit__(self, *args):
        for handle in self.handles: handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--data", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--max-length", type=int, default=1024); parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(); device = torch.device(args.device); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    rows = load_jsonl(args.data)[:args.max_samples]; entries, lengths = [], []
    with NativeCapture(model, SELECTED_LAYERS) as capture, torch.inference_mode():
        for index, row in enumerate(tqdm(rows, desc="p3d3_native_cache")):
            evidence = evidence_block(row)
            encoded = tokenizer(evidence, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=args.max_length, add_special_tokens=True)
            offsets = encoded.pop("offset_mapping")[0].tolist(); inputs = {name: value.to(device) for name, value in encoded.items()}
            capture.states.clear(); model(**inputs, use_cache=False)
            keys, values = [], []
            for layer_index in SELECTED_LAYERS:
                key, value = capture.states[layer_index]
                keys.append(key[0].transpose(0, 1).reshape(key.shape[2], -1).float().cpu())
                values.append(value[0].transpose(0, 1).reshape(value.shape[2], -1).float().cpu())
            valid = [end > start for start, end in offsets]
            support = overlap_mask(offsets, row["support_char_spans"]); answer = overlap_mask(offsets, row["answer_char_spans"])
            filename = f"sample_{index:05d}.pt"
            torch.save({"row": row, "evidence": evidence, "keys": torch.stack(keys).half(), "values": torch.stack(values).half(),
                        "metadata": {"offsets": offsets, "token_ids": encoded["input_ids"][0].tolist(), "valid_mask": valid,
                                     "support_token_mask": support, "answer_token_mask": answer, "selected_layers": SELECTED_LAYERS}}, output / filename)
            entries.append({"id": row["id"], "file": filename, "answer": row["answer"]}); lengths.append(len(offsets))
    result = {"status": "complete", "entries": entries, "samples": len(entries), "layers": 16, "original_layer_indices": SELECTED_LAYERS,
              "memory_dim": 1024, "max_tokens": max(lengths), "question_independent": True, "pre_rope_keys": True,
              "sender_input": "evidence_only", "model": args.model, "model_config_sha256": file_sha256(Path(args.model) / "config.json")}
    write_json(output / "index.json", result); write_json(output / "SUCCESS.json", result)


if __name__ == "__main__": main()
