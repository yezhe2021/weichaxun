import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3d3_common import SELECTED_LAYERS, evidence_block, file_sha256, load_jsonl, write_json


def overlap_mask(offsets, spans):
    return [bool(end > start and any(end > left and start < right for left, right in spans)) for start, end in offsets]


class HeadwiseCapture:
    def __init__(self, model):
        self.model, self.states, self.handles = model, {}, []
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
        for handle in self.handles: handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--data", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--max-length", type=int, default=1024); parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(); output = Path(args.out); output.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    rows = load_jsonl(args.data)[:args.max_samples]; entries, lengths = [], []
    with HeadwiseCapture(model) as capture, torch.inference_mode():
        for index, row in enumerate(tqdm(rows, desc="p3e_a_qwen4_native_headwise_cache")):
            evidence = evidence_block(row)
            encoded = tokenizer(evidence, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=args.max_length, add_special_tokens=True)
            offsets = encoded.pop("offset_mapping")[0].tolist(); capture.states.clear()
            model(**{name: value.to(device) for name, value in encoded.items()}, use_cache=False)
            keys = torch.stack([capture.states[layer][0][0].float().cpu() for layer in SELECTED_LAYERS])
            values = torch.stack([capture.states[layer][1][0].float().cpu() for layer in SELECTED_LAYERS])
            if keys.shape[-2:] != (8, 128): raise RuntimeError(f"Unexpected Qwen3-4B Native KV shape: {tuple(keys.shape)}")
            valid = [end > start for start, end in offsets]
            support = overlap_mask(offsets, row["support_char_spans"]); answer = overlap_mask(offsets, row["answer_char_spans"])
            filename = f"sample_{index:05d}.pt"
            torch.save({"row": row, "evidence": evidence, "keys": keys.half(), "values": values.half(),
                        "metadata": {"offsets": offsets, "token_ids": encoded["input_ids"][0].tolist(), "valid_mask": valid,
                                     "support_token_mask": support, "answer_token_mask": answer, "selected_layers": SELECTED_LAYERS}}, output / filename)
            entries.append({"id": row["id"], "file": filename, "answer": row["answer"]}); lengths.append(len(offsets))
    result = {"status": "complete", "samples": len(entries), "entries": entries, "layers": 16,
              "original_layer_indices": SELECTED_LAYERS, "memory_shape": "[16,T,8,128]", "kv_heads": 8, "head_dim": 128,
              "max_tokens": max(lengths), "question_independent": True, "sender_input": "evidence_only", "pre_rope_keys": True,
              "native_values": True, "model": args.model, "model_config_sha256": file_sha256(Path(args.model) / "config.json")}
    write_json(output / "index.json", result); write_json(output / "SUCCESS.json", result)


if __name__ == "__main__": main()
